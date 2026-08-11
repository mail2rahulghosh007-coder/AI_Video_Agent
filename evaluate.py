# evaluate.py
# Fixed, dependency-free evaluation for the whole pipeline:
#   1. RAG Q&A  -> faithfulness (LLM-as-judge) + answer similarity (embeddings)
#   2. Summary  -> LLM-judge rubric score (completeness / conciseness / accuracy)
#   3. Extraction (action items / decisions / questions) -> LLM-judge rubric score
#
# WHY THIS VERSION IS DIFFERENT FROM YOUR evaluate_rag.py / evaluate_rag_custom.py:
#   - No reliance on `rag_chain.first["context"]` or `.steps__` -- both fail
#     because RunnableParallel isn't subscriptable that way. We build the
#     retriever directly from core.vector_store instead.
#   - No RAGAS -- avoids the langchain_community/langchain_openai dependency
#     conflict you already ran into.
#   - Evaluates summary + action items + decisions + questions too, not just RAG.
#
# HOW TO USE:
#   1. Run your real pipeline once on a short (2-5 min) video you know well.
#   2. Save the transcript it produces to a file, e.g. transcripts/sample1.txt
#      (add a `print`/file-write in main.py after transcribe_all() if you
#      don't already save it -- see note at bottom of this file).
#   3. Fill in TRANSCRIPT_PATH and EVAL_QUESTIONS below with REAL answers
#      you know are correct (by reading the transcript yourself).
#   4. Run: python evaluate.py

import os
import json
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util

from core.vector_store import build_vector_store, get_retriever
from core.rag_engine import ask_question
from core.summarize import summarize
from core.extractor import extract_action_items, extract_key_decision, extract_question

load_dotenv()

# ---------------------------------------------------------------------------
# STEP 1: give it ONE video source. That's it -- no manual transcript file,
# no manually attaching multiple links. The script runs your real pipeline
# itself and evaluates the result.
#
# TO EVALUATE MULTIPLE VIDEOS: change VIDEO_LABEL for each run (e.g. "v1",
# "v2", "meeting_sample", "hinglish_sample") -- each gets its own cached
# transcript and its own row in results/summary.json, so you can build up
# an aggregate picture across several videos instead of judging the
# pipeline off just one.
# ---------------------------------------------------------------------------
SOURCE = "https://www.youtube.com/watch?v=2FZEznNC-Fs"   # <-- change this
LANGUAGE = "english"
VIDEO_LABEL = "v3"   # <-- change this each time you evaluate a new video

# ---------------------------------------------------------------------------
# RETRIEVAL EXPERIMENT KNOBS
# Keep VIDEO_LABEL fixed (same transcript) and change ONLY these between
# runs, one at a time, to see which setting actually improves faithfulness.
# EXPERIMENT_LABEL goes into the results filename so runs don't overwrite
# each other -- you'll end up with a clean comparison table.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500       # try: 500 (baseline) -> 800 -> 1000
CHUNK_OVERLAP = 50     # try: 50 (baseline) -> 100
RETRIEVAL_K = 4        # try: 4 (baseline) -> 6 -> 8
EXPERIMENT_LABEL = f"cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}_k{RETRIEVAL_K}"

CACHED_TRANSCRIPT_PATH = f"transcripts/{VIDEO_LABEL}.txt"
RESULTS_PATH = f"evaluation_results_{VIDEO_LABEL}_{EXPERIMENT_LABEL}.json"
SUMMARY_LOG_PATH = "evaluation_summary.json"  # running log across all videos

# OPTIONAL ground truth answers, ONLY needed for the "answer similarity"
# metric. This is the one piece that genuinely can't be automated -- it's
# your own known-correct answer, which is what similarity is measured
# against. Leave the list EMPTY to skip that metric entirely and rely on
# faithfulness only (fully automatic, no manual work).
MANUAL_GROUND_TRUTH = {
    "Why did senior doctors compare social media to smoking?": "Not because of the same physical health impact, but because they believe it's a serious health issue that needs to be taken seriously — similar to how attitudes changed about smoking and seatbelts.",
    "What does 'behind the curve' mean in this context?": "It means reacting or acting more slowly than others to a change or trend — used to describe the UK PM being slow on a social media ban for under-16s.",
    "What measures is the UK government considering besides a full ban?": "Curfews on app usage at night, stronger age checks, and restricting features like autoplay and infinite scroll.",
}

N_AUTO_QUESTIONS = 5  # how many questions to auto-generate from the transcript

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a meeting transcript, generate {n} diverse, specific questions "
     "a user might ask a meeting assistant about it (facts, decisions, "
     "action items, deadlines, opinions expressed). Return ONLY a JSON "
     'array of strings, e.g. ["question 1", "question 2", ...].'),
    ("human", "TRANSCRIPT:\n{transcript}")
])


def generate_eval_questions(transcript: str, n: int = N_AUTO_QUESTIONS) -> list:
    chain = QUESTION_GEN_PROMPT | get_judge_llm() | StrOutputParser()
    raw = chain.invoke({"transcript": transcript[:4000], "n": n})
    try:
        questions = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        print(f"Could not parse auto-generated questions, got: {raw}")
        questions = []
    return [
        {"question": q, "ground_truth": MANUAL_GROUND_TRUTH.get(q)}
        for q in questions
    ]


def get_eval_questions(transcript: str) -> list:
    # WHY THIS CHANGED: previously questions were auto-generated by an LLM
    # every run, so MANUAL_GROUND_TRUTH (keyed by exact question text)
    # almost never matched -- the LLM phrases things slightly differently
    # each time, so answer_similarity stayed None even when you'd filled
    # in ground truth. Fix: if you've written ground-truth questions, use
    # those EXACT questions (guarantees a match). Only auto-generate when
    # MANUAL_GROUND_TRUTH is empty.
    if MANUAL_GROUND_TRUTH:
        print(f"Using {len(MANUAL_GROUND_TRUTH)} manually written ground-truth questions.")
        return [{"question": q, "ground_truth": gt} for q, gt in MANUAL_GROUND_TRUTH.items()]
    return generate_eval_questions(transcript)


def get_transcript() -> str:
    if os.path.exists(CACHED_TRANSCRIPT_PATH):
        print(f"Using cached transcript at {CACHED_TRANSCRIPT_PATH}")
        with open(CACHED_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()

    if "REPLACE_ME" in SOURCE:
        raise RuntimeError(
            "Set SOURCE to a real video URL/path (or point CACHED_TRANSCRIPT_PATH "
            "at a transcript you already have)."
        )

    print(f"No cached transcript found. Running pipeline on: {SOURCE}")
    from utils.audio_processor import process_input
    from core.transcriber import transcribe_all

    chunks = process_input(SOURCE)
    transcript = transcribe_all(chunks, language=LANGUAGE)

    os.makedirs(os.path.dirname(CACHED_TRANSCRIPT_PATH), exist_ok=True)
    with open(CACHED_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Saved transcript to {CACHED_TRANSCRIPT_PATH} for reuse next time.")
    return transcript

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict evaluator. Given a CONTEXT and an ANSWER, determine "
     "if the ANSWER is fully supported by the CONTEXT, with no invented "
     "information. Respond with ONLY a JSON object: "
     '{{"score": <0.0 to 1.0>, "reason": "<one sentence explanation>"}}. '
     "Score 1.0 = fully supported, 0.0 = completely unsupported/hallucinated."),
    ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}")
])

RUBRIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict evaluator of a meeting-assistant AI's output. Given the "
     "ORIGINAL TRANSCRIPT and the AI's OUTPUT (a {kind}), score it 1-5 on each "
     "of: completeness (did it capture everything relevant?), accuracy (is "
     "anything wrong or invented?), and conciseness (is it free of filler?). "
     'Respond with ONLY JSON: {{"completeness": <1-5>, "accuracy": <1-5>, '
     '"conciseness": <1-5>, "reason": "<one sentence>"}}.'),
    ("human", "TRANSCRIPT:\n{transcript}\n\nAI OUTPUT:\n{output}")
])


def get_judge_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.0,  # deterministic for evaluation
    )


def _parse_json(raw: str) -> dict:
    raw = _strip_code_fence(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": f"Could not parse judge response: {raw}"}


def _strip_code_fence(raw: str) -> str:
    # WHY: Mistral often wraps JSON answers in a markdown code fence
    # (```json ... ```) even when told "respond with ONLY JSON". Both
    # _parse_json and generate_eval_questions need this, so it's shared.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]        # take content between first pair of fences
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def score_faithfulness(context: str, answer: str) -> dict:
    chain = FAITHFULNESS_PROMPT | get_judge_llm() | StrOutputParser()
    return _parse_json(chain.invoke({"context": context, "answer": answer}))


def score_rubric(kind: str, transcript: str, output: str) -> dict:
    chain = RUBRIC_PROMPT | get_judge_llm() | StrOutputParser()
    return _parse_json(chain.invoke({"kind": kind, "transcript": transcript, "output": output}))


def evaluate_rag(transcript: str, embedder) -> list:
    # Build the vector store ONCE and reuse it for both the retriever
    # (used here to inspect retrieved chunks) and the RAG chain -- do NOT
    # build it twice, or the second build_vector_store() call will delete
    # the collection the first one just created (see rag_engine.py note).
    vector_store = build_vector_store(transcript, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    retriever = get_retriever(vector_store, k=RETRIEVAL_K)

    from core.rag_engine import build_rag_chain
    rag_chain = build_rag_chain(vector_store=vector_store, k=RETRIEVAL_K)

    eval_questions = get_eval_questions(transcript)
    if not eval_questions:
        print("No eval questions available -- skipping RAG evaluation.")
        return []

    results = []
    for item in eval_questions:
        question, ground_truth = item["question"], item.get("ground_truth")
        print(f"\n[RAG] Asking: {question}")
        answer = ask_question(rag_chain, question)

        retrieved_docs = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Fully automatic -- no ground truth needed.
        faithfulness_result = score_faithfulness(context_text, answer)

        entry = {
            "question": question,
            "generated_answer": answer,
            "faithfulness_score": faithfulness_result.get("score"),
            "faithfulness_reason": faithfulness_result.get("reason"),
        }
        print(f"  Faithfulness: {faithfulness_result.get('score')} ({faithfulness_result.get('reason')})")

        # Only computed if you filled in MANUAL_GROUND_TRUTH for this question.
        if ground_truth:
            emb_answer = embedder.encode(answer, convert_to_tensor=True)
            emb_truth = embedder.encode(ground_truth, convert_to_tensor=True)
            similarity = float(util.cos_sim(emb_answer, emb_truth)[0][0])
            entry["ground_truth"] = ground_truth
            entry["answer_similarity"] = round(similarity, 3)
            print(f"  Answer similarity: {similarity:.3f}")

        results.append(entry)

    return results


def evaluate_pipeline_outputs(transcript: str) -> dict:
    outputs = {
        "summary": summarize(transcript),
        "action items": extract_action_items(transcript),
        "key decisions": extract_key_decision(transcript),
        "open questions": extract_question(transcript),
    }
    scored = {}
    for kind, output in outputs.items():
        print(f"\n[{kind}] scoring...")
        result = score_rubric(kind, transcript, output)
        result["output"] = output
        scored[kind] = result
        print(f"  {result}")
    return scored


def run_evaluation():
    transcript = get_transcript()
    if not transcript:
        print("ERROR: got an empty transcript.")
        return

    print("Loading local embedding model for similarity scoring...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    rag_results = evaluate_rag(transcript, embedder)
    output_scores = evaluate_pipeline_outputs(transcript)

    if not rag_results:
        print("\n(No RAG results to summarize.)")
        return

    valid_faith = [r["faithfulness_score"] for r in rag_results if isinstance(r.get("faithfulness_score"), (int, float))]
    avg_faith = sum(valid_faith) / len(valid_faith) if valid_faith else None

    sims = [r["answer_similarity"] for r in rag_results if "answer_similarity" in r]
    avg_sim = sum(sims) / len(sims) if sims else None

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"RAG avg faithfulness:       {avg_faith}")
    print(f"RAG avg answer similarity:  {avg_sim}")
    for kind, result in output_scores.items():
        print(f"{kind:15s} completeness={result.get('completeness')} "
              f"accuracy={result.get('accuracy')} conciseness={result.get('conciseness')}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"rag": rag_results, "outputs": output_scores}, f, indent=2)
    print(f"\nSaved detailed results to {RESULTS_PATH}")

    # Append this video's summary numbers to a running cross-video log, so
    # after evaluating 3-4 videos you have one file with the aggregate
    # picture instead of separate disconnected JSON dumps.
    log_entry = {
        "video_label": VIDEO_LABEL,
        "experiment": EXPERIMENT_LABEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_k": RETRIEVAL_K,
        "source": SOURCE,
        "rag_avg_faithfulness": avg_faith,
        "rag_avg_answer_similarity": avg_sim,
        "n_rag_questions": len(rag_results),
        "output_scores": {
            kind: {k: v for k, v in result.items() if k != "output"}
            for kind, result in output_scores.items()
        },
    }
    log = []
    if os.path.exists(SUMMARY_LOG_PATH):
        with open(SUMMARY_LOG_PATH, "r") as f:
            log = json.load(f)
    log = [e for e in log if not (e.get("video_label") == VIDEO_LABEL and e.get("experiment") == EXPERIMENT_LABEL)]
    log.append(log_entry)
    with open(SUMMARY_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Appended to running summary log: {SUMMARY_LOG_PATH} ({len(log)} video(s) so far)")


if __name__ == "__main__":
    if not os.getenv("MISTRAL_API_KEY"):
        print("ERROR: MISTRAL_API_KEY not set. Add it to your .env file first.")
    else:
        run_evaluation()