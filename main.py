from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items,extract_key_decision,extract_question
from core.rag_engine import build_rag_chain,ask_question

load_dotenv()

def run_pipeline(source:str,language:str="english")->dict:
    print("starting AI Video Assistant")

    chunks=process_input(source)

    transcript=transcribe_all(chunks,language=language)
    print(f"raw transcription (first 300 characters) {transcript[:300]}")

    title=generate_title(transcript)

    summary=summarize(transcript)

    action_item=extract_action_items(transcript)

    decision=extract_key_decision(transcript)

    question=extract_question(transcript)

    rag_chain=build_rag_chain(transcript)
    return{
        "title":title,
        "summary":summary,
        "action_item":action_item,
        "key_decision":decision,
        "open_question":question,
        "rag_chain":rag_chain

    }


if __name__=="__main__":
    # CLI entry point
    source=input("enter youtube url or local file path:").strip()
    language=input("language english/hinglish:").strip() or "english"
    result=run_pipeline(source,language)

    print("\n"+"="*60)
    print(f"Title:{result['title']}")
    print(f"\n summary:\n{result['summary']}")
    print(f"\n Action Items:\n{result['action_item']}")
    print(f"\n Key Decision:\n{result['key_decision']}")
    print(f"\n Open Question :\n{result['open_question']}")
    print("="*60)

    #phase 2 chat with your meeting via rag
    print("\n chat with your meeting and type exit to quit\n")
    rag_chain=result["rag_chain"]
    while True:
        question=input("you:").strip()
        if question.lower() in ["exit","quit","q"]:
            print("goodbye!")
            break
        if not question:
            continue
        answer=ask_question(rag_chain,question)
        print(f"\n Assistant:{answer}\n")

