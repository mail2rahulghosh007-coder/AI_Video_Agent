# not only we just want to summarizes tings also we want to fetch actionableitems(like what we have to do), decision and questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from utils.retry_utils import retry_on_failure

def get_llm():
     return ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2)


def build_chain(system_prompt: str):
     llm = get_llm()
     return (RunnablePassthrough() | RunnableLambda(lambda x: {'text': x}) |
             ChatPromptTemplate.from_messages([
                  ("system", system_prompt),
                  ("human", "{text}"),
             ]) | llm | StrOutputParser()
             )


# Applying retry to each extraction call -- these hit the Mistral API,
# which can fail transiently (rate limits, timeouts). Without this,
# a single dropped request would crash the whole pipeline AFTER the
# expensive transcription step has already completed.

@retry_on_failure(max_attempts=3, initial_delay=2, backoff_factor=2)
def extract_action_items(transcript: str) -> str:
     chain = build_chain(
          "you are an expert meeting analyst.From the meeting transcript,"
          "extract all action for each provide:\n"
          "-Task description\n"
          "-owner(who is responsible)\n"
          "-dead line (if mentioned ,else write 'not specified')\n"
          "format as numbered list. If none found say 'No action items found"
     )
     try:
          return chain.invoke(transcript)
     except Exception as e:
          # If ALL retries are exhausted, return a clear fallback message
          # instead of letting the whole pipeline crash -- the user still
          # gets their summary and other results, just not this one section.
          print(f"[extract_action_items] Failed after retries: {e}")
          return "Could not extract action items (API error). Please try again later."


@retry_on_failure(max_attempts=3, initial_delay=2, backoff_factor=2)
def extract_key_decision(transcript: str) -> str:
     chain = build_chain(
          "you are an expert meeting analyst. From the meeting transcript,"
          "extract all key decision made.Format as a numbered list."
          "if none found say 'no key decision found'."
     )
     try:
          return chain.invoke(transcript)
     except Exception as e:
          print(f"[extract_key_decision] Failed after retries: {e}")
          return "Could not extract key decisions (API error). Please try again later."


@retry_on_failure(max_attempts=3, initial_delay=2, backoff_factor=2)
def extract_question(transcript: str) -> str:
     chain = build_chain(
          "from the meeting transcript , extract all unresolved question"
          "or topiv needing follow up.Format as numbered list."
          "if none found say 'no open question found'."
     )
     try:
          return chain.invoke(transcript)
     except Exception as e:
          print(f"[extract_question] Failed after retries: {e}")
          return "Could not extract open questions (API error). Please try again later."