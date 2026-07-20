# not only we just want to summarizes tings also we want to fetch actionableitems(like what we have to do), decision and questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
import os 

def get_llm():
     return ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2)


def build_chain(system_prompt:str):
     llm=get_llm()
     return (RunnablePassthrough() | RunnableLambda(lambda x:{'text':x})|
             ChatPromptTemplate.from_messages([
                  ("system",system_prompt),
                  ("human","{text}"),
             ])| llm |StrOutputParser()
             )
def extract_action_items(transcript:str)->str:
     chain=build_chain(
          "you are an expert meeting analyst.From the meeting transcript,"
          "extract all action for each provide:\n"
          "-Task description\n"
          "-owner(who is responsible)\n"
          "-dead line (if mentioned ,else write 'not specified')\n"
          "format as numbered list. If none found say 'No action items found"
     )
     return chain.invoke(transcript)

def extract_key_decision(transcript:str)->str:
     chain=build_chain(
          "you are an expert meeting analyst. From the meeting transcript,"
          "extract all key decision made.Format as a numbered list."
          "if none found say 'no key decision found'."
     )
     return chain.invoke(transcript)

def extract_question(transcript:str)->str:
     chain=build_chain(
          "from the meeting transcript , extract all unresolved question"
          "or topiv needing follow up.Format as numbered list."
          "if none found say 'no open question found'."
     )
     return chain.invoke(transcript)

