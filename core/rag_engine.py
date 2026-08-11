import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from core.vector_store import build_vector_store,load_vector_store,get_retriever

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )

def format_docs(docs):
    return"\n\n".join([doc.page_content for doc in docs])
def build_rag_chain(transcript: str = None, chunk_size: int = 500, chunk_overlap: int = 50, k: int = 4,
                     vector_store=None):
    # WHY vector_store is now an optional param: previously this function
    # always built its own vector store internally, which meant callers
    # who'd already built one (like evaluate.py, to get a retriever for
    # inspecting retrieved chunks) ended up building it TWICE. Since
    # build_vector_store() now resets/deletes the collection each time it
    # runs (a separate bug fix), the second call was deleting the first
    # call's collection out from under an already-in-use retriever,
    # causing "Collection does not exist" errors. Passing an existing
    # vector_store in avoids the duplicate build entirely.
    if vector_store is None:
        vector_store = build_vector_store(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    retriever = get_retriever(vector_store, k=k)
    llm=get_llm()
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """you are an expert assistant. Answer the user's question based only on the source content provided below.
        If the answer is not found in the content, say:
        "I could not find this information in the source content."
        Always be precise and concise. If quoting someone, mention it clearly.
        source content:
        {context}"""),
        ("human", "{question}"),
    ]
)
    # full LCEL Rag pipeline
    rag_chain=(
        {"context":retriever| RunnableLambda(format_docs),
         "question":RunnablePassthrough()}
         | prompt |llm|StrOutputParser()
    )
    return rag_chain
def load_rag_chain():
    vector_store=load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    llm=get_llm()
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """you are an expert assistant. Answer the user's question based only on the source content provided below.
        If the answer is not found in the content, say:
        "I could not find this information in the source content."
        Always be precise and concise. If quoting someone, mention it clearly.
        source content:
        {context}"""),
        ("human", "{question}"),
    ]
)

    rag_chain=(
        {"context":retriever| RunnableLambda(format_docs),
         "question":RunnablePassthrough()}
         | prompt |llm|StrOutputParser()
    )
    return rag_chain
def ask_question(rag_chain,question:str)->str:
    print(f"Question:{question}")
    answer=rag_chain.invoke(question)
    print(f"answer:{answer}")
    return answer