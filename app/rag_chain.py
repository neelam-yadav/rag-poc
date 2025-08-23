import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.embeddings_e5 import E5Embeddings

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL","http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY","")
COLLECTION = os.getenv("QDRANT_COLLECTION","rag_poc")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","mistral")

def build_llm():
    # Ollama via the new package
    return OllamaLLM(model=OLLAMA_MODEL)

def build_retriever(k: int = 4):
    embeddings = E5Embeddings()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    vs = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    return vs.as_retriever(search_type="mmr",
                           search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5})

def build_qa_chain():
    llm = build_llm()
    retriever = build_retriever(k=4)

    TEMPLATE = """You are a precise assistant. Use the following context to answer the question.
Cite sources as [source] if helpful. If unsure, say you don't know.

Question: {question}

Context:
{context}

Answer:"""
    prompt = PromptTemplate.from_template(TEMPLATE)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain
