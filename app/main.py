from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import build_qa_chain

app = FastAPI(title="RAG Chatbot (Qdrant + E5 + Mistral)")

class Query(BaseModel):
    query: str

qa_chain = None

@app.on_event("startup")
def _startup():
    global qa_chain
    qa_chain = build_qa_chain()

@app.post("/chat")
def chat(q: Query):
    # RetrievalQA handles calling the retriever + LLM
    res = qa_chain({"query": q.query})
    answer = res["result"]
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else "")
        }
        for d in res.get("source_documents", [])
    ]
    return {"answer": answer, "sources": sources}
