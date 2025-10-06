from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# --- Import your RAG pipeline ---
from rag_core import retrieve_candidates, llm_recommend_and_call_tool

# --- Pydantic models (request/response) ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

# --- App setup ---
app = FastAPI(title="Smart Librarian API", version="1.0.0")

# --- CORS (allow your frontend origins) ---
FRONTEND_ORIGINS = [
    # Local dev
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # GitHub Pages project site (replace with your values later)
    "https://<your-gh-username>.github.io",
    "https://<your-gh-username>.github.io/<your-frontend-repo>",
    # Custom domain if you add one:
    # "https://books.example.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# --- Health check ---
# @app.get("/health")
# def health() -> dict:
#     ok = bool(os.getenv("OPENAI_API_KEY"))
#     return {"ok": ok}

# --- Main chat endpoint ---
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    # 1) Retrieve semantic candidates
    cands = retrieve_candidates(q, top_k=5)

    # 2) Let the LLM choose + tool-call (or ask clarifying / refuse based on your guards)
    answer_text = llm_recommend_and_call_tool(q, cands)

    # 3) Return normalized shape the frontend expects
    return ChatResponse(answer=answer_text)


# --- Optional: rebuild index endpoint (useful during dev) ---
@app.post("/admin/reindex")
def reindex() -> dict:
    """
    Danger: keep this private or remove in production.
    You can call `python ingest.py` separately instead.
    """
    import shutil
    from ingest import ensure_index
    from config import CHROMA_PATH

    try:
        # Delete vector store and rebuild
        if CHROMA_PATH.exists():
            shutil.rmtree(CHROMA_PATH)
        ensure_index()
        return {"status": "ok", "message": "Rebuilt vector index"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
