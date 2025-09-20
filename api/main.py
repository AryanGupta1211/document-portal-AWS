import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Import your existing modules
from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from auth.utils import AuthUtils
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler
from logger import GLOBAL_LOGGER as log

# Import authentication modules
from auth.database import create_tables
from auth.routes import router as auth_router
from auth.middleware import AuthMiddleware
from auth.utils import get_current_active_user

# Configuration
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

# Initialize FastAPI app
app = FastAPI(title="Document Portal API", version="0.1")

# Create database tables
create_tables()

# Setup static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware
app.add_middleware(AuthMiddleware)

# Include authentication routes
app.include_router(auth_router)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        # no cookie → go to login
        return RedirectResponse(url="/auth/", status_code=302)

    try:
        # try to decode/validate token
        username = AuthUtils.verify_token(token, None)
        if username:
            return RedirectResponse(url="/dashboard", status_code=302)
    except:
        pass

    # invalid/expired token → go to login
    return RedirectResponse(url="/auth/", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard(request: Request, current_user = Depends(get_current_active_user)):
    """Serve the main dashboard (protected route)"""
    log.info(f"Dashboard accessed by user: {current_user.username}")
    resp = templates.TemplateResponse(request, "dashboard.html", {
        "request": request, 
        "user": current_user
    })
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}

# ---------- PROTECTED ROUTES ----------
@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...), 
    current_user = Depends(get_current_active_user)
) -> Any:
    try:
        log.info(f"Document analysis requested by {current_user.username}: {file.filename}")
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        log.info("Document analysis complete.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/compare")
async def compare_documents(
    reference: UploadFile = File(...), 
    actual: UploadFile = File(...),
    current_user = Depends(get_current_active_user)
) -> Any:
    try:
        log.info(f"Document comparison requested by {current_user.username}: {reference.filename} vs {actual.filename}")
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        log.info("Document comparison completed.")
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
    current_user = Depends(get_current_active_user)
) -> Any:
    try:
        log.info(f"Chat indexing requested by {current_user.username}. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        ci.built_retriver(
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    current_user = Depends(get_current_active_user)
) -> Any:
    try:
        log.info(f"Chat query from {current_user.username}: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# Command for executing the FastAPI
# uvicorn api.main:app --port 8080 --reload    
# uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload