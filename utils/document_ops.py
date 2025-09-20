from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from fastapi import UploadFile
from langchain.schema import Document
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, 
    UnstructuredMarkdownLoader, UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
import sqlite3
import pandas as pd
from unstructured.partition.auto import partition
from unstructured.staging.base import dict_to_elements
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
# SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".xlsx", ".csv", 
    ".ppt", ".pptx", ".db", ".sqlite", ".sqlite3"
}

def load_documents(paths: Iterable[Path], enriched_document: bool = False) -> List[Document]:
    """
    Load docs using appropriate loader based on extension.
    
    Args:
        paths: Iterable of file paths to load
        enriched_document: If True, use unstructured library for enhanced processing 
                          with table and image extraction
    """
    if enriched_document:
        return _load_documents_enriched(paths)
    else:
        return _load_documents_standard(paths)

def _load_documents_standard(paths: Iterable[Path]) -> List[Document]:
    """Standard document loading without enriched processing."""
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(str(p))
            elif ext == ".csv":
                loader = CSVLoader(str(p))
            elif ext == ".xlsx":
                loader = UnstructuredExcelLoader(str(p))
            elif ext in [".ppt", ".pptx"]:
                loader = UnstructuredPowerPointLoader(str(p))
            elif ext in [".db", ".sqlite", ".sqlite3"]:
                docs.extend(_load_sqlite_database(p))
                continue
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
                
            docs.extend(loader.load())
            
        log.info("Documents loaded (standard mode)", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e
    
def _load_sqlite_database(db_path: Path) -> List[Document]:
    """Load SQLite database content as documents."""
    docs: List[Document] = []
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table_name, in tables:
            try:
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # Get table data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000")  # Limit to prevent memory issues
                rows = cursor.fetchall()
                
                if rows:
                    # Convert to DataFrame for better formatting
                    df = pd.DataFrame(rows, columns=column_names)
                    
                    # Create document with table content
                    content = f"Table: {table_name}\n"
                    content += f"Columns: {', '.join(column_names)}\n"
                    content += f"Data:\n{df.to_string()}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(db_path),
                            "table_name": table_name,
                            "row_count": len(rows),
                            "columns": column_names,
                            "file_type": "sqlite"
                        }
                    )
                    docs.append(doc)
                    
            except Exception as table_e:
                log.warning(f"Failed to load table {table_name} from {db_path}", error=str(table_e))
                continue
                
        conn.close()
        log.info(f"Loaded SQLite database", db_path=str(db_path), tables_loaded=len(docs))
        
    except Exception as e:
        log.error(f"Failed to load SQLite database {db_path}", error=str(e))
        
    return docs

def _load_documents_enriched(paths: Iterable[Path]) -> List[Document]:
    """Enriched document loading with unstructured for tables and images."""
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            
            if ext in [".db", ".sqlite", ".sqlite3"]:
                # SQLite files handled separately
                docs.extend(_load_sqlite_database(p))
                continue
                
            # Use unstructured for enriched processing
            try:
                elements = partition(
                    filename=str(p),
                    strategy="hi_res",  # High resolution for better table/image extraction
                    infer_table_structure=True,  # Extract table structure
                    extract_images_in_pdf=True,  # Extract images from PDFs
                    include_page_breaks=True,
                    chunking_strategy="by_title",
                    max_characters=4000,
                    new_after_n_chars=3800,
                    combine_text_under_n_chars=2000,
                )
                
                # Convert elements to Documents
                for element in elements:
                    metadata = {
                        "source": str(p),
                        "element_type": str(type(element).__name__),
                        "page_number": getattr(element, 'metadata', {}).get('page_number', 1),
                    }
                    
                    # Add specific metadata for tables
                    if hasattr(element, 'metadata') and element.metadata:
                        if element.metadata.get('text_as_html'):
                            metadata['table_html'] = element.metadata.get('text_as_html')
                        if element.metadata.get('table_as_cells'):
                            metadata['table_cells'] = str(element.metadata.get('table_as_cells'))
                    
                    # Add image metadata if present
                    if hasattr(element, 'metadata') and element.metadata:
                        if element.metadata.get('image_base64'):
                            metadata['has_image'] = True
                            # Note: We store that there's an image but not the base64 to save space
                            metadata['image_description'] = element.metadata.get('image_text', 'Image found')
                    
                    doc = Document(
                        page_content=str(element),
                        metadata=metadata
                    )
                    docs.append(doc)
                    
            except Exception as e:
                log.warning(f"Failed to process {p} with unstructured, falling back to standard", error=str(e))
                # Fallback to standard loading for this file
                try:
                    fallback_docs = _load_single_file_standard(p)
                    docs.extend(fallback_docs)
                except Exception as fallback_e:
                    log.error(f"Fallback also failed for {p}", error=str(fallback_e))
                    
        log.info("Documents loaded (enriched mode)", count=len(docs))
        return docs
        
    except Exception as e:
        log.error("Failed loading enriched documents", error=str(e))
        raise DocumentPortalException("Error loading enriched documents", e) from e


def _load_single_file_standard(p: Path) -> List[Document]:
    """Load a single file using standard loaders (fallback method)."""
    ext = p.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(p))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(p))
    elif ext == ".txt":
        loader = TextLoader(str(p), encoding="utf-8")
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(str(p))
    elif ext == ".csv":
        loader = CSVLoader(str(p))
    elif ext == ".xlsx":
        loader = UnstructuredExcelLoader(str(p))
    elif ext in [".ppt", ".pptx"]:
        loader = UnstructuredPowerPointLoader(str(p))
    else:
        return []
    
    return loader.load()

# def load_documents(paths: Iterable[Path]) -> List[Document]:
#     """Load docs using appropriate loader based on extension."""
#     docs: List[Document] = []
#     try:
#         for p in paths:
#             ext = p.suffix.lower()
#             if ext == ".pdf":
#                 loader = PyPDFLoader(str(p))
#             elif ext == ".docx":
#                 loader = Docx2txtLoader(str(p))
#             elif ext == ".txt":
#                 loader = TextLoader(str(p), encoding="utf-8")
#             else:
#                 log.warning("Unsupported extension skipped", path=str(p))
#                 continue
#             docs.extend(loader.load())
#         log.info("Documents loaded", count=len(docs))
#         return docs
    # except Exception as e:
    #     log.error("Failed loading documents", error=str(e))
    #     raise DocumentPortalException("Error loading documents", e) from e

def concat_for_analysis(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"

# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def read_pdf_via_handler(handler, path: str) -> str:
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")