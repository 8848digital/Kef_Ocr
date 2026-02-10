from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
from ocr.ocr_router import smart_ocr
from extraction.llama_json_extractor import LlamaJSONExtractor

app = FastAPI(
    title="OCR + LLM Extraction API",
    description="Indian Document OCR & Structured Extraction",
    version="1.0.0"
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Don't instantiate at module level - use lazy loading instead
llm_extractor = None

def get_llm_extractor():
    """
    Lazy load the LLM model only when first needed.
    This prevents double-loading during uvicorn --reload
    """
    global llm_extractor
    if llm_extractor is None:
        print(" Initializing LLM extractor (first time only)...")
        llm_extractor = LlamaJSONExtractor()
        print(" LLM extractor ready")
    return llm_extractor

def save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

@app.post("/ocr/raw-text")
async def extract_raw_text(file: UploadFile = File(...)):
    """
    Returns ONLY raw OCR text (no LLM)
    """
    file_path = save_upload(file)
    try:
        ocr_result = smart_ocr(file_path)
        if not ocr_result.get("success", True):
            raise HTTPException(status_code=500, detail="OCR failed")
        return {
            "success": True,
            "document_type": ocr_result.get("doc_type", "unknown"),
            "raw_text": ocr_result["text"]
        }
    finally:
        if file_path.exists():
            file_path.unlink()

@app.post("/ocr/llm-extract")
async def extract_structured_data(file: UploadFile = File(...)):
    """
    Runs OCR + LLM and returns structured JSON
    """
    file_path = save_upload(file)
    try:
        # 1️ OCR
        ocr_result = smart_ocr(file_path)
        if not ocr_result.get("success", True):
            raise HTTPException(status_code=500, detail="OCR failed")
        
        raw_text = ocr_result["text"]
        doc_type = ocr_result.get("doc_type", "unknown")
        
        # 2️ LLM Extraction (lazy load model)
        extractor = get_llm_extractor()
        structured = extractor.extract_json(
            raw_text=raw_text,
            doc_type=doc_type
        )
        
        return {
            "success": True,
            "document_type": doc_type,
            "structured_data": structured,
        }
    finally:
        if file_path.exists():
            file_path.unlink()

@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "status": "running",
        "model_loaded": llm_extractor is not None
    }