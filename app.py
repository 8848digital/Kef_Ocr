from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
import traceback
import asyncio
from typing import List

from ocr.ocr_router import smart_ocr
from extraction.llama_json_extractor import LlamaJSONExtractor

# ================================
# Global LLM Instance
# ================================
llm_extractor = None


# ================================
# Lifespan Event Handler
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_extractor
    print(" Loading LLM model at startup...")
    try:
        llm_extractor = LlamaJSONExtractor()
        print(" LLM model loaded successfully")
    except Exception as e:
        print(" Failed to load LLM model")
        traceback.print_exc()
        raise e

    yield

    print(" Shutting down...")


# ================================
# App Instance
# ================================
app = FastAPI(
    title="OCR + LLM Extraction API",
    description="Indian Document OCR & Structured Extraction",
    version="1.0.0",
    lifespan=lifespan
)

# ================================
# File Upload Directory
# ================================
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ================================
# Utility: Save Uploaded File
# ================================
def save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


# ================================
# Utility: Process a Single File
# ================================
async def process_single_file(file: UploadFile) -> dict:
    """
    Process a single file: OCR + LLM extraction.
    Runs blocking calls in a thread pool to avoid blocking the event loop.
    """
    file_path = save_upload(file)
    try:
        loop = asyncio.get_event_loop()

        # Run blocking OCR in thread pool
        ocr_result = await loop.run_in_executor(None, smart_ocr, file_path)

        if not ocr_result.get("success", True):
            return {
                "filename": file.filename,
                "success": False,
                "error": "OCR failed"
            }

        raw_text = ocr_result["text"]
        doc_type = ocr_result.get("doc_type", "unknown")

        # Run blocking LLM extraction in thread pool
        structured = await loop.run_in_executor(
            None,
            lambda: llm_extractor.extract_json(raw_text=raw_text, doc_type=doc_type)
        )

        return {
            "filename": file.filename,
            "success": True,
            "document_type": doc_type,
            "structured_data": structured
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "filename": file.filename,
            "success": False,
            "error": str(e)
        }
    finally:
        if file_path.exists():
            file_path.unlink()


# ================================
# Endpoint 1: OCR Only
# ================================
@app.post("/ocr/raw-text")
async def extract_raw_text(file: UploadFile = File(...)):
    """Returns ONLY raw OCR text (no LLM)"""
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
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()



# Endpoint 2: OCR + LLM (Single or Multiple Files)

@app.post("/ocr/llm-extract")
async def extract_structured_data(files: List[UploadFile] = File(...)):
    """
    Accepts one or more files.
    Runs OCR + LLM extraction on all files in parallel and returns a list of results.
    """
    if llm_extractor is None:
        raise HTTPException(status_code=500, detail="LLM model not loaded")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results = await asyncio.gather(
        *[process_single_file(file) for file in files]
    )

    total = len(results)
    succeeded = sum(1 for r in results if r.get("success"))

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "results": results
    }



# Health Check

@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": llm_extractor is not None
    }