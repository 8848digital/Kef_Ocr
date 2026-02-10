from pathlib import Path
from ocr.ocr_router import smart_ocr
from extraction.llama_json_extractor import LlamaJSONExtractor

def run_full_pipeline(file_path: Path):
    # 1️ OCR (router decides)
    ocr_result = smart_ocr(file_path)

    if not ocr_result.get("success"):
        raise RuntimeError("OCR failed")

    raw_text = ocr_result["text"]
    doc_type = ocr_result.get("doc_type", "unknown")

    print("\n OCR RAW TEXT SENT TO LLM")
    print("-" * 60)
    print(raw_text[:800])  # preview
    print("-" * 60)

    # 2️ LLM extraction
    extractor = LlamaJSONExtractor()
    structured_json = extractor.extract_json(
        raw_text=raw_text,
        doc_type=doc_type
    )

    return structured_json


if __name__ == "__main__":
    file_path = Path(
        r"C:\Users\Admin\Desktop\KJSS\Income certificate of 2024-25  (File responses)\11936443___Anshika Prajapati___2025.pdf"
    )

    result = run_full_pipeline(file_path)

    print("\n" + "=" * 80)
    print("FINAL STRUCTURED OUTPUT")
    print("=" * 80)
    print(result)
    print("=" * 80)
