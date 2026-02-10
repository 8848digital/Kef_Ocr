from pathlib import Path
from routing.script_router import is_gibberish_or_devanagari,is_marksheet


from ocr.eng_ocr import extract_text_from_document
from ocr.devnagari_ocr import run_income_certificate_ocr  

def smart_ocr(file_path: Path):
    """
    Master OCR router
    """
    print("\n Running primary OCR (DocTR)...")
    result = extract_text_from_document(file_path)

    if not result['success']:
        raise RuntimeError("DocTR OCR failed")

    text = result['text']

    if is_marksheet(text):
        print(" Marksheet detected → keeping English OCR result")
        return result


    if is_gibberish_or_devanagari(text):
        print(" Detected Devanagari / gibberish → Switching to Income Certificate OCR")
        return run_income_certificate_ocr(file_path)

    print(" English OCR accepted")
    return result

