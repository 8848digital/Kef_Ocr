import torch
import threading
import re
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json

_doctr_model = None
_doctr_lock = threading.Lock()


def initialize_doctr_model():
    """Initialize DocTR OCR model once (thread-safe, singleton pattern)"""
    global _doctr_model

    if _doctr_model is not None:
        return _doctr_model

    with _doctr_lock:
        if _doctr_model is not None:
            return _doctr_model

        print(" Initializing DocTR OCR model (ONE-TIME SETUP)...")
        model = ocr_predictor(pretrained=True)

        if torch.cuda.is_available():
            model = model.cuda()
            print("✓ DocTR model loaded on GPU!")
        else:
            print("✓ DocTR model loaded on CPU")

        _doctr_model = model
        return model


def is_model_loaded() -> bool:
    return _doctr_model is not None


# ─────────────────────────────────────────────────────────────
# Document Type Detection
# ─────────────────────────────────────────────────────────────

def extract_sample_text_doctr(result) -> str:
    """Extract sample text from DocTR result for type detection"""
    sample_text = ""
    word_count = 0
    
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    sample_text += word.value + " "
                    word_count += 1
                    if word_count > 150:  # Check first 150 words
                        return sample_text
    
    return sample_text


def detect_document_type_from_text(text: str) -> str:
    """
    Detect document type from raw text string
    Returns: 'aadhaar', 'pan', 'marksheet', 'passport', 'driving_license', 
             'voter_id', 'bank_statement', 'passbook', 'utility_bill', 'unknown'
    """
    if not text:
        return 'unknown'
    
    sample_lower = text.lower()
    
    # Aadhaar Card
    aadhaar_indicators = ['government of india', 'aadhaar', 'aadhar', 'आधार', 
                          'uidai', 'मेरा आधार', 'भारत सरकार']
    if sum(1 for ind in aadhaar_indicators if ind in sample_lower) >= 2:
        if re.search(r'\d{4}\s*\d{4}\s*\d{4}', text):
            return 'aadhaar'
    
    # PAN Card
    pan_indicators = ['income tax', 'permanent account number', 'pan', 'आयकर']
    if sum(1 for ind in pan_indicators if ind in sample_lower) >= 2:
        if re.search(r'[A-Z]{5}\d{4}[A-Z]', text):
            return 'pan'
    
    # Marksheet Detection
    marksheet_indicators = [
        'marksheet', 'mark sheet', 'statement of marks', 'marks card',
        'grade sheet', 'transcript', 'report card', 'academic record',
        'examination', 'result', 'marks obtained', 'maximum marks',
        'total marks', 'percentage', 'roll number', 'roll no',
        'subject', 'grade', 'cgpa', 'gpa', 'pass', 'fail',
        'board', 'university', 'semester', 'class', 'standard'
    ]
    
    # Count marksheet indicators
    marksheet_count = sum(1 for ind in marksheet_indicators if ind in sample_lower)
    
    # Check for subject-marks pattern (e.g., "Mathematics: 85/100" or "English 78 100")
    has_subject_marks = bool(re.search(r'(mathematics|english|science|hindi|history|geography|physics|chemistry|biology|economics|commerce|accountancy)\s*[:|\s]\s*\d{1,3}\s*[/|\s]\s*\d{1,3}', sample_lower))
    
    # Check for marks obtained pattern
    has_marks_pattern = bool(re.search(r'(marks?\s*obtained|total\s*marks?|maximum\s*marks?)', sample_lower))
    
    # Marksheet detection: needs at least 3 indicators OR subject-marks pattern + marks pattern
    if marksheet_count >= 3 or (has_subject_marks and has_marks_pattern):
        return 'marksheet'
    
    # Passport
    if sum(1 for ind in ['passport', 'republic of india', 'nationality'] if ind in sample_lower) >= 2:
        return 'passport'
    
    # Driving License
    if sum(1 for ind in ['driving licence', 'driving license', 'transport'] if ind in sample_lower) >= 2:
        return 'driving_license'
    
    # Voter ID
    if sum(1 for ind in ['election commission', 'elector', 'epic'] if ind in sample_lower) >= 2:
        return 'voter_id'
    
    # Pass Book - IMPROVED DETECTION
    passbook_indicators = [
        'State Bank of India', 'bank', 'passbook', 'pass book', 'saving account',
        'State Bank of Indla', 'state bank', 'bank of baroda', 'ifsc', 'micr', 
        'Account No : ', 'account no', 'a/c no', 'account name', 'branch name',
        'customer id', 'joint name', 'nominee', 'pass-book', 'A/C Number', 
        'AIC NO', 'Account No '
    ]
    
    # Count how many passbook indicators are present
    passbook_count = sum(1 for ind in passbook_indicators if ind in sample_lower)
    
    # Also check for account number pattern (9-16 digits)
    has_account_number = bool(re.search(r'account\s*no[:\s]*(\d{9,16})', sample_lower))
    
    # Check for IFSC code pattern (e.g., BARB0CHEBOM)
    has_ifsc = bool(re.search(r'ifsc[:\s]*[A-Z]{4}[0-9A-Z]{7}', text, re.IGNORECASE))
    
    # Passbook detection: needs at least 1 indicator OR account number + IFSC
    if passbook_count >= 1 or (has_account_number and has_ifsc):
        return 'pass_book'
    
    # Bank Statement (different from passbook - has transactions)
    if sum(1 for ind in ['bank statement', 'account statement', 'transaction'] if ind in sample_lower) >= 2:
        if 'debit' in sample_lower and 'credit' in sample_lower:
            return 'bank_statement'
    
    # Utility Bill
    if sum(1 for ind in ['electricity bill', 'water bill', 'gas bill'] if ind in sample_lower) >= 2:
        return 'utility_bill'
    
    return 'unknown'


# ─────────────────────────────────────────────────────────────
# Method 1: Simple Block Extraction (ID Cards, Certificates)
# ─────────────────────────────────────────────────────────────

def extract_simple_blocks(result):
    """
    Simple paragraph-based extraction for ID cards and certificates
    Preserves natural document flow without column detection
    """
    paragraphs = []

    for page in result.pages:
        for block in page.blocks:
            lines = []
            for line in block.lines:
                text = " ".join(word.value for word in line.words)
                text = " ".join(text.split())
                if text:
                    lines.append(text)

            if lines:
                paragraphs.append(" ".join(lines))

    return "\n\n".join(paragraphs)


# ─────────────────────────────────────────────────────────────
# Method 2: Smart Table-Aware Extraction (Marksheets, Statements)
# ─────────────────────────────────────────────────────────────

def extract_table_aware(result):
    """
    Table-aware extraction for documents with tabular data
    Handles: HSC, SSC marksheets, Bank Statements, Passbooks
    """
    all_words = []
    
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    bbox = word.geometry
                    y_center = (bbox[0][1] + bbox[1][1]) / 2
                    x_left = bbox[0][0]
                    x_right = bbox[1][0]
                    
                    all_words.append({
                        'text': word.value,
                        'x': x_left,
                        'y': y_center,
                        'x_right': x_right,
                        'width': x_right - x_left
                    })
    
    if not all_words:
        return ""
    
    # Sort by Y position (top to bottom)
    all_words = sorted(all_words, key=lambda w: w['y'])
    
    # Detect table region
    table_region = detect_table_region(all_words)
    
    # Process regions separately
    lines = []
    current_line = []
    current_y = None
    y_threshold = 0.015
    
    for word in all_words:
        if current_y is None or abs(word['y'] - current_y) > y_threshold:
            if current_line:
                line_y = current_line[0]['y']
                
                if table_region and table_region['start'] <= line_y <= table_region['end']:
                    line_text = format_table_line(current_line)
                else:
                    line_text = format_regular_line(current_line)
                
                lines.append(line_text)
            
            current_line = [word]
            current_y = word['y']
        else:
            current_line.append(word)
    
    # Process last line
    if current_line:
        line_y = current_line[0]['y']
        if table_region and table_region['start'] <= line_y <= table_region['end']:
            line_text = format_table_line(current_line)
        else:
            line_text = format_regular_line(current_line)
        lines.append(line_text)
    
    return '\n'.join(lines)


def detect_table_region(words):
    """Detect the Y-coordinate range of tables"""
    table_keywords = [
        'Subject', 'Marks', 'Grade', 'Total', 'Percentage',
        'Transaction', 'Date', 'Debit', 'Credit', 'Balance',
        'Description', 'Unit', 'Price', 'Qty', 'Amount',
        'विषय', 'गुण', 'एकूण', 'S.No', 'Sr.No'
    ]
    
    table_start = None
    table_end = None
    
    y_groups = {}
    for word in words:
        y_key = round(word['y'], 2)
        if y_key not in y_groups:
            y_groups[y_key] = []
        y_groups[y_key].append(word)
    
    for y_pos in sorted(y_groups.keys()):
        line_words = y_groups[y_pos]
        line_text = ' '.join(w['text'] for w in line_words)
        
        if table_start is None:
            if any(kw in line_text for kw in table_keywords):
                table_start = y_pos
        
        if table_start is not None:
            if any(keyword in line_text.upper() for keyword in ['TOTAL', 'GRAND TOTAL', 'BALANCE', 'एकूण']):
                table_end = y_pos
                break
    
    if table_start and table_end:
        return {'start': table_start, 'end': table_end}
    
    return None


def format_table_line(words):
    """Format a line within tables - preserves column structure"""
    words = sorted(words, key=lambda w: w['x'])
    
    result = []
    prev_x_right = 0
    
    for word in words:
        gap = word['x'] - prev_x_right
        
        if gap > 0.05:
            result.append('  ')
        elif result:
            result.append(' ')
        
        result.append(word['text'])
        prev_x_right = word['x_right']
    
    return ''.join(result)


def format_regular_line(words):
    """Format a regular line with column detection"""
    if len(words) <= 1:
        return words[0]['text'] if words else ""
    
    words = sorted(words, key=lambda w: w['x'])
    
    x_positions = [w['x'] for w in words]
    x_gaps = []
    
    for i in range(len(x_positions) - 1):
        gap = x_positions[i + 1] - x_positions[i]
        x_gaps.append((gap, i))
    
    if x_gaps:
        max_gap, split_idx = max(x_gaps, key=lambda x: x[0])
        
        if max_gap > 0.3:
            left_words = words[:split_idx + 1]
            right_words = words[split_idx + 1:]
            
            left_text = ' '.join(w['text'] for w in left_words)
            right_text = ' '.join(w['text'] for w in right_words)
            
            return f"{left_text}  |  {right_text}"
    
    return ' '.join(w['text'] for w in words)


# ─────────────────────────────────────────────────────────────
# Text Cleaning and Formatting
# ─────────────────────────────────────────────────────────────

def clean_document_text(text: str, doc_type: str) -> str:
    """Clean common OCR errors based on document type"""
    
    # Remove excessive spaces (preserve table spacing)
    text = re.sub(r' {3,}', '  ', text)
    
    # Fix punctuation
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r',,+', ',', text)
    
    # Fix common OCR substitutions
    text = re.sub(r'(\w{4,})l(\s+[A-Z])', r'\1ce\2', text)
    text = re.sub(r'\b1(\d{4,})\b', r'\1', text)
    
    # Document-specific cleaning
    if doc_type in ['aadhaar', 'pan', 'passport']:
        # Fix ID number spacing
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    if doc_type in ['hsc', 'ssc']:
        # Preserve subject-marks spacing
        text = re.sub(r'([A-Za-z]+)\s+(\d{1,3})\s+(\d{1,3})', r'\1  \2  \3', text)
    
    if doc_type in ['bank_statement', 'pass_book']:
        # Preserve transaction columns
        text = re.sub(r'(\d{2}/\d{2}/\d{4})', r'\n\1', text)
    
    return text.strip()


def format_document_text(text: str, doc_type: str) -> str:
    """Format text based on document type"""
    
    # Clean first
    text = clean_document_text(text, doc_type)
    
    # Add section breaks for better readability
    if doc_type in ['aadhaar', 'pan']:
        # Add breaks before key labels
        labels = ['Name', 'Date of Birth', 'Address', 'Father', 'Gender']
        for label in labels:
            text = re.sub(f'(?<!\n)({label})', r'\n\n\1', text, flags=re.IGNORECASE)
    
    if doc_type in ['hsc', 'ssc']:
        # Add breaks before sections
        sections = ['Name', 'Roll Number', 'Subject', 'Total', 'Result']
        for section in sections:
            text = re.sub(f'(?<!\n)({section})', r'\n\n\1', text, flags=re.IGNORECASE)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# ─────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────

def extract_text_from_document(file_path: Path) -> dict:
    """
    Extract text from Indian documents using DocTR only
    
    Returns:
        dict: {
            'text': str,
            'doc_type': str,
            'success': bool,
            'error': str (if failed)
        }
    """
    try:
        print(f" Processing: {file_path.name}")
        
        # Initialize DocTR model
        model = initialize_doctr_model()
        
        # Load document
        if file_path.suffix.lower() == '.pdf':
            print(" Loading PDF...")
            doc = DocumentFile.from_pdf(str(file_path))
        else:
            print(" Loading image...")
            doc = DocumentFile.from_images(str(file_path))
        
        # Run DocTR OCR
        print(" Running DocTR OCR...")
        result = model(doc)
        
        # STEP 1: Extract text FIRST (before detection)
        print(f" Extracting text...")
        
        # Try table-aware extraction first
        full_text = extract_table_aware(result)
        
        # Fallback to simple if extraction failed
        if not full_text or len(full_text) < 50:
            full_text = extract_simple_blocks(result)
        
        # STEP 2: NOW detect document type from the EXTRACTED TEXT
        doc_type = detect_document_type_from_text(full_text)
        print(f" Detected: {doc_type.upper().replace('_', ' ')}")
        
        # STEP 3: Re-extract with optimal method based on detected type
        if doc_type in ['hsc', 'ssc', 'bank_statement', 'pass_book']:
            print(f" Using: Table-Aware Extraction")
            full_text = extract_table_aware(result)
        else:
            print(f" Using: Simple Block Extraction")
            full_text = extract_simple_blocks(result)
        
        # Format and clean
        full_text = format_document_text(full_text, doc_type)
        
        # Display preview
        print("\n" + "="*80)
        print("EXTRACTED TEXT PREVIEW:")
        print("="*80)
        preview = full_text[:1000] + "..." if len(full_text) > 1000 else full_text
        print(preview)
        print("="*80)
        print(f"✓ Extraction complete! ({len(full_text)} characters)\n")
        
        return {
            'text': full_text,
            'doc_type': doc_type,
            'success': True
        }

    except Exception as e:
        print(f" Extraction failed for {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'text': '',
            'doc_type': 'unknown',
            'success': False,
            'error': str(e)
        }


# ─────────────────────────────────────────────────────────────
# INTEGRATED JSON EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_and_convert_to_json(file_path: Path, save_json=True) -> dict:
    """
    Complete pipeline: OCR → Document Detection → JSON Extraction
    
    Args:
        file_path: Path to document (PDF or image)
        save_json: Whether to save JSON to file (default: True)
    
    Returns:
        dict: Structured JSON data
    """
    print(f"\n{'='*80}")
    print(f" INTEGRATED OCR + JSON EXTRACTION PIPELINE")
    print(f"{'='*80}\n")
    
    # Step 1: Extract text using OCR
    print(" STEP 1: OCR TEXT EXTRACTION")
    print("-" * 80)
    ocr_result = extract_text_from_document(file_path)
    
    if not ocr_result['success']:
        return {
            "error": "OCR extraction failed", 
            "file": str(file_path),
            "details": ocr_result.get('error', 'Unknown error')
        }
    
    raw_text = ocr_result['text']
    doc_type = ocr_result['doc_type']
    
    print(f"\n OCR Complete:")
    print(f"   • Extracted: {len(raw_text)} characters")
    print(f"   • Document Type: {doc_type.upper()}\n")
    
    # Step 2: Extract structured JSON using Llama
    print("2️ STEP 2: JSON EXTRACTION WITH LLAMA")
    print("-" * 80)
    
    try:
        # Import here to avoid circular dependency
        from llama_json_extractor import LlamaJSONExtractor
        
        extractor = LlamaJSONExtractor()
        structured_data = extractor.extract_json(raw_text, doc_type)
        
        print(" JSON Extraction Complete!\n")
        
        # Display JSON
        print("="*80)
        print("EXTRACTED JSON DATA:")
        print("="*80)
        print(json.dumps(structured_data, indent=2, ensure_ascii=False))
        print("="*80)
        
        # Display raw text
        print("\n" + "="*80)
        print("RAW OCR TEXT:")
        print("="*80)
        print(raw_text)
        print("="*80 + "\n")
        
        # Save JSON to file
        if save_json:
            json_file = file_path.parent / f"{file_path.stem}_extracted.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            print(f" JSON saved to: {json_file.absolute()}")
        
        return structured_data
        
    except ImportError:
        print(" Error: llama_json_extractor.py not found!")
        print("   Make sure llama_json_extractor.py is in the same directory.")
        return {
            "error": "llama_json_extractor module not found",
            "ocr_text": raw_text,
            "doc_type": doc_type
        }
    except Exception as e:
        print(f" JSON extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"JSON extraction failed: {str(e)}",
            "ocr_text": raw_text,
            "doc_type": doc_type
        }


# ─────────────────────────────────────────────────────────────
# Testing function
# ─────────────────────────────────────────────────────────────

def test_document_ocr(pdf_folder: str = "./test_documents", extract_json=True):
    """
    Test OCR on multiple documents in a folder
    
    Args:
        pdf_folder: Path to folder containing documents
        extract_json: Whether to extract JSON (default: True)
    """
    folder_path = Path(pdf_folder)
    
    if not folder_path.exists():
        print(f" Folder not found: {folder_path}")
        print("Creating test folder...")
        folder_path.mkdir(exist_ok=True)
        print(f"✓ Created folder: {folder_path.absolute()}")
        print(f" Place your PDFs/images in: {folder_path.absolute()}")
        return
    
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp','.heic']
    
    documents = []
    for ext in supported_extensions:
        documents.extend(folder_path.glob(f'*{ext}'))
        documents.extend(folder_path.glob(f'*{ext.upper()}'))
    
    if not documents:
        print(f" No documents found in {folder_path}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return
    
    print(f"\n Found {len(documents)} document(s) to process\n")
    
    results = []
    for doc_path in documents:
        print(f"\n{'='*80}")
        
        if extract_json:
            result = extract_and_convert_to_json(doc_path)
        else:
            result = extract_text_from_document(doc_path)
        
        results.append({
            'filename': doc_path.name,
            **result
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r.get('success', True) and 'error' not in r)
    print(f"✓ Successful: {successful}/{len(results)}")
    print(f" Failed: {len(results) - successful}/{len(results)}")
    
    return results


# ─────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(" DocTR-Only OCR Service for Indian Documents")
    print("   + Integrated Llama JSON Extraction")
    print("="*80)
    print("Supports: Aadhaar, PAN, HSC, SSC, Passport, DL, Voter ID,")
    print("          Bank Statements, Passbooks, Utility Bills")
    print("="*80 + "\n")
    
    # Single file test with JSON extraction
    test_file = Path(r"C:\Users\Admin\Desktop\KJSS\KJS 10th Grade Results\10th_Marksheet_10186360_2218797394P9ECB.jpg")
    if test_file.exists():
        # Extract both OCR text AND JSON
        json_result = extract_and_convert_to_json(test_file, save_json=False)
        
    else:
        print(f" Test file not found: {test_file}")
        print("\nRunning batch test instead...\n")
        test_document_ocr(extract_json=True)