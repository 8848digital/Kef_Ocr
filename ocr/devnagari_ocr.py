import easyocr
import cv2
import os
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import torch
import unicodedata
import re


def normalize_devanagari_text(text: str) -> str:
    """
    Fix Unicode normalization issues in Devanagari text
    Combines separated combining characters (matras, viramas)
    """
    if not text:
        return text
    
    # Unicode NFC normalization - combines characters properly
    normalized = unicodedata.normalize('NFC', text)
    
    # Additional fixes for common PDF text layer issues
    fixes = {
        ' ं': 'ं',    ' ँ': 'ँ',    ' ः': 'ः',    ' ा': 'ा',
        ' ि': 'ि',    ' ी': 'ी',    ' ु': 'ु',    ' ू': 'ू',
        ' े': 'े',    ' ै': 'ै',    ' ो': 'ो',    ' ौ': 'ौ',
        ' ्': '्',    ' ॅ': 'ॅ',    ' ॉ': 'ॉ',
    }
    
    for wrong, right in fixes.items():
        normalized = normalized.replace(wrong, right)
    
    # Remove extra spaces (but preserve single spaces)
    normalized = re.sub(r'  +', ' ', normalized)
    
    return normalized


def group_text_by_lines(result, line_threshold=25):
    """
    Group OCR results by lines based on vertical position
    """
    if not result:
        return []
    
    # Extract bounding boxes and text
    detections = []
    for detection in result:
        # EasyOCR returns: (bbox, text, confidence)
        bbox = detection[0]
        text = detection[1]
        confidence = detection[2]
        
        # Get vertical center of bbox
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_left = bbox[0][0]
        detections.append({
            'text': text,
            'confidence': confidence,
            'y': y_center,
            'x': x_left,
            'bbox': bbox
        })
    
    # Sort by vertical position first, then horizontal
    detections.sort(key=lambda d: (d['y'], d['x']))
    
    # Group into lines
    lines = []
    current_line = []
    current_y = None
    
    for det in detections:
        if current_y is None:
            current_y = det['y']
            current_line.append(det)
        elif abs(det['y'] - current_y) < line_threshold:
            # Same line
            current_line.append(det)
        else:
            # New line
            if current_line:
                # Sort current line by x position
                current_line.sort(key=lambda d: d['x'])
                lines.append(current_line)
            current_line = [det]
            current_y = det['y']
    
    # Add last line
    if current_line:
        current_line.sort(key=lambda d: d['x'])
        lines.append(current_line)
    
    return lines


def is_income_certificate_text(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return False

    text = normalize_devanagari_text(text)
    score = 0

    strong_keywords = [
        "उत्पन्नाचे प्रमाणपत्र",
        "वर्षासाठी उत्पन्नाचे प्रमाणपत्र",
        "आय प्रमाणपत्र",
        "प्रमाणित करण्यात येते",
        "प्रमाणित किया जाता है"
    ]

    medium_keywords = [
        "उत्पन्न",
        "प्रमाणपत्र",
        "तलाठी",
        "तहसील",
        "जिल्हा",
        "महसूल",
        "अर्जदार",
        "कुटुंब"
    ]

    for kw in strong_keywords:
        if kw in text:
            score += 3

    for kw in medium_keywords:
        if kw in text:
            score += 1

    # Amount (supports Devanagari + English digits)
    if re.search(r"[₹]?\s?[0-9०-९,]+", text):
        score += 2

    # Year detection
    if re.search(r"[0-9०-९]{4}", text):
        score += 1

    return score >= 5


# Initialize EasyOCR reader (lazy loading)
_reader = None

def get_reader():
    """Lazy load the EasyOCR reader"""
    global _reader
    if _reader is None:
        use_gpu = torch.cuda.is_available()
        print(f" Initializing EasyOCR reader (GPU: {use_gpu})")
        if use_gpu:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        _reader = easyocr.Reader(['en', 'hi'], gpu=use_gpu)
    return _reader


def run_income_certificate_ocr(file_path):
    """
    Wrapper used by router
    """
    reader = get_reader()  # Lazy load reader
    path = str(file_path)

    full_text = []

    # Process based on file type
    if path.lower().endswith('.pdf'):
        images = convert_from_path(path)
        for img in images:
            img = np.array(img)
            result = reader.readtext(img, detail=1)
            lines = group_text_by_lines(result)
            for line in lines:
                avg_confidence = sum([det['confidence'] for det in line]) / len(line)
                if avg_confidence > 0.25:
                    text = normalize_devanagari_text(
                        ' '.join(d['text'] for d in line)
                    )
                    full_text.append(text)
    else:
        img = cv2.imread(path)
        result = reader.readtext(img, detail=1)
        lines = group_text_by_lines(result)
        for line in lines:
            avg_confidence = sum([det['confidence'] for det in line]) / len(line)
            if avg_confidence > 0.25:
                text = normalize_devanagari_text(
                    ' '.join(d['text'] for d in line)
                )
                full_text.append(text)

    final_text = '\n'.join(full_text)
    detected = is_income_certificate_text(final_text)

    return {
        'text': final_text,
        'doc_type': 'income_certificate' if detected else 'unknown',
        'success': True,
        'confidence': 'high' if detected else 'low'
    }


# ============================================================================
# TEST CODE - Only runs when file is executed directly
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING TEST MODE")
    print("="*80 + "\n")
    
    # File path
    file_path = r"C:\Users\Admin\Desktop\KJSS\KJS 10th Grade Results\10th_Marksheet_7710986_9624162646AW0AV.jpg"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()
    
    # Get file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    reader = get_reader()
    
    # Process based on file type
    if file_ext == '.pdf':
        print("\nProcessing PDF file...\n")
        images = convert_from_path(file_path)
        
        for page_num, pil_image in enumerate(images, 1):
            print(f"{'='*60}")
            print(f"PAGE {page_num}")
            print('='*60)
            
            img = np.array(pil_image)
            result = reader.readtext(img, detail=1)
            
            print("\n=== GROUPED BY LINES (Clean Output) ===\n")
            lines = group_text_by_lines(result, line_threshold=15)
            
            for line in lines:
                line_text = ' '.join([det['text'] for det in line])
                line_text = normalize_devanagari_text(line_text)
                avg_confidence = sum([det['confidence'] for det in line]) / len(line)
                
                if avg_confidence > 0.25:
                    print(f"{line_text}")
            
            print("\n\n=== STRUCTURED OUTPUT WITH CONFIDENCE ===\n")
            
            for line in lines:
                line_text = ' '.join([det['text'] for det in line])
                line_text = normalize_devanagari_text(line_text)
                avg_confidence = sum([det['confidence'] for det in line]) / len(line)
                
                if avg_confidence > 0.25:
                    print(f"[{avg_confidence:.2f}] {line_text}")
            
            print("\n")
    
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        print("\nProcessing image file...\n")
        img = cv2.imread(file_path)
        
        if img is None:
            print("Failed to load image")
            exit()
        
        result = reader.readtext(img, detail=1)
        
        print("=== GROUPED BY LINES (Clean Output) ===\n")
        lines = group_text_by_lines(result, line_threshold=15)
        
        for line in lines:
            line_text = ' '.join([det['text'] for det in line])
            line_text = normalize_devanagari_text(line_text)
            avg_confidence = sum([det['confidence'] for det in line]) / len(line)
            
            if avg_confidence > 0.25:
                print(f"{line_text}")
        
        print("\n\n=== STRUCTURED OUTPUT WITH CONFIDENCE ===\n")
        
        for line in lines:
            line_text = ' '.join([det['text'] for det in line])
            line_text = normalize_devanagari_text(line_text)
            avg_confidence = sum([det['confidence'] for det in line]) / len(line)
            
            if avg_confidence > 0.25:
                print(f"[{avg_confidence:.2f}] {line_text}")
    
    else:
        print(f"Unsupported file type: {file_ext}")