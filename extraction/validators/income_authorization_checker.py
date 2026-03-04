from rapidfuzz import fuzz
import re


def normalize_text(text: str) -> str:
    """
    Clean text for better fuzzy matching
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s]', ' ', text)  # keep Hindi + English
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_contains(text: str, keyword: str, threshold: int = 80) -> bool:
    """
    Check if keyword approximately exists in text
    """
    words = text.split()
    for i in range(len(words)):
        window = " ".join(words[i:i+len(keyword.split())])
        if fuzz.ratio(window, keyword) >= threshold:
            return True
    return False


def check_authorization(raw_text: str) -> str:
    """
    Robust authorization detection using fuzzy matching.
    """
    if not raw_text:
        return "no"

    text = normalize_text(raw_text)

    authorization_keywords = [
        "digitally signed",
        "digital signature",
        "डिजीटल स्वाक्षरी",
        "नायब तहसीलदार",
        "tahasildar",
        "तहसिलदार",
        "tahasil office"
    ]

    matches = 0

    for keyword in authorization_keywords:
        if fuzzy_contains(text, keyword, threshold=80):
            matches += 1

    # Require at least 1 strong match
    if matches >= 1:
        return "yes"

    return "no"