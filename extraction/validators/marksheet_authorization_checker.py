import re
from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fuzzy_contains(text: str, keyword: str, threshold: int = 85) -> bool:
    words = text.split()
    key_words = keyword.split()
    window_size = len(key_words)

    for i in range(len(words)):
        window = " ".join(words[i:i+window_size])
        if fuzz.ratio(window, keyword.lower()) >= threshold:
            return True
    return False


def has_serial_number(text: str) -> bool:
    """
    Match strong certificate serial numbers only.
    Avoid matching seat numbers like A237370.
    """

    patterns = [
        r"\bS\d{8,12}\b",              # S1244273467 (strong pattern)
        r"\bSR\.?\s*NO\.?\s*\d+\b",    # SR.NO. 273467
        r"\b\d{10,14}\b",              # long numeric certificate IDs
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def check_marksheet_authorization(raw_text: str) -> bool:
    if not raw_text:
        return False

    text = normalize_text(raw_text)

    authorization_keywords = [
        "chief executive & secretary chairman council for the indian school certificate examinations",
        "controller of examinations",
        "Chief Executive & Secretary Chairman Council for the Indian Schooce Council"
    ]

    for keyword in authorization_keywords:
        if fuzzy_contains(text, keyword, threshold=85):
            return True

    # Serial number fallback
    if has_serial_number(raw_text):
        return True

    return False