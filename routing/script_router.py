import re
from collections import Counter

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
ENGLISH_WORD_RE = re.compile(r'\b[a-zA-Z]{3,}\b')


MARKSHEET_KEYWORDS = {
    'statement of marks',
    'secondary school certificate',
    'certificate examination',
    'seat no',
    'centre no',
    'month & year of exam',
    'subject name',
    'marks obtained',
    'marks in figures',
    'marks in words',
    'mathematics',
    'science',
    'english',
    'marathi',
    'hindi',
    'social sciences',
    'ssc',
    'hsc'
}




# Common words in financial documents
FINANCIAL_KEYWORDS = {
    'bank', 'account', 'name', 'address', 'branch', 'code', 'number',
    'customer', 'date', 'mobile', 'email', 'ifsc', 'micr', 'nominee',
    'occupation', 'student', 'manager', 'details', 'contact', 'mumbai',
    'maharashtra', 'india', 'canara', 'opened', 'building', 'floor',
}

COMMON_WORDS = {
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had','Marks','Subject Name'
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
    'can', 'this', 'that', 'these', 'those', 'and', 'or', 'but', 'if', 'because',
    'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'on', 'off','Mothers Name',"MOTHER'S NAME","MATHEMATICS"
}

# Common Devanagari OCR artifacts (from bad Marathi/Hindi OCR)
DEVANAGARI_OCR_ARTIFACTS = {
    'THIOT', 'REUIR', 'foleel', 'Ridle', 'mocal', 'HGRIa', 'JUCR', 'HPIFT',
    'fholdd', 'gyfa', 'Vobdet', 'chFIG4aiCI', 'yuda', 'yiarra', 'aypfias',
    'VIUITT', 'prIdt', 'gradt', '3TET', '3TEIRIGR', 'gefle', 'asardo',
    'shHich', 'gAo', 'dethes', 'CRUIIT', 'geitldleryald', 'fucerferprta',
    'Meeldsiferbr', 'pRloe', 'Rasooce', 'Saat', 'Rasta', 'Guruji', 'chooece',
    'GrTgqut', 'HGRAT', 'GrGOT', 'mocal', 'chFIG', 'yarda', 'prIdt'
}


def is_marksheet(text: str) -> bool:
    text_l = text.lower()
    hits = sum(1 for kw in MARKSHEET_KEYWORDS if kw in text_l)

    # Marksheets are very distinctive
    if hits >= 3:
        print(f"  Marksheet detected ({hits} strong signals)")
        return True

    return False




def is_gibberish_or_devanagari(text: str) -> bool:
    """
    Detect TRUE gibberish or Devanagari text.
    Enhanced to catch Devanagari OCR artifacts and digit-letter mixing.
    """
    if not text or len(text.strip()) < 50:
        if text and is_marksheet(text):
             print("   🎓 Short text but marksheet → forcing ENGLISH OCR")
             return False
        return True
    
    text = text.strip()
    total_chars = len(text)

    if is_marksheet(text):
        print("   🎓 Marksheet detected → not gibberish, use English OCR")
        return False
    
    # 1. Devanagari check - if >5% Devanagari, definitely switch
    devanagari_chars = len(DEVANAGARI_RE.findall(text))
    if devanagari_chars / total_chars > 0.05:
        print(f"   ✓ Devanagari detected: {devanagari_chars}/{total_chars} chars ({devanagari_chars/total_chars:.1%})")
        return True
    
    # 2. Extract words
    words = ENGLISH_WORD_RE.findall(text)
    if len(words) < 20:
        print(f"   ✓ Too few words: {len(words)} < 20")
        return True
    
    word_list = [w.lower() for w in words]
    unique_words = set(word_list)
    
    # 3. NEW: Check for Devanagari OCR artifacts (CRITICAL!)
    # These are telltale signs of badly OCR'd Marathi/Hindi text
    artifact_matches = sum(1 for artifact in DEVANAGARI_OCR_ARTIFACTS if artifact in text)
    if artifact_matches >= 3:
        print(f"   ✓ Found {artifact_matches} Devanagari OCR artifacts")
        return True
    
    # 4. NEW: Check for excessive digit-letter mixing (huge red flag)
    # Patterns like: 3n4ch, R1ET, 46/m, 9g, 4T
    digit_letter_pattern = re.compile(r'\b(?:[0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+)\b')
    mixed_words = digit_letter_pattern.findall(text)
    mixed_ratio = len(mixed_words) / max(len(text.split()), 1)
    
    if mixed_ratio > 0.25:  # >25% of tokens have digit-letter mixing
        print(f"   ✓ Excessive digit-letter mixing: {mixed_ratio:.1%} ({len(mixed_words)} occurrences)")
        return True
    
    # 5. Check for financial/document keywords (IMPORTANT!)
    # If we find financial keywords, it's likely a real document with OCR errors
    financial_word_count = sum(1 for w in unique_words if w in FINANCIAL_KEYWORDS)
    if financial_word_count >= 3:  # Found 3+ financial keywords
        print(f"   ✗ Found {financial_word_count} financial keywords - likely real document")
        return False  # It's a real document, not gibberish
    
    # 6. Check for common English words
    common_word_count = sum(1 for w in unique_words if w in COMMON_WORDS)
    if common_word_count < 5 and not is_marksheet(text):
        print(f"   ✓ Too few common words: {common_word_count} < 5")
        return True
    
    # 7. NEW: Check for unusual capital letter patterns
    # Real English has predictable capitalization; gibberish has random caps
    caps_pattern = re.compile(r'\b[A-Z]+\b')
    all_caps_words = caps_pattern.findall(text)
    # Filter out reasonable acronyms (2-4 letters) and check remaining
    unusual_caps = [w for w in all_caps_words if len(w) > 4 and w.lower() not in COMMON_WORDS | FINANCIAL_KEYWORDS]
    unusual_caps_ratio = len(unusual_caps) / max(len(words), 1)
    
    if unusual_caps_ratio > 0.20:  # >20% unusual all-caps words
        print(f"   ✓ Unusual capitalization: {unusual_caps_ratio:.1%} ({len(unusual_caps)} words)")
        return True
    
    # 8. Consonant cluster check (stricter)
    # Only flag if MANY words have impossible clusters
    bad_cluster_count = 0
    for word in words:
        consonants = 'bcdfghjklmnpqrstvwxyz'
        consecutive = 0
        max_consecutive = 0
        for char in word.lower():
            if char in consonants:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        # Only count truly impossible clusters (5+)
        if max_consecutive >= 5:
            bad_cluster_count += 1
    
    # More than 30% of words have impossible clusters
    if bad_cluster_count / len(words) > 0.30:
        print(f"   ✓ Impossible consonant clusters: {bad_cluster_count}/{len(words)} words ({bad_cluster_count/len(words):.1%})")
        return True
    
    # 9. NEW: Single letter "words" check
    # Gibberish often has many scattered single letters
    single_letter_pattern = re.compile(r'\b[a-zA-Z]\b')
    single_letters = single_letter_pattern.findall(text)
    single_letter_ratio = len(single_letters) / max(len(text.split()), 1)
    
    if single_letter_ratio > 0.15:  # >15% single letters
        print(f"   ✓ Too many single letters: {single_letter_ratio:.1%} ({len(single_letters)} occurrences)")
        return True
    
    # 10. Word length check
    word_lengths = [len(w) for w in words]
    avg_length = sum(word_lengths) / len(word_lengths)
    if avg_length < 2.5 or avg_length > 12:
        print(f"   ✓ Unusual average word length: {avg_length:.1f}")
        return True
    
    # If we got here, it seems like reasonably clean English
    print(f"   ✗ Appears to be valid English (passed all checks)")
    return False