
import re


# ═══════════════════════════════════════════════════════════════════════════
# FORMAT VERSION 1 (CURRENT) - RULES
# ═══════════════════════════════════════════════════════════════════════════

_SSC_RULES_V1 = {
    "code_map": {
        # Core subjects (stable since 2010)
        "01": "Marathi (1st Lang)",
        "02": "Hindi (1st Lang)",
        "03": "English (1st Lang)",
        "71": "Mathematics",
        "72": "Science & Technology",
        "73": "Social Sciences",
        
        # Language variants (added 2015)
        "15": "Hindi (2/3 Lang)",
        "16": "Marathi (2/3 Lang)",
        "17": "English (2/3 Lang)",
        "27": "Sanskrit (2/3 Lang)",
        
        # OCR error variants (keep for backward compatibility)
        "72:": "Science & Technology",  # Colon OCR error
        "725": "Science & Technology",  # '5' OCR error
        "73:": "Social Sciences",       # Colon OCR error
        "151": "Hindi (2/3 Lang)",      # Alternative code
        
        # Grade-only subjects (format changed 2018)
        "P1": "Health & Physical Education",
        "P2": "Scouting/Guiding",
        "P4": "Defence Studies",
        "P41": "Defence Studies",  # Alternative code
        "PA": "Defence Studies",   # Alternative code
        "R8": "Water Security",
        "RB": "Water Security",    # Alternative code
    },
    "code_pattern": r"\b({all_codes})\s+",
    "marks_pattern": r"100\s+0?(\d{2,3})\s+([A-Z]+)",
    "grade_pattern": r"([A-E])(?:\s|$|[^A-Z])",
}

# ═══════════════════════════════════════════════════════════════════════════
# FORMAT VERSION 2 (FUTURE) - PLACEHOLDER RULES
# ═══════════════════════════════════════════════════════════════════════════
# When SSC changes format, update these rules and add detection logic

_SSC_RULES_V2 = {
    "code_map": {
        # Copy from V1 and add new codes as needed
        **_SSC_RULES_V1["code_map"],
        # Example future additions:
        # "NEW_CODE": "New Subject Name",
    },
    # Example: if format changes to "Code: XX"
    "code_pattern": r"(?:Code|CODE):\s*({all_codes})",
    # Example: if marks format changes to "Marks: XX/100"
    "marks_pattern": r"Marks:\s*(\d{2,3})/100",
    "grade_pattern": r"Grade:\s*([A-E])",
}


# ═══════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Subjects whose marks are always null — only a grade letter
_SSC_GRADE_ONLY_NAMES = {
    "health & physical education",
    "scouting/guiding",
    "water security",
    "defence studies",
}

_SSC_VALID_GRADE_LETTERS = {"A", "B", "C", "D", "E"}

# Markers that appear at/after the last subject row — used to locate totals area
_SSC_TOTALS_MARKERS = [
    "WATER SECURITY",
    "SCOUTING",
    "HEALTH & PHYSICAL",
    "SOCIAL SCIENCES",
    "DEFENCE STUDIES",
]

# (raw_text_marker, canonical_name) pairs for injection of dropped subjects
_SSC_GRADE_ONLY_INJECTION = [
    ("P1", "Health & Physical Education"),
    ("HEALTH & PHYSICAL", "Health & Physical Education"),
    ("P2", "Scouting/Guiding"),
    ("SCOUTING", "Scouting/Guiding"),
    ("R8", "Water Security"),
    ("WATER SECURITY", "Water Security"),
    ("RB", "Water Security"),
    ("P4", "Defence Studies"),
    ("PA", "Defence Studies"),
    ("P41", "Defence Studies"),
    ("DEFENCE STUDIES", "Defence Studies"),
]


# ═══════════════════════════════════════════════════════════════════════════
# VERSION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_ssc_format_version(raw_text: str) -> dict:
    """
    Auto-detect which rule set to use based on text patterns.
    
    Returns the appropriate rules dict (_SSC_RULES_V1 or _SSC_RULES_V2).
    Add more detection logic here when new formats are introduced.
    """
    raw_upper = raw_text.upper()
    
    # Check for v2 indicators (add these when v2 is released)
    v2_indicators = [
        "ACADEMIC YEAR 2025",  # Example: new year format
        "REVISED MARKSHEET",   # Explicit revision marker
        "FORMAT VERSION 2",    # Direct version indicator
        "NEW PATTERN",         # Pattern change indicator
    ]
    
    for indicator in v2_indicators:
        if indicator in raw_upper:
            print(f"  📋 Detected SSC format v2 (indicator: '{indicator}')")
            return _SSC_RULES_V2
    
    # Check for v1 indicators (current format)
    v1_indicators = [
        "STATE BOARD",
        "MAHARASHTRA",
        "SEAT NO",
        "MSBSHSE",  # Maharashtra State Board of Secondary and Higher Secondary Education
    ]
    
    found_v1 = sum(1 for ind in v1_indicators if ind in raw_upper)
    if found_v1 >= 2:
        print("  📋 Detected SSC format v1 (current)")
        return _SSC_RULES_V1
    
    # Default to v1 if no clear indicators
    print("  📋 Using default SSC format v1")
    return _SSC_RULES_V1


def _detect_potential_format_change(raw_text: str) -> bool:
    """
    Detect if marksheet might be using a new/unknown format.
    Returns True if suspicious patterns found.
    """
    raw_upper = raw_text.upper()
    
    # Suspicious patterns that suggest format changes
    suspicious_patterns = [
        r"FORMAT\s+VERSION\s+[2-9]",  # Explicit version 2+
        r"REVISED\s+MARK",             # Revised marksheet
        r"NEW\s+PATTERN",              # New pattern
        r"UPDATED\s+FORMAT",           # Updated format
        r"\b20(2[5-9]|[3-9]\d)\b",     # Years 2025+ (adjust as needed)
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, raw_upper):
            return True
    
    # Check if known markers are missing (might indicate structure change)
    known_markers = ["SOCIAL SCIENCES", "MATHEMATICS", "SEAT NO", "SEAT NUMBER"]
    found_markers = sum(1 for marker in known_markers if marker in raw_upper)
    
    if found_markers < 2:
        return True
    
    # Check if code pattern is completely different
    rules = detect_ssc_format_version(raw_text)
    all_codes = "|".join(re.escape(code) for code in rules["code_map"].keys())
    code_pattern = rules["code_pattern"].replace("{all_codes}", all_codes)
    
    matches = list(re.finditer(code_pattern, raw_text))
    if len(matches) < 3:  # Less than 3 subject codes found
        return True
    
    return False


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION STRATEGY 1: CODE-POSITION BASED (MOST PRECISE)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_by_code_position(raw_text: str, rules: dict) -> list:
    """
    Extract SSC subjects by finding subject codes and parsing segments.
    Uses code-position splitting: finds each code, grabs text between it and next code,
    then parses marks or grade from that segment.
    
    Returns list of subject dicts, or [] if fewer than 4 subjects found.
    """
    subjects = []
    code_map = rules["code_map"]
    
    # Build pattern from all possible codes
    all_codes = "|".join(re.escape(code) for code in code_map.keys())
    code_pattern = rules["code_pattern"].replace("{all_codes}", all_codes)
    
    code_positions = [
        (m.group(1), m.start(), m.end())
        for m in re.finditer(code_pattern, raw_text)
    ]
    
    if not code_positions:
        return []
    
    for i, (code, _start, end) in enumerate(code_positions):
        # Extract segment between this code and next code
        segment = (
            raw_text[end : code_positions[i + 1][1]].strip()
            if i < len(code_positions) - 1
            else raw_text[end:].strip()
        )
        
        # Normalize code variants (e.g. "725" → "72", "73:" → "73")
        code_clean = code.replace(":", "").replace("5", "")
        subject_name = code_map.get(code, code_map.get(code_clean, "Unknown"))
        
        if subject_name == "Unknown":
            continue
        
        # Check if this is a numeric subject or grade-only
        if "100" in segment:
            # Numeric subject: marks appear after "100 "
            m = re.search(rules["marks_pattern"], segment)
            if m:
                subjects.append({
                    "subject_name": subject_name,
                    "marks_obtained": int(m.group(1)),
                    "max_marks": 100,
                    "grade": None,
                })
        else:
            # Grade-only subject: find the single letter grade
            m = re.search(rules["grade_pattern"], segment)
            if m:
                subjects.append({
                    "subject_name": subject_name,
                    "marks_obtained": None,
                    "max_marks": None,
                    "grade": m.group(1),
                })
    
    return subjects if len(subjects) >= 4 else []


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION STRATEGY 2: TABLE STRUCTURE BASED (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════════

def _fuzzy_match_subject_name(text: str) -> str:
    """
    Match OCR'd text to canonical subject names using fuzzy logic.
    Handles OCR errors and variations.
    """
    text_clean = text.upper().strip()
    
    # Remove common noise
    text_clean = re.sub(r"[^A-Z0-9\s&()]", "", text_clean)
    
    # Direct substring matching with priority order
    if "MATH" in text_clean:
        return "Mathematics"
    
    if "SCIENCE" in text_clean or "SCI" in text_clean or "TECH" in text_clean:
        return "Science & Technology"
    
    if "SOCIAL" in text_clean:
        return "Social Sciences"
    
    if "MARATHI" in text_clean:
        # Distinguish between 1st and 2/3 language
        if "1ST" in text_clean or "FIRST" in text_clean or "1 LANG" in text_clean:
            return "Marathi (1st Lang)"
        return "Marathi (2/3 Lang)"
    
    if "HINDI" in text_clean:
        if "1ST" in text_clean or "FIRST" in text_clean or "1 LANG" in text_clean:
            return "Hindi (1st Lang)"
        return "Hindi (2/3 Lang)"
    
    if "ENGLISH" in text_clean:
        if "1ST" in text_clean or "FIRST" in text_clean or "1 LANG" in text_clean:
            return "English (1st Lang)"
        return "English (2/3 Lang)"
    
    if "SANSKRIT" in text_clean:
        return "Sanskrit (2/3 Lang)"
    
    if "HEALTH" in text_clean or "PHYSICAL" in text_clean:
        return "Health & Physical Education"
    
    if "SCOUT" in text_clean or "GUID" in text_clean:
        return "Scouting/Guiding"
    
    if "WATER" in text_clean or "SECURITY" in text_clean:
        return "Water Security"
    
    if "DEFENCE" in text_clean or "DEFENSE" in text_clean:
        return "Defence Studies"
    
    return None


def _extract_by_table_structure(raw_text: str) -> list:
    """
    Fallback extraction: Look for table-like structure without exact codes.
    Handles cases where codes are OCR'd incorrectly but structure is intact.
    
    Pattern: <subject_text> 100 <marks> <grade_letter>
    """
    subjects = []
    
    # Pattern for numeric subjects: text + "100" + 2-3 digit marks + optional grade
    numeric_pattern = r"(.{10,50}?)\s+100\s+0?(\d{2,3})(?:\s+([A-Z]))?"
    
    for match in re.finditer(numeric_pattern, raw_text):
        subject_text = match.group(1).strip()
        marks = int(match.group(2))
        
        # Skip if marks are invalid
        if marks > 100:
            continue
        
        # Try to map to canonical name
        canonical = _fuzzy_match_subject_name(subject_text)
        if canonical and canonical.lower() not in _SSC_GRADE_ONLY_NAMES:
            subjects.append({
                "subject_name": canonical,
                "marks_obtained": marks,
                "max_marks": 100,
                "grade": None,
            })
    
    # Pattern for grade-only subjects: subject name + single letter grade
    grade_only_pattern = r"(HEALTH.*?PHYSICAL|SCOUT.*?GUID|WATER.*?SECURITY|DEFENCE.*?STUD)\s+([A-E])(?:\s|$)"
    
    for match in re.finditer(grade_only_pattern, raw_text, re.IGNORECASE):
        subject_text = match.group(1).strip()
        grade = match.group(2).upper()
        
        canonical = _fuzzy_match_subject_name(subject_text)
        if canonical:
            subjects.append({
                "subject_name": canonical,
                "marks_obtained": None,
                "max_marks": None,
                "grade": grade,
            })
    
    return subjects


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION STRATEGY 3: FUZZY MATCHING (LAST RESORT)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_by_fuzzy_matching(raw_text: str) -> list:
    """
    Last resort: scan entire text for subject names and nearby marks/grades.
    Less reliable but catches cases where structure is badly mangled.
    """
    subjects = []
    seen_subjects = set()
    
    # List of subject keywords to search for
    subject_keywords = [
        ("MATHEMATICS", "Mathematics"),
        ("SCIENCE", "Science & Technology"),
        ("SOCIAL", "Social Sciences"),
        ("MARATHI", None),  # Will determine 1st/2nd/3rd based on context
        ("HINDI", None),
        ("ENGLISH", None),
        ("SANSKRIT", "Sanskrit (2/3 Lang)"),
    ]
    
    for keyword, canonical_base in subject_keywords:
        # Find all occurrences of keyword
        for match in re.finditer(rf"\b{keyword}\b", raw_text, re.IGNORECASE):
            pos = match.start()
            
            # Look within 100 chars after keyword for marks pattern
            segment = raw_text[pos:pos + 100]
            
            # Try to find marks
            marks_match = re.search(r"100\s+0?(\d{2,3})", segment)
            if marks_match:
                marks = int(marks_match.group(1))
                if marks > 100:
                    continue
                
                # Determine canonical name for language subjects
                if canonical_base is None:
                    if "1ST" in segment.upper() or "FIRST" in segment.upper():
                        canonical = f"{keyword.title()} (1st Lang)"
                    else:
                        canonical = f"{keyword.title()} (2/3 Lang)"
                else:
                    canonical = canonical_base
                
                # Avoid duplicates
                if canonical.lower() not in seen_subjects:
                    subjects.append({
                        "subject_name": canonical,
                        "marks_obtained": marks,
                        "max_marks": 100,
                        "grade": None,
                    })
                    seen_subjects.add(canonical.lower())
    
    return subjects


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SUBJECT EXTRACTION WITH FALLBACK CHAIN
# ═══════════════════════════════════════════════════════════════════════════

def extract_subjects_from_raw_text(raw_text: str) -> list:
    """
    Extract SSC subjects using multiple strategies with fallback chain.
    
    Strategy 1: Code-position based (most precise)
    Strategy 2: Table structure based (handles OCR code errors)
    Strategy 3: Fuzzy matching (last resort for mangled text)
    
    Returns list of subject dicts, or [] if all strategies fail.
    """
    # Detect format version and get rules
    rules = detect_ssc_format_version(raw_text)
    
    # Strategy 1: Code-position extraction (most reliable)
    subjects = _extract_by_code_position(raw_text, rules)
    if len(subjects) >= 4:
        print(f"  ✓ Extracted {len(subjects)} subjects using code-position method")
        return subjects
    
    # Strategy 2: Table structure extraction (more flexible)
    subjects = _extract_by_table_structure(raw_text)
    if len(subjects) >= 4:
        print(f"  ✓ Extracted {len(subjects)} subjects using table-structure method")
        return subjects
    
    # Strategy 3: Fuzzy matching (last resort)
    subjects = _extract_by_fuzzy_matching(raw_text)
    if len(subjects) >= 4:
        print(f"  ⚠ Extracted {len(subjects)} subjects using fuzzy matching (verify recommended)")
        return subjects
    
    print("  ✗ All extraction strategies failed")
    return []


# ═══════════════════════════════════════════════════════════════════════════
# TOTALS AREA HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _totals_area(raw_text: str) -> str:
    """Return the slice of raw_text that starts at the last subject marker."""
    raw_upper = raw_text.upper()
    start = 0
    for marker in _SSC_TOTALS_MARKERS:
        idx = raw_upper.rfind(marker)
        if idx != -1 and idx > start:
            start = idx
    return raw_text[start:] if start > 0 else raw_text


def extract_percentage_from_text(raw_text: str):
    """
    Fallback: find the percentage decimal in the totals area.
    Skips numbers embedded in seat/centre codes (e.g. '33.01.042').
    """
    area = _totals_area(raw_text)
    
    # Look for XX.XX or XX.X format (percentage decimals)
    for m in re.finditer(r"(?<!\d)(\d{2}\.\d{1,2})(?!\d)", area):
        candidate = float(m.group(1))
        if 0.0 <= candidate <= 100.0:
            return candidate
    
    # Try without decimal (e.g., "75" meaning 75%)
    for m in re.finditer(r"\b([1-9]\d)(?:\s|$|%)", area):
        candidate = float(m.group(1))
        if 30.0 <= candidate <= 100.0:
            return candidate
    
    return None


def extract_total_max_marks_from_text(raw_text: str) -> int:
    """
    Extract total max marks (must be 400, 500, or 600).
    Always verifies against raw text to catch LLM errors.
    """
    area = _totals_area(raw_text)
    
    # SSC allows only these totals
    valid_totals = [400, 500, 600]
    
    # Look for explicit "Total: XXX" or "Grand Total: XXX" patterns
    for pattern in [r"TOTAL.*?(\d{3})", r"GRAND.*?(\d{3})", r"MAX.*?MARKS.*?(\d{3})"]:
        m = re.search(pattern, area, re.IGNORECASE)
        if m:
            candidate = int(m.group(1))
            if candidate in valid_totals:
                return candidate
    
    # Fallback: find any valid total in the totals area
    for m in re.finditer(r"\b(\d{3})\b", area):
        candidate = int(m.group(1))
        if candidate in valid_totals:
            return candidate
    
    return None


def extract_total_marks_obtained_from_text(raw_text: str, total_max_marks: int) -> int:
    """
    Extract total marks obtained. Must be < total_max_marks and appear in totals area.
    Avoids false positives from seat numbers and subject marks.
    """
    area = _totals_area(raw_text)
    
    # Look for 3-digit numbers in the totals area that make sense
    candidates = []
    for m in re.finditer(r"\b([1-9]\d{2})\b", area):
        candidate = int(m.group(1))
        if candidate < total_max_marks and candidate >= total_max_marks * 0.5:
            candidates.append(candidate)
    
    # Return the highest valid candidate (usually the actual total)
    if candidates:
        return max(candidates)
    
    # If multiple exact matches for a specific number, it's likely the total
    for m in re.finditer(r"\b([1-9]\d{2})\b", area):
        candidate = int(m.group(1))
        count = len(re.findall(rf"\b{candidate}\b", area))
        if count == 1 and candidate < total_max_marks and candidate >= total_max_marks * 0.5:
            return candidate
    
    # Last resort: first 3-digit number in range
    for m in re.finditer(r"\b([1-9]\d{2})\b", area):
        candidate = int(m.group(1))
        if candidate < total_max_marks and candidate >= total_max_marks * 0.5:
            return candidate
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

def validate_and_fix_ssc_marksheet(data: dict, raw_text: str) -> dict:
    """
    Hard-enforce all SSC extraction rules after the LLM responds.
    
    Features:
    - Multiple extraction strategies with fallback
    - Format version auto-detection
    - Anomaly detection for format changes
    - Comprehensive validation and correction
    
    Steps:
    1. Fix scalar name fields
    2. Replace LLM subjects with regex-parsed subjects from raw text
    3. Enforce grade-only / numeric rules per subject
    4. Inject any grade-only subjects the LLM dropped
    5. Recompute and verify totals & percentage
    6. Flag anomalies for manual review
    
    Args:
        data: LLM-extracted dict (may have errors / missing fields).
        raw_text: Full OCR string from the marksheet image.
    
    Returns:
        Corrected data dict. Caller wraps in success/document_type envelope.
    """
    print("  Running SSC marksheet post-processing...")
    
    anomalies = []
    
    # ── 1. NAMES ────────────────────────────────────────────────────────────
    # mother_name → exactly 1 word
    if data.get("mother_name"):
        words = str(data["mother_name"]).strip().split()
        if len(words) > 1:
            print(f"  FIX mother_name: '{data['mother_name']}' → '{words[0]}'")
            data["mother_name"] = words[0]
    
    # student_name → at most 3 words
    if data.get("student_name"):
        words = str(data["student_name"]).strip().split()
        if len(words) > 3:
            fixed = " ".join(words[:3])
            print(f"  FIX student_name: '{data['student_name']}' → '{fixed}'")
            data["student_name"] = fixed
    
    # Normalize father_name key
    if "fathers_name" in data and "father_name" not in data:
        data["father_name"] = data.pop("fathers_name")
    if data.get("father_name") in ("null", "None", ""):
        data["father_name"] = None
    
    # ── 2. SUBJECTS ─────────────────────────────────────────────────────────
    actual_subjects = extract_subjects_from_raw_text(raw_text)
    
    if actual_subjects:
        data["subjects"] = actual_subjects
        data["_extraction_method"] = "regex"
    else:
        print("  ⚠ Pattern extraction failed — using LLM output with fixes")
        anomalies.append("subject_extraction_failed")
        data["_extraction_method"] = "llm_fallback"
        
        # Deduplicate LLM subjects
        seen, deduped = {}, []
        for subj in data.get("subjects", []):
            name_lower = subj.get("subject_name", "").lower().strip()
            is_grade_only = name_lower in _SSC_GRADE_ONLY_NAMES
            
            if name_lower in seen and not is_grade_only:
                existing_marks = seen[name_lower].get("marks_obtained") or 0
                current_marks = subj.get("marks_obtained") or 0
                if current_marks > existing_marks:
                    print(f"  FIX: Duplicate '{subj['subject_name']}' — keeping marks={current_marks}")
                    deduped = [s for s in deduped if s.get("subject_name", "").lower().strip() != name_lower]
                    deduped.append(subj)
                    seen[name_lower] = subj
                else:
                    print(f"  FIX: Skipping duplicate '{subj['subject_name']}' marks={current_marks}")
                continue
            
            deduped.append(subj)
            if not is_grade_only:
                seen[name_lower] = subj
        
        data["subjects"] = deduped
    
    # ── 3. PER-SUBJECT RULES ────────────────────────────────────────────────
    for subj in data.get("subjects", []):
        name_lower = subj.get("subject_name", "").lower().strip()
        is_grade_only = name_lower in _SSC_GRADE_ONLY_NAMES
        
        if is_grade_only:
            # Grade-only subjects: marks must be null, grade must be valid
            if subj.get("marks_obtained") is not None or subj.get("max_marks") is not None:
                print(f"  FIX grade-only '{subj['subject_name']}': marks → null")
            subj["marks_obtained"] = None
            subj["max_marks"] = None
            
            g = subj.get("grade")
            if isinstance(g, str) and g.strip().upper() in _SSC_VALID_GRADE_LETTERS:
                subj["grade"] = g.strip().upper()
            else:
                if g not in (None, ""):
                    print(f"  FIX grade for '{subj['subject_name']}': '{g}' → 'A'")
                subj["grade"] = "A"
        else:
            # Numeric subjects: grade must be null, marks must be valid
            g = subj.get("grade")
            if g is not None:
                print(f"  FIX numeric subject '{subj['subject_name']}': grade='{g}' → null")
                subj["grade"] = None
            
            mo = subj.get("marks_obtained")
            if mo is not None:
                try:
                    subj["marks_obtained"] = int(float(str(mo)))
                except (ValueError, TypeError):
                    print(f"  FIX marks_obtained for '{subj['subject_name']}': '{mo}' → null")
                    subj["marks_obtained"] = None
            
            subj["max_marks"] = 100
    
    # ── 4. INJECT DROPPED GRADE-ONLY SUBJECTS ───────────────────────────────
    raw_upper = raw_text.upper()
    existing_names = {s.get("subject_name", "").lower().strip() for s in data.get("subjects", [])}
    already_injected: set = set()
    
    for marker, canonical_name in _SSC_GRADE_ONLY_INJECTION:
        canonical_lower = canonical_name.lower()
        if canonical_lower in already_injected:
            continue
        
        if re.search(r"\b" + re.escape(marker) + r"\b", raw_upper) and canonical_lower not in existing_names:
            print(f"  INJECT missing grade-only subject: '{canonical_name}' (marker='{marker}')")
            data["subjects"].append({
                "subject_name": canonical_name,
                "marks_obtained": None,
                "max_marks": None,
                "grade": "A",
            })
            existing_names.add(canonical_lower)
            already_injected.add(canonical_lower)
    
    # ── 5. TOTALS ────────────────────────────────────────────────────────────
    # total_max_marks — must be 400 / 500 / 600; always verify against raw text
    tmm = data.get("total_max_marks")
    try:
        tmm = int(float(str(tmm))) if tmm not in (None, "null", "") else None
    except (ValueError, TypeError):
        tmm = None
    
    tmm_text = extract_total_max_marks_from_text(raw_text)
    if tmm != tmm_text:
        print(f"  FIX total_max_marks: LLM={tmm} → text={tmm_text}")
        tmm = tmm_text
    data["total_max_marks"] = tmm
    
    # total_marks_obtained — always verify against raw text
    tmo = data.get("total_marks_obtained")
    try:
        tmo = int(float(str(tmo))) if tmo not in (None, "null", "") else None
    except (ValueError, TypeError):
        tmo = None
    
    if tmm is not None:
        tmo_text = extract_total_marks_obtained_from_text(raw_text, tmm)
        if tmo != tmo_text:
            print(f"  FIX total_marks_obtained: LLM={tmo} → text={tmo_text}")
            tmo = tmo_text
        
        if tmo is not None and tmo > tmm:
            print(f"  FIX: total_marks_obtained {tmo} > total_max_marks {tmm} → null")
            tmo = None
    
    data["total_marks_obtained"] = tmo
    
    # percentage — must be in range AND appear in the totals area of the document
    pct = data.get("percentage")
    try:
        pct = float(str(pct)) if pct not in (None, "null", "") else None
    except (ValueError, TypeError):
        pct = None
    
    if pct is not None and 0.0 <= pct <= 100.0:
        pct_str = f"{pct:.2f}"
        if not re.search(re.escape(pct_str) + r"(?!\d)(?!\.)", raw_text):
            print(f"  FIX percentage: {pct} not found in totals area → re-extracting")
            pct = None
    
    if pct is None or not (0.0 <= pct <= 100.0):
        if pct is not None:
            print(f"  FIX percentage: {pct} out of range → re-extracting")
        pct = extract_percentage_from_text(raw_text)
        print(f"  percentage from raw_text: {pct}")
    
    data["percentage"] = round(pct, 2) if pct is not None else None
    
    # ── 6. ANOMALY DETECTION ─────────────────────────────────────────────────
    if _detect_potential_format_change(raw_text):
        anomalies.append("possible_format_change")
        print("  ⚠️  Possible format change detected - review recommended")
    
    # Check for missing critical data
    if not data.get("student_name"):
        anomalies.append("missing_student_name")
    if not data.get("subjects") or len(data["subjects"]) < 5:
        anomalies.append("insufficient_subjects")
    if data.get("percentage") is None:
        anomalies.append("missing_percentage")
    
    # Add anomalies to data
    if anomalies:
        data["_extraction_warnings"] = anomalies
        data["_needs_manual_review"] = True
        print(f"  ⚠️  ANOMALIES: {', '.join(anomalies)}")
    
    print(
        f"  SSC post-processing done: "
        f"mother='{data.get('mother_name')}' | "
        f"pct={data.get('percentage')} | "
        f"max={data.get('total_max_marks')} | "
        f"obtained={data.get('total_marks_obtained')} | "
        f"subjects={len(data.get('subjects', []))}"
    )
    
    return data


