"""
SSC (Maharashtra State Board) marksheet post-processing.

All functions are pure — they take data dicts and raw OCR text, return
corrected data dicts. No LLM calls, no model state, no side effects.

Called from LlamaJSONExtractor._validate_and_fix_extraction when
detect_marksheet_board(raw_text) == 'ssc'.
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT CODE → CANONICAL NAME MAP
# ─────────────────────────────────────────────────────────────────────────────

_SSC_CODE_MAP = {
    "01":  "Marathi (1st Lang)",
    "02":  "Hindi (1st Lang)",
    "03":  "English (1st Lang)",
    "15":  "Hindi (2/3 Lang)",
    "16":  "Marathi (2/3 Lang)",
    "17":  "English (2/3 Lang)",
    "151": "Hindi (2/3 Lang)",   # alternative code
    "27":  "Sanskrit (2/3 Lang)",
    "71":  "Mathematics",
    "72":  "Science & Technology",
    "72:": "Science & Technology",
    "725": "Science & Technology",
    "73":  "Social Sciences",
    "73:": "Social Sciences",
    "P1":  "Health & Physical Education",
    "P2":  "Scouting/Guiding",
    "P4":  "Defence Studies",
    "P41": "Defence Studies",
    "PA":  "Defence Studies",
    "R8":  "Water Security",
    "RB":  "Water Security",
}

# Subjects whose marks are always null — only a grade letter
_SSC_GRADE_ONLY_NAMES = {
    "health & physical education",
    "scouting/guiding",
    "water security",
    "defence studies",
}

# (raw_text_marker, canonical_name) pairs for injection of dropped subjects
_SSC_GRADE_ONLY_INJECTION = [
    ("P1",             "Health & Physical Education"),
    ("HEALTH & PHYSICAL", "Health & Physical Education"),
    ("P2",             "Scouting/Guiding"),
    ("SCOUTING",       "Scouting/Guiding"),
    ("R8",             "Water Security"),
    ("WATER SECURITY", "Water Security"),
    ("RB",             "Water Security"),
    ("P4",             "Defence Studies"),
    ("PA",             "Defence Studies"),
    ("DEFENCE STUDIES","Defence Studies"),
]

_SSC_VALID_GRADE_LETTERS = {"A", "B", "C", "D", "E"}

# Markers that appear at / after the last subject row — used to locate totals area
_SSC_TOTALS_MARKERS = [
    "WATER SECURITY", "SCOUTING", "HEALTH & PHYSICAL", "SOCIAL SCIENCES"
]


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_subjects_from_raw_text(raw_text: str) -> list:
    """
    Extract SSC subjects directly from OCR text — bypasses LLM entirely.

    Uses code-position splitting: finds each subject code, grabs the text
    segment between it and the next code, then parses marks or grade from
    that segment.

    Returns list of subject dicts, or [] if fewer than 4 subjects found.
    """
    subjects = []

    all_codes = "|".join(re.escape(code) for code in _SSC_CODE_MAP.keys())
    code_pattern = rf"\b({all_codes})\s+"
    code_positions = [
        (m.group(1), m.start(), m.end())
        for m in re.finditer(code_pattern, raw_text)
    ]

    if not code_positions:
        return []

    for i, (code, _start, end) in enumerate(code_positions):
        segment = (
            raw_text[end : code_positions[i + 1][1]].strip()
            if i < len(code_positions) - 1
            else raw_text[end:].strip()
        )

        # Normalise code variants (e.g. "725" → "72", "73:" → "73")
        code_clean = code.replace(":", "").replace("5", "")

        if "100" in segment:
            # Numeric subject: marks appear after "100 <obtained>"
            m = re.search(r"100\s+0?(\d{2,3})\s+([A-Z]+)", segment)
            if m:
                subjects.append({
                    "subject_name":   _SSC_CODE_MAP.get(code, _SSC_CODE_MAP.get(code_clean, "Unknown")),
                    "marks_obtained": int(m.group(1)),
                    "max_marks":      100,
                    "grade":          None,
                })
        else:
            # Grade-only subject: find the single letter grade
            m = re.search(r"([A-E])(?:\s|$|[^A-Z])", segment)
            if m:
                subjects.append({
                    "subject_name":   _SSC_CODE_MAP.get(code, _SSC_CODE_MAP.get(code_clean, "Unknown")),
                    "marks_obtained": None,
                    "max_marks":      None,
                    "grade":          m.group(1),
                })

    return subjects if len(subjects) >= 4 else []


# ─────────────────────────────────────────────────────────────────────────────
# TOTALS AREA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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
    for m in re.finditer(r"(?<!\d)(\d{1,3}\.\d{2})(?!\d)(?!\.)", area):
        val = float(m.group(1))
        if 0.0 <= val <= 100.0:
            return val
    return None


def extract_total_max_marks_from_text(raw_text: str):
    """
    Find 400, 500, or 600 in the totals area (whole token only).
    Returns the first match — not the largest — to avoid false positives.
    """
    area = _totals_area(raw_text)
    for candidate in [400, 500, 600]:
        if re.search(r"(?<!\d)" + str(candidate) + r"(?!\d)", area):
            return candidate
    return None


def extract_total_marks_obtained_from_text(raw_text: str, total_max_marks: int):
    """
    Find total_marks_obtained in the totals area.
    Validates candidate: must be < total_max_marks and ≥ 50% of it.
    """
    area = _totals_area(raw_text)

    patterns = [
        r"TOTA[CL]E?\s+MARKS?\s+(\d{3})",      # "Total Marks 456" / "Totace Marks 456"
        r"MARKS?\s+(\d{3})",                     # "Marks 456"
        str(total_max_marks) + r"\s+(\d{3})",   # "500 456"
    ]
    for pattern in patterns:
        m = re.search(pattern, area.upper())
        if m:
            candidate = int(m.group(1))
            if 0 < candidate < total_max_marks and candidate >= total_max_marks * 0.5:
                return candidate

    # Last resort: first 3-digit number in range
    for m in re.finditer(r"\b([1-9]\d{2})\b", area):
        candidate = int(m.group(1))
        if candidate < total_max_marks and candidate >= total_max_marks * 0.5:
            return candidate

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_fix_ssc_marksheet(data: dict, raw_text: str) -> dict:
    """
    Hard-enforce all SSC extraction rules after the LLM responds.

    Steps:
      1. Fix scalar name fields
      2. Replace LLM subjects with regex-parsed subjects from raw text
      3. Enforce grade-only / numeric rules per subject
      4. Inject any grade-only subjects the LLM dropped
      5. Recompute and verify totals & percentage

    Args:
        data:     LLM-extracted dict (may have errors / missing fields).
        raw_text: Full OCR string from the marksheet image.

    Returns:
        Corrected data dict. Caller wraps in success/document_type envelope.
    """
    print("  Running SSC marksheet post-processing...")

    # ── 1. NAMES ─────────────────────────────────────────────────────────────

    # mother_name → exactly 1 word
    if data.get("mother_name"):
        words = str(data["mother_name"]).strip().split()
        if len(words) > 1:
            print(f"  FIX mother_name: '{data['mother_name']}' → '{words[0]}'")
        data["mother_name"] = words[0] if words else None

    # student_name → at most 3 words
    if data.get("student_name"):
        words = str(data["student_name"]).strip().split()
        if len(words) > 3:
            fixed = " ".join(words[:3])
            print(f"  FIX student_name: '{data['student_name']}' → '{fixed}'")
            data["student_name"] = fixed

    # Normalise father_name key
    if "fathers_name" in data and "father_name" not in data:
        data["father_name"] = data.pop("fathers_name")
    if data.get("father_name") in ("null", "None", ""):
        data["father_name"] = None

    # ── 2. SUBJECTS ───────────────────────────────────────────────────────────

    actual_subjects = extract_subjects_from_raw_text(raw_text)

    if actual_subjects:
        print(f"  Found {len(actual_subjects)} subjects in raw text")
        data["subjects"] = actual_subjects
    else:
        print("  ⚠ Could not parse subjects from raw text — using LLM output with fixes")
        seen, deduped = {}, []

        for subj in data.get("subjects", []):
            name_lower = subj.get("subject_name", "").lower().strip()
            is_grade_only = name_lower in _SSC_GRADE_ONLY_NAMES

            if name_lower in seen and not is_grade_only:
                existing_marks = seen[name_lower].get("marks_obtained") or 0
                current_marks  = subj.get("marks_obtained") or 0
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

    # ── 3. PER-SUBJECT RULES ─────────────────────────────────────────────────

    for subj in data.get("subjects", []):
        name_lower = subj.get("subject_name", "").lower().strip()
        is_grade_only = name_lower in _SSC_GRADE_ONLY_NAMES

        if is_grade_only:
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

    # ── 4. INJECT DROPPED GRADE-ONLY SUBJECTS ────────────────────────────────

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
                "subject_name":   canonical_name,
                "marks_obtained": None,
                "max_marks":      None,
                "grade":          "A",
            })
            existing_names.add(canonical_lower)
            already_injected.add(canonical_lower)

    # ── 5. TOTALS ─────────────────────────────────────────────────────────────

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

    print(
        f"  SSC post-processing done: "
        f"mother='{data.get('mother_name')}' | "
        f"pct={data.get('percentage')} | "
        f"max={data.get('total_max_marks')} | "
        f"obtained={data.get('total_marks_obtained')}"
    )
    return data