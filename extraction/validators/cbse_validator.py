"""
CBSE (Central Board of Secondary Education) Class 10 marksheet post-processing.

All functions are pure — they take data dicts and raw OCR text, return
corrected data dicts. No LLM calls, no model state, no side effects.

Called from LlamaJSONExtractor._validate_and_fix_extraction when
detect_marksheet_board(raw_text) == 'cbse'.

Output fields (CBSE marksheet only contains):
  student_name, mother_name, father_name, exam_year, roll_number,
  result, board, document_type, subjects, additional_subjects
  
NOT included (not printed on CBSE marksheet):
  date_of_birth, total_marks_obtained, total_max_marks, percentage
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT MASTER DATA
# ─────────────────────────────────────────────────────────────────────────────

_CODE_TO_NAME = {
    "001": "Hindi Course-A",        "002": "Hindi Course-B",
    "003": "Urdu Course-A",         "004": "Punjabi",
    "005": "Bengali",               "006": "Tamil",
    "007": "Telugu",                "008": "Sindhi",
    "009": "Marathi",               "010": "Gujarati",
    "011": "Manipuri",              "012": "Malayalam",
    "013": "Odia",                  "014": "Assamese",
    "015": "Kannada",               "018": "French",
    "019": "German",                "020": "Russian",
    "022": "Nepali",                "031": "English Communicative",
    "041": "Mathematics Standard",  "241": "Mathematics Basic",
    "042": "Science",               "049": "Painting",
    "054": "Elements of Book Keeping & Accountancy",
    "064": "Home Science",          "065": "Computer Applications",
    "066": "Elements of Business",  "067": "Sanskrit",
    "084": "English Language & Literature",
    "085": "Hindi Course-B",        "086": "Science",
    "087": "Social Science",        "119": "Sanskrit (Communicative)",
    "122": "Sanskrit",              "165": "Hindi Course-A",
    "184": "English Language & Literature",
    "402": "Information Technology",
    "404": "Artificial Intelligence",   # pre-2024 code
    "417": "Artificial Intelligence",   # 2024+ code
    "436": "National Cadet Corps",
}

# (regex_pattern, canonical_name) — tried in order; first full/partial match wins
_NAME_ALIASES = [
    (r"ENGLISH\s+L(?:NG|ANG(?:UAGE)?)\s*&\s*LIT\.?", "English Language & Literature"),
    (r"ENGLISH\s+COMMUNICATIVE",                       "English Communicative"),
    (r"ENGLISH",                                       "English Language & Literature"),
    (r"HINDI\s+COURSE[-\s]*A",                         "Hindi Course-A"),
    (r"HINDI\s+COURSE[-\s]*B",                         "Hindi Course-B"),
    (r"HINDI",                                         "Hindi Course-A"),
    (r"SANSKRIT\s*\(?COMMUNICATIVE\)?",                "Sanskrit (Communicative)"),
    (r"SANSKRIT",                                      "Sanskrit"),
    (r"MATHEMATICS?\s+STANDARD",                       "Mathematics Standard"),
    (r"MATHEMATICS?\s+BASIC",                          "Mathematics Basic"),
    (r"MATH(?:EMATICS?)?",                             "Mathematics Standard"),
    (r"SCIENCE\s*&?\s*TECHNOLOGY",                     "Science"),
    (r"SCIENCE",                                       "Science"),
    (r"SOCIAL\s+SCIENCE",                              "Social Science"),
    (r"SST\b",                                         "Social Science"),
    (r"INFORMATION\s+TECHNOLOGY",                      "Information Technology"),
    (r"IT\b",                                          "Information Technology"),
    (r"ARTIFICIAL\s+INTELLIGENCE",                     "Artificial Intelligence"),
    (r"COMPUTER\s+APPLICATIONS?",                      "Computer Applications"),
    (r"HOME\s+SCIENCE",                                "Home Science"),
    (r"PAINTING",                                      "Painting"),
    (r"FRENCH",                                        "French"),
    (r"GERMAN",                                        "German"),
    (r"URDU",                                          "Urdu Course-A"),
    (r"PUNJABI",                                       "Punjabi"),
    (r"BENGALI",                                       "Bengali"),
    (r"MARATHI",                                       "Marathi"),
    (r"GUJARATI",                                      "Gujarati"),
    (r"KANNADA",                                       "Kannada"),
    (r"TAMIL",                                         "Tamil"),
    (r"TELUGU",                                        "Telugu"),
    (r"MALAYALAM",                                     "Malayalam"),
    (r"ELEMENTS?\s+OF\s+BOOK\s*KEEPING",               "Elements of Book Keeping & Accountancy"),
    (r"ELEMENTS?\s+OF\s+BUSINESS",                     "Elements of Business"),
    (r"(?:NCC|NATIONAL\s+CADET\s+CORPS)",              "National Cadet Corps"),
]

# Subjects always graded with a letter (no numeric marks)
_GRADE_ONLY_SUBJECTS = {
    "national cadet corps",
    "work experience",
    "art education",
    "health & physical education",
    "health and physical education",
}

# Codes for additional / optional subjects — NOT counted in main subjects list
_ADDITIONAL_CODES = {
    "402", "404", "417",              # IT / AI variants
    "049", "064", "065", "066",       # Painting, Home Sci, Comp Apps, Elem Business
    "054", "436",                     # Book Keeping, NCC
}

# CBSE 9-point grading scale  (lo, hi inclusive → grade)
_GRADE_BANDS = [
    (91, 100, "A1"), (81, 90, "A2"),
    (71, 80,  "B1"), (61, 70, "B2"),
    (51, 60,  "C1"), (41, 50, "C2"),
    (33, 40,  "D"),  (21, 32, "E1"),
    (0,  20,  "E2"),
]
_VALID_GRADES = {"A1", "A2", "B1", "B2", "C1", "C2", "D", "E1", "E2", "E"}

# Compiled subject row patterns
_PAT_A = re.compile(          # full columns: theory + IA/PR + total
    r"(\d{3})\s+"
    r"((?:[A-Z][A-Z0-9\s&./:'\(\)\-]+?)+?)\s+"
    r"(\d{2,3})\s+(\d{2,3})\s+(\d{2,3})\s+"
    r"([A-Z][A-Z\s]+?)\s+"
    r"(A1|A2|B1|B2|C1|C2|D|E1|E2|E)\b",
    re.IGNORECASE,
)
_PAT_B = re.compile(          # single marks column (no theory/IA split)
    r"(\d{3})\s+"
    r"((?:[A-Z][A-Z0-9\s&./:'\(\)\-]+?)+?)\s+"
    r"(\d{2,3})\s+"
    r"([A-Z][A-Z\s]+?)\s+"
    r"(A1|A2|B1|B2|C1|C2|D|E1|E2|E)\b",
    re.IGNORECASE,
)
_PAT_C = re.compile(          # grade-only (co-scholastic)
    r"(\d{3})\s+"
    r"((?:[A-Z][A-Z0-9\s&./:'\(\)\-]+?)+?)\s+"
    r"(A\+?|B\+?|C\+?|[A-E])\b",
    re.IGNORECASE,
)

# Hindi/Devanagari Unicode block — used to strip OCR bleed into name fields
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")

# Trailing noise after a valid all-caps name.
# Triggers only on: lowercase-starting words, TitleCase words, known fillers.
# Does NOT strip short all-caps — Indian names can be short (e.g. RIYA, OM).
_NAME_NOISE_RE = re.compile(
    r"\s+(?:"
    r"[a-z]\w*"                              # starts with lowercase  → noise
    r"|[A-Z][a-z]\w*"                        # TitleCase word         → noise
    r"|\b(?:OR|AND|THE)\b"                  # known filler tokens    → noise
    r").*$",
)


# ─────────────────────────────────────────────────────────────────────────────
# PURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def marks_to_grade(marks: int) -> str:
    """Convert numeric marks to CBSE 9-point grade string."""
    for lo, hi, grade in _GRADE_BANDS:
        if lo <= marks <= hi:
            return grade
    return "E2"


def grade_to_marks_range(grade: str):
    """Return (lo, hi) for a grade, or None if grade is unrecognised."""
    for lo, hi, g in _GRADE_BANDS:
        if g == grade:
            return lo, hi
    return None


def canonical_subject_name(code: str, ocr_name: str) -> str:
    """
    Return canonical subject name.
    Priority: code lookup → full alias match → partial alias match → title-case fallback.
    """
    if code and code in _CODE_TO_NAME:
        return _CODE_TO_NAME[code]
    clean = ocr_name.strip()
    for pattern, name in _NAME_ALIASES:
        if re.fullmatch(pattern, clean, re.IGNORECASE):
            return name
    for pattern, name in _NAME_ALIASES:
        if re.search(pattern, clean, re.IGNORECASE):
            return name
    return clean.title()


def _clean_name_field(raw_value: str) -> str:
    """
    Remove Hindi OCR bleed-through and mixed-case noise from a name field.

    Strategy:
      1. Strip Devanagari characters entirely.
      2. Strip any suffix that starts with a lowercase word or known filler.
      3. Collapse whitespace.
      4. Keep only words that are entirely uppercase letters (valid name tokens).
    """
    if not raw_value:
        return raw_value

    # Remove Devanagari script bleed
    cleaned = _DEVANAGARI_RE.sub(" ", raw_value)

    # Strip trailing noise starting at first mixed-case or filler word
    cleaned = _NAME_NOISE_RE.sub("", cleaned)

    # Keep only ALL-CAPS words (genuine name tokens are all uppercase on CBSE)
    words = [w for w in cleaned.split() if re.fullmatch(r"[A-Z]+", w)]

    return " ".join(words).strip()


# ─────────────────────────────────────────────────────────────────────────────
# SCALAR FIELD EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def extract_student_name(raw_text: str):
    """
    CBSE format: 'This is to certify that NAME Roll No.'
    Returns the cleaned ALL-CAPS name, or None.
    """
    m = re.search(
        r"certify\s+that\s+([A-Z][A-Za-z\s]{4,80}?)\s+Roll\s+No",
        raw_text, re.IGNORECASE,
    )
    if m:
        name = _clean_name_field(m.group(1))
        if name and len(name) >= 3:
            return name
    return None


def extract_parent_name(raw_text: str, parent: str):
    """
    Extract mother or father name from CBSE marksheet.
    Returns only the first word for mother (CBSE standard — only first name printed).
    Cleans Hindi OCR noise from both.
    """
    if parent == "mother":
        patterns = [
            r"Mother['\u2019]?s?\s+Name\s+([A-Z][A-Za-z\s]{1,60}?)(?:\s{2,}|\n|Father|Guardian|Date)",
            r"Mother['\u2019]?s?\s+Name\s+([A-Z][A-Za-z\s]{1,50})",
        ]
    else:
        patterns = [
            r"(?:Father|Guardian)['\u2019]?s?\s*(?:/\s*(?:Guardian|Father)['\u2019]?s?)?\s+Name\s+"
            r"([A-Z][A-Za-z\s]{1,60}?)(?:\s{2,}|\n|Date|School|\u091C)",
            r"(?:Father|Guardian)['\u2019]?s?\s+Name\s+([A-Z][A-Za-z\s]{1,60})",
        ]
    for pat in patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            val = _clean_name_field(m.group(1))
            if val and len(val) >= 2:
                # Mother: CBSE prints only first name
                return val.split()[0] if parent == "mother" else val
    return None


def extract_roll_number(raw_text: str):
    """
    CBSE roll numbers are 7–9 digits after 'Roll No.'
    Allows up to 9 digits to handle OCR correctly reading 8-digit rolls
    that shorter patterns might truncate.
    """
    m = re.search(r"Roll\s+No\.?\s*[:\-]?\s*(\d{7,9})\b", raw_text, re.IGNORECASE)
    return m.group(1) if m else None


def extract_exam_year(raw_text: str):
    """Pull 4-digit year from 'SECONDARY SCHOOL EXAMINATION, YYYY'."""
    m = re.search(r"SECONDARY\s+SCHOOL\s+EXAMINATION[,\s]+(\d{4})", raw_text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"EXAMINATION[,\s]+(\d{4})", raw_text, re.IGNORECASE)
    return m.group(1) if m else None


def extract_result(raw_text: str) -> str:
    """Parse PASS / FAIL / COMPARTMENT from the Result line."""
    m = re.search(
        r"\bResult\b\s+(PASS|FAIL|COMPARTMENT|ESSENTIAL\s+REPEAT)\b",
        raw_text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper().strip()
    if re.search(r"\bPASS\b", raw_text, re.IGNORECASE):
        return "PASS"
    if re.search(r"\bFAIL\b", raw_text, re.IGNORECASE):
        return "FAIL"
    return "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def extract_subjects_from_raw_text(raw_text: str) -> list:
    """
    Parse CBSE subject rows directly from OCR text — bypasses LLM entirely.

    Handles three row formats (tried in order per line):
      A) CODE  NAME  THEORY  IA  TOTAL  WORDS  GRADE   — standard 2017+
      B) CODE  NAME  TOTAL   WORDS  GRADE               — simplified / older OCR
      C) CODE  NAME  GRADE                              — co-scholastic, grade-only

    Returns list of subject dicts (with '_is_additional' flag), or []
    if fewer than 4 subjects are found.
    """
    subjects: list = []
    seen_codes: set = set()
    is_additional_section = False

    for line in raw_text.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue

        if re.search(r"ADDITIONAL\s+SUBJECT", line_clean, re.IGNORECASE):
            is_additional_section = True
            continue

        # Pattern A — full columns (theory + IA + total)
        m = _PAT_A.search(line_clean)
        if m:
            code, name_raw = m.group(1), m.group(2).strip()
            theory, ia_pr, total = int(m.group(3)), int(m.group(4)), int(m.group(5))
            grade = m.group(7).upper()
            if code in seen_codes:
                continue
            seen_codes.add(code)
            if grade not in _VALID_GRADES:
                grade = marks_to_grade(total)
            subjects.append({
                "subject_code":   code,
                "subject_name":   canonical_subject_name(code, name_raw),
                "theory_marks":   theory,
                "ia_pr_marks":    ia_pr,
                "marks_obtained": total,
                "max_marks":      100,
                "grade":          grade,
                "_is_additional": is_additional_section or code in _ADDITIONAL_CODES,
            })
            continue

        # Pattern B — single marks column
        m = _PAT_B.search(line_clean)
        if m:
            code, name_raw = m.group(1), m.group(2).strip()
            total = int(m.group(3))
            grade = m.group(5).upper()
            if code in seen_codes:
                continue
            seen_codes.add(code)
            if grade not in _VALID_GRADES:
                grade = marks_to_grade(total)
            subjects.append({
                "subject_code":   code,
                "subject_name":   canonical_subject_name(code, name_raw),
                "theory_marks":   None,
                "ia_pr_marks":    None,
                "marks_obtained": total,
                "max_marks":      100,
                "grade":          grade,
                "_is_additional": is_additional_section or code in _ADDITIONAL_CODES,
            })
            continue

        # Pattern C — grade-only co-scholastic
        m = _PAT_C.search(line_clean)
        if m:
            code, name_raw, grade = m.group(1), m.group(2).strip(), m.group(3).upper()
            cname = canonical_subject_name(code, name_raw)
            if cname.lower() not in _GRADE_ONLY_SUBJECTS:
                continue
            if code in seen_codes:
                continue
            seen_codes.add(code)
            subjects.append({
                "subject_code":   code,
                "subject_name":   cname,
                "theory_marks":   None,
                "ia_pr_marks":    None,
                "marks_obtained": None,
                "max_marks":      None,
                "grade":          grade,
                "_is_additional": True,
            })

    return subjects if len(subjects) >= 4 else []


def _fix_marks_against_grade(subj: dict) -> dict:
    """
    Cross-check marks_obtained against grade.
    If the marks don't fall in the grade's band, the LLM misread a digit
    (e.g. OCR '088' → LLM '68'). In that case, replace marks with the
    midpoint of the grade band as best estimate and log the fix.
    """
    grade = subj.get("grade", "")
    mo = subj.get("marks_obtained")
    if mo is None or grade not in _VALID_GRADES:
        return subj

    band = grade_to_marks_range(grade)
    if band is None:
        return subj

    lo, hi = band
    if not (lo <= mo <= hi):
        # marks inconsistent with grade — use midpoint of correct band
        corrected = (lo + hi) // 2
        print(
            f"  FIX marks_obtained for '{subj.get('subject_name')}': "
            f"{mo} outside grade {grade} range [{lo}-{hi}] → {corrected} (midpoint estimate)"
        )
        subj["marks_obtained"] = corrected

    return subj


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_fix_cbse_marksheet(data: dict, raw_text: str) -> dict:
    """
    Hard-enforce all CBSE Class 10 extraction rules after the LLM responds.

    Output fields kept (present on CBSE marksheet):
        student_name, mother_name, father_name, exam_year, roll_number,
        result, board, document_type, subjects, additional_subjects

    Fields removed (NOT printed on CBSE marksheet):
        date_of_birth, total_marks_obtained, total_max_marks, percentage
    """
    print("  Running CBSE marksheet post-processing...")

    # ── 1. SCALAR FIELDS ─────────────────────────────────────────────────────

    # student_name — strip Hindi OCR noise from LLM output, then re-extract from raw
    raw_name = extract_student_name(raw_text)
    if raw_name:
        llm_name = _clean_name_field(data.get("student_name") or "")
        if llm_name.upper() != raw_name.upper():
            print(f"  FIX student_name: '{data.get('student_name')}' → '{raw_name}'")
        data["student_name"] = raw_name
    else:
        # raw_text parse failed — at least clean the LLM value
        cleaned = _clean_name_field(data.get("student_name") or "")
        if cleaned != data.get("student_name"):
            print(f"  FIX student_name (noise strip): '{data.get('student_name')}' → '{cleaned}'")
        data["student_name"] = cleaned or None

    # mother_name — first word only (CBSE standard)
    raw_mother = extract_parent_name(raw_text, "mother")
    if raw_mother:
        if (data.get("mother_name") or "").upper() != raw_mother.upper():
            print(f"  FIX mother_name: '{data.get('mother_name')}' → '{raw_mother}'")
        data["mother_name"] = raw_mother
    else:
        # clean whatever LLM gave us and take first word
        cleaned = _clean_name_field(data.get("mother_name") or "")
        data["mother_name"] = cleaned.split()[0] if cleaned else None

    # father_name — full name, cleaned
    raw_father = extract_parent_name(raw_text, "father")
    if raw_father:
        if (data.get("father_name") or "").upper() != raw_father.upper():
            print(f"  FIX father_name: '{data.get('father_name')}' → '{raw_father}'")
        data["father_name"] = raw_father
    else:
        cleaned = _clean_name_field(data.get("father_name") or "")
        if cleaned != data.get("father_name"):
            print(f"  FIX father_name (noise strip): '{data.get('father_name')}' → '{cleaned}'")
        data["father_name"] = cleaned or None

    # Normalise key variant the LLM sometimes uses
    if "fathers_name" in data and "father_name" not in data:
        data["father_name"] = data.pop("fathers_name")

    # roll_number
    raw_roll = extract_roll_number(raw_text)
    if raw_roll:
        if str(data.get("roll_number", "")) != raw_roll:
            print(f"  FIX roll_number: '{data.get('roll_number')}' → '{raw_roll}'")
        data["roll_number"] = raw_roll
    elif not data.get("roll_number"):
        data["roll_number"] = None

    # exam_year
    raw_year = extract_exam_year(raw_text)
    if raw_year:
        if str(data.get("exam_year", "")) != raw_year:
            print(f"  FIX exam_year: '{data.get('exam_year')}' → '{raw_year}'")
        data["exam_year"] = raw_year

    # result
    raw_result = extract_result(raw_text)
    if (data.get("result") or "").upper() != raw_result:
        print(f"  FIX result: '{data.get('result')}' → '{raw_result}'")
    data["result"] = raw_result

    # Board stamp
    data["board"] = "CBSE"
    data["document_type"] = "marksheet"

    # ── 2. REMOVE FIELDS NOT ON CBSE MARKSHEET ───────────────────────────────
    for field in ("date_of_birth", "total_marks_obtained", "total_max_marks", "percentage"):
        if field in data:
            print(f"  REMOVE '{field}' (not present on CBSE marksheet)")
            del data[field]

    # ── 3. SUBJECTS ───────────────────────────────────────────────────────────

    parsed = extract_subjects_from_raw_text(raw_text)

    if parsed:
        print(f"  Parsed {len(parsed)} subjects from raw text (replacing LLM output)")
        data["subjects"] = parsed
    else:
        print("  ⚠ Could not parse subjects from raw text — validating LLM subjects")
        cleaned_list, seen = [], set()

        for subj in data.get("subjects", []):
            name = subj.get("subject_name", "").strip()
            name_lower = name.lower()

            if name_lower in seen:
                print(f"  FIX: skipping duplicate subject '{name}'")
                continue
            seen.add(name_lower)

            subj["subject_name"] = canonical_subject_name(subj.get("subject_code", ""), name)

            if name_lower in _GRADE_ONLY_SUBJECTS:
                subj["marks_obtained"] = None
                subj["max_marks"] = None
                if subj.get("grade") not in _VALID_GRADES:
                    subj["grade"] = "A"
            else:
                mo = subj.get("marks_obtained")
                try:
                    mo = int(float(str(mo))) if mo not in (None, "null", "") else None
                except (ValueError, TypeError):
                    mo = None
                subj["marks_obtained"] = mo
                subj["max_marks"] = 100

                g = subj.get("grade", "")
                if g not in _VALID_GRADES and mo is not None:
                    computed = marks_to_grade(mo)
                    print(f"  FIX grade for '{name}': '{g}' → '{computed}'")
                    subj["grade"] = computed
                elif g not in _VALID_GRADES:
                    subj["grade"] = None

                # Cross-check: marks must be consistent with grade
                subj = _fix_marks_against_grade(subj)

            subj.setdefault("_is_additional", subj.get("subject_code", "") in _ADDITIONAL_CODES)
            cleaned_list.append(subj)

        data["subjects"] = cleaned_list

    # ── 4. SPLIT MAIN vs ADDITIONAL ───────────────────────────────────────────

    main_subjects, additional_subjects = [], []
    for subj in data.get("subjects", []):
        is_add = subj.pop("_is_additional", False) or subj.get("subject_code", "") in _ADDITIONAL_CODES
        (additional_subjects if is_add else main_subjects).append(subj)

    data["subjects"] = main_subjects
    data["additional_subjects"] = additional_subjects

    print(
        f"  CBSE post-processing done: "
        f"student='{data.get('student_name')}' | "
        f"roll={data.get('roll_number')} | "
        f"subjects={len(main_subjects)} main + {len(additional_subjects)} additional | "
        f"result={data.get('result')}"
    )
    return data