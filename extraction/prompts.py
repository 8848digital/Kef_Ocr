"""
Document Extraction Prompts
Contains all prompts for extracting structured data from OCR text
"""


def detect_marksheet_board(raw_text: str) -> str:
    """
    Detect which board issued the marksheet.
    Returns: 'ssc', 'cbse', 'icse', or 'unknown'
    """
    text_upper = raw_text.upper()
    
    # CBSE indicators
    if any(indicator in text_upper for indicator in [
        'CENTRAL BOARD OF SECONDARY EDUCATION',
        'CBSE',
        'CENTRAL BOARD OF: SECONDARY EDUCATION' 
    ]):
        return 'cbse'
    
    # ICSE indicators
    if any(indicator in text_upper for indicator in [
        'COUNCIL FOR THE INDIAN SCHOOL',
        'CISCE',
        'INDIAN SCHOOL CERTIFICATE',
    ]):
        return 'icse'
    
    # SSC indicators
    if any(indicator in text_upper for indicator in [
        'MAHARASHTRA STATE BOARD',
        'MSBSHSE',
        'MUMBAI DIVISIONAL BOARD',
        'PUNE DIVISIONAL BOARD',
        'NAGPUR DIVISIONAL BOARD',
        'DIVISIONAL BOARD',
    ]) or any(code in raw_text for code in ['(1ST LANG)', '(2/3 LANG)']):
        return 'ssc'
    
    return 'unknown'


def get_schema_for_doc_type(doc_type: str) -> dict:
    """Get expected JSON schema for each document type"""
    
    schemas = {
        'pass_book': {
            "account_holder_name": "string or null",
            "account_number": "string or null",
            "parent_name": "string or null",
            "address": "string or null"
        },
        'aadhaar': {
            "document_type": "aadhaar",
            "aadhaar_number": "string or null",
            "name": "string or null",
            "father_name": "string or null",
            "mother_name": "string or null",
            "date_of_birth": "string or null",
            "gender": "string or null",
            "address": "string or null"
        },
        'pan': {
            "document_type": "pan",
            "pan_number": "string or null",
            "name": "string or null",
            "father_name": "string or null",
            "date_of_birth": "string or null"
        },
        'income_certificate': {
            "document_type": "income_certificate",
            "parent_name": "string or null",
            "student_name": "string or null",
            "income_years": "array of objects",
            "address": "string or null",
            "validity_date": "string or null"
        },

       
        'marksheet': {
            "document_type": "marksheet",
            "student_name": "string — exactly 3 words, e.g. 'Yadav Janvi Ajit'",
            "mother_name": "string  or Null",
            "father_name": "string or null",
            "exam_year": "string or null, e.g. 'March 2024'",
            "subjects": [
                {
                    # marks_obtained: integer for numeric subjects, null for grade-only (P1/P2/P4/R8)
                    "subject_name": "string — subject name only, no code number",
                    "marks_obtained": "integer or null — null when grade field has a letter",
                    "max_marks": "integer 100 or null — null when grade field has a letter",
                    # grade: null for ALL numeric subjects. Only A/B/C/D/E for P1/P2/P4/R8 subjects.
                    # NEVER put IN_WORDS text (NINETYFIVE) here. NEVER put a number here.
                    "grade": "null for numeric subjects — only A or B or C or D or E for activity subjects"
                }
            ],
            "total_marks_obtained": "integer or null — from IN_WORDS spelling after max_marks",
            "total_max_marks": "integer — must be exactly 400 or 500 or 600",
            "percentage": "decimal number with 2 places or null — e.g. 85.60 not 428",
            "result": "PASS or FAIL or DISTINCTION or ATKT or QUALIFIED or null"
        }
    }
    
    return schemas.get(doc_type, {
        "document_type": doc_type,
        "extracted_data": "object"
    })


def create_ssc_extraction_prompt(raw_text: str, schema: dict) -> str:
    """
    Simplified prompt for 3B models. Key changes:
    1. Remove complex code lookup - just parse what's there
    2. Simpler pattern matching rules
    3. More explicit examples
    4. Better formatting cues
    """
    prompt = f"""Extract data from this SSC marksheet OCR text into JSON.

RAW OCR TEXT:
{raw_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION RULES:

1. STUDENT DETAILS 
student_name: text after "CANDIDATE'S FULL NAME" or "ICANDIDATE'S FULL NAME"
mother_name:  text after "CANDIDATE'S MOTHER'S NAME" or "ICANDIDATE'SI MOTHER'S NAME"
father_name:  text after "FATHER'S NAME" — null if absent

NAME RULES:
  student_name → ALWAYS exactly 3 words (Surname + First + Middle/Father).
  mother_name  → ALWAYS exactly 1 word (first clean word only).

NAME STOP RULE: Stop at the first invalid token. Invalid = has digits, symbols, or 3+ consecutive consonants.
  "Pangale Saloni Dilip TAgenreat"  → "Pangale Saloni Dilip"   (TAgenreat: grt = 3 consonants)
  "Jayashree favarun mmafne"        → "Jayashree"              (favarun: vrn = 3 consonants)
  "Nilofar CHI SnTa"                → "Nilofar"                (CHI: all consonants)
  "Reshama FRIA NTOr"               → "Reshama"                (FRIA: all consonants)
  "Rohini Ravura Hsde"              → mother_name = "Rohini"   (Ravura: rvr = 3 consonants → stop)

WORD COUNT ENFORCEMENT:
  student_name: take the first 3 valid words only — no more, no less (if 3 clean words exist).
  mother_name:  take the SINGLE first valid word only — stop immediately after it. ONE WORD ONLY.
  Text after the name is always OCR label noise — discard everything after the name.

  ✗ WRONG: mother_name = "Rohini Ravura Hsde"   ← 3 words, completely wrong
  ✓ RIGHT: mother_name = "Rohini"                ← 1 word, stop immediately

2. SUBJECTS (process in order they appear):
   
   TWO TYPES:
   
   Type A - NUMERIC SUBJECTS (have "100" as max marks):
   Pattern: [code] [name] 100 [3-digit marks] [words]
   Example: "71 MATHEMATICS 100 090 NINETY"
   
   Extract as:
   {{
     "subject_name": "Mathematics",           ← subject name WITHOUT the code
     "marks_obtained": 90,                    ← the 3-digit number (090→90)
     "max_marks": 100,
     "grade": null                            ← ALWAYS null for numeric subjects
   }}
   
   Type B - GRADE-ONLY SUBJECTS (no marks, just a letter grade):
   Pattern: [code] [name] [single letter A/B/C/D/E]
   Example: "P1 HEALTH & PHYSICAL EDUCATION A"
   
   Extract as:
   {{
     "subject_name": "Health & Physical Education",
     "marks_obtained": null,                  ← ALWAYS null
     "max_marks": null,                       ← ALWAYS null  
     "grade": "A"                             ← just the letter
   }}
   
   CRITICAL: For grade-only subjects, STOP at the grade letter.
   Everything after "A" is NOT part of this subject.
   
   Example:
   "R8 WATER SECURITY A Percentage 85.20 Total 500..."
                       ↑
                    STOP HERE
   
   Do NOT extract 85.20, 500, or any numbers after the grade letter.

3. TOTALS (at the end, after all subjects):
   
   Look for a line with these numbers:
   - A decimal with 2 decimal places (XX.XX) → percentage
   - The number 400, 500, or 600 → total_max_marks
   - A 3-digit number less than max_marks → total_marks_obtained
   
   Example line:
   "Percentage 85.20 Total Marks 500 428 FOUR HUNDRED TWENTYEIGHT"
                 ↑              ↑   ↑
           percentage    max_marks  marks_obtained
   
   Extract:
   - percentage: 85.20
   - total_max_marks: 500
   - total_marks_obtained: 428

4. OTHER FIELDS:
   - exam_year: Find "MARCH-2024" or similar
   - result: Find "PASS" or "FAIL"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMMON MISTAKES TO AVOID:

❌ WRONG: Putting word-form marks in "grade" field
   {{"subject_name": "English", "grade": "EIGHTYFOUR"}}
   
✓ CORRECT: Grade is null for numeric subjects
   {{"subject_name": "English", "marks_obtained": 84, "grade": null}}

❌ WRONG: Putting percentage or totals in grade-only subjects
   {{"subject_name": "Water Security", "marks_obtained": 85.2, "grade": "A"}}
   
✓ CORRECT: Only the letter grade, nulls for marks
   {{"subject_name": "Water Security", "marks_obtained": null, "max_marks": null, "grade": "A"}}

❌ WRONG: Swapping total_marks_obtained and total_max_marks
   {{"total_marks_obtained": 500, "total_max_marks": 428}}
   
✓ CORRECT: Max is always bigger (400/500/600)
   {{"total_marks_obtained": 428, "total_max_marks": 500}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCHEMA:
{schema}

OUTPUT FORMAT:
Return ONLY a valid JSON object. Use null for missing fields (not empty strings).
Percentage must have exactly 2 decimal places (85.20 not 85.2).

FINAL CHECKS BEFORE RETURNING:
□ student_name has exactly 3 words?
□ mother_name has exactly 1 word?
□ All numeric subjects have grade=null?
□ All grade-only subjects have marks_obtained=null AND max_marks=null?
□ percentage is a decimal (has a dot)?
□ total_max_marks is 400, 500, or 600?
□ total_marks_obtained is less than total_max_marks?

Return JSON now:"""
    
    return prompt


def create_cbse_extraction_prompt(raw_text: str, schema: dict) -> str:
    prompt = f"""You are an OCR post-processing system for CBSE marksheets. Extract data ONLY from the RAW OCR TEXT below.

  Raw OCR text:
  {raw_text}

  ━━━ 1. STUDENT DETAILS ━━━
student_name: text after "This is to certify that" or before "Roll No"
mother_name: text after "Mother's Name"
father_name: text after "Father's Name" or "Father's/Guardian's Name"
exam_year: extract year from "EXAMINATION, YYYY" or "EXAMINATION YYYY"

━━━ 2. SUBJECTS ━━━
CBSE subjects have this pattern:
  [CODE] [SUBJECT_NAME] [THEORY] [PRACTICAL/IA] [TOTAL] [IN_WORDS] [GRADE]

Example: "184 ENGLISH LNG & LIT. 068 020 088 EIGHTY EIGHT A2"
  subject_name = "English Lng & Lit."
  marks_obtained = 88 (use TOTAL, or convert IN_WORDS if more reliable)
  max_marks = 100 (or 200 for some language subjects)
  grade = "A2" (keep full grade: A1, A2, B1, B2, etc.)

EXTRACTION RULES:
1. Subject name: Clean text between code and marks, remove code number
2. marks_obtained: Use TOTAL column (third number) OR convert IN_WORDS
3. max_marks: Usually 100, but check if 200 is shown
4. grade: Extract full grade (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2)

SPECIAL CASES:
  - "ADDITIONAL SUBJECT": Include if present
  - Strip subject code from subject_name
  - Convert IN_WORDS like "EIGHTY EIGHT" to 88

━━━ 3. TOTALS ━━━
CBSE marksheets typically DO NOT show totals/percentage on the certificate.

percentage = null
total_max_marks = null
total_marks_obtained = null
result: Extract "PASS"/"FAIL"/"COMPARTMENT" if shown, else null

NEVER calculate totals by summing subjects.

━━━ 4. OUTPUT ━━━
Return ONLY valid JSON. Use null for missing fields.

SCHEMA: {schema}

Return ONLY the JSON object."""
    return prompt


def create_icse_extraction_prompt(raw_text: str, schema: dict) -> str:
    prompt = f"""You are an OCR post-processing system for ICSE/ISC marksheets. Extract data ONLY from the RAW OCR TEXT below.

  Raw OCR text:
  {raw_text}

  ━━━ 1. STUDENT DETAILS ━━━
student_name: text after "Name :" or "Name:"
mother_name: text after "Mother" or "Mother:"
father_name: text after "Father:" or "Father's Name"
exam_year: extract year from "YEAR YYYY EXAMINATION"

━━━ 2. SUBJECTS ━━━
ICSE subjects have varied patterns:

Format 1: Main subject with marks
  "ENGLISH 89 EIGHT NINE 2 TWO"
  subject_name = "English"
  marks_obtained = 89 (first number after subject name)
  max_marks = 100
  grade = null

Format 2: Composite subjects
  "HISTORY, CIVICS & GEOGRAPHY 93 NINE THREE 1 ONE"
  Extract ONLY the main composite subject.
  SKIP sub-component lines like "HISTORY & CIVICS 093"

Format 3: Grade-only subjects
  "SUPW AND COMMUNITY SERVICE A"
  OR "Interpersonal Assessment SUPW AND COMMUNITY SERVICE A"
  marks_obtained = null
  max_marks = null
  grade = "A"

EXTRACTION RULES:
1. Subject name: Text before first number, clean it up
2. marks_obtained: First number after subject name OR convert IN_WORDS
3. max_marks: 100 (standard)
4. grade: null for numeric subjects, "A"/"B"/"C" for activity subjects

IMPORTANT: Skip sub-component breakdown lines (they duplicate main subject marks)

━━━ 3. TOTALS ━━━
ICSE marksheets typically DO NOT show totals/percentage.

percentage = null
total_max_marks = null
total_marks_obtained = null
result: Extract "QUALIFIED"/"PASS" if shown, else null

NEVER calculate totals by summing subjects.

━━━ 4. OUTPUT ━━━
Return ONLY valid JSON. Use null for missing fields.

SCHEMA: {schema}

Return ONLY the JSON object."""
    return prompt


def create_marksheet_extraction_prompt(raw_text: str, schema: dict) -> str:
    board = detect_marksheet_board(raw_text)
    if board == 'ssc':
        return create_ssc_extraction_prompt(raw_text, schema)
    elif board == 'cbse':
        return create_cbse_extraction_prompt(raw_text, schema)
    elif board == 'icse':
        return create_icse_extraction_prompt(raw_text, schema)
    else:
        return create_ssc_extraction_prompt(raw_text, schema)


def create_income_certificate_extraction_prompt(raw_text: str, schema: dict) -> str:
    """Create extraction prompt for Indian Income Certificate - Handles Multiple Years"""
    
    truncated_text = raw_text[:4000] if len(raw_text) > 4000 else raw_text

    prompt = f"""
You are extracting data from an INDIAN INCOME CERTIFICATE.

⚠️ CRITICAL WARNING: You are extracting from THIS SPECIFIC document only. DO NOT use values from previous documents, examples, or your training data. If you cannot find a field in the text below, set it to null.

CRITICAL RULES (MUST FOLLOW):
1. Extract ONLY information that is EXPLICITLY present in the OCR text below
2. DO NOT guess, infer, calculate, or translate values
3. For address: Extract ONLY the residential address (village/city name). 
   DO NOT include signature blocks, digital signature text, or office stamps.
   If address contains "Digitally Signed" or office headers, set to null.
4. DO NOT use values from other documents or your memory
5. Preserve the ORIGINAL script (Marathi / Devanagari / English) EXACTLY as written
6. If a field is not found, return null
7. Respond with ONLY valid JSON OBJECT (use curly braces {{}}, NOT square brackets [])
8. NO markdown formatting (no ``` or ```json)
9. NO explanations or notes
10. REMOVE prefixes and suffixes - extract only the CORE value

FIELDS TO EXTRACT:

1. document_type: Always "income_certificate"

2. parent_name:
   - Find pattern: "सदरचा दाखला श्री॰ [NAME] यांना" (if no student mentioned)
   - OR pattern: "सदरचा दाखला श्री॰ [NAME] यांची मुलगी" or "यांचा मुलगा" (if student exists)
   - Extract ONLY the name between "श्री॰" and the next word ("यांना" OR "यांची" OR "यांचा" OR "राहणार")
   - DO NOT include: "श्री॰", "यांना", "यांची", "यांचा" - these are NOT part of the name
   - Example: "श्री॰ परवेद्य अहमद गुलाम मोहम्मद मोमिन यांना" → Extract: "परवेद्य अहमद गुलाम मोहम्मद मोमिन"
   - Return null if not found

3. student_name:
   - Find the line starting with "सदरचा दाखला श्री॰ [PARENT] यांचा मुलगा" or "यांची मुलगी"
   - Look for pattern: "मुलगा कुमार [NAME] यांना" OR "मुलगी कुमारी [NAME] यांना"
   - Extract ONLY the name between "कुमार"/"कुमारी" and "यांना"
   - ⚠️ IMPORTANT: The student name appears AFTER the words "मुलगा कुमार" or "मुलगी कुमारी"
   - Examples:
     * "मुलगा कुमार आर्यन सचिन मोरे यांना" → Extract: "आर्यन सचिन मोरे"
     * "मुलगी कुमारी हुमा युनुस शेख यांना" → Extract: "हुमा युनुस शेख"
     * "मुलगा कुमार मराड सार्थक संदीपान यांना" → Extract: "मराड सार्थक संदीपान"
   - DO NOT include: "कुमारी", "कुमार", "मुलगा", "मुलगी", "यांना"
   - If the certificate is for the parent directly (no मुलगी/मुलगा mentioned), return null
   - Return null if not found

4. income_years: ⚠️ THIS IS NOW AN ARRAY - Can contain 1 or multiple year entries
   
   HOW TO EXTRACT MULTIPLE YEAR ENTRIES:
   
   Step 1: Find the income table section (usually after "वार्षिक उत्पन्न" or "उत्पत्र खालील प्रमाणे")
   
   Step 2: Look for ALL lines matching these patterns:
   - "२०२२ २०२३ [AMOUNT] [WORDS]"
   - "२०२३ २०२४ [AMOUNT] [WORDS]"
   - "२०२४ २०२५ [AMOUNT] [WORDS]"
   - Or single year: "२०२४ [AMOUNT] [WORDS]" (without second year)
   
   Step 3: For EACH line found, create a separate object with these fields:
   {{
     "year": "exact year string from document (e.g., '२०२२ २०२३' or '२०२४ २०२५')",
     "income_value": "numeric value EXACTLY as written (e.g., '१,००,०००' or '१ २०,०००')",
     "income_in_words": "Marathi words EXACTLY as written (e.g., 'एक लाख मात्र' or 'एक लाख वीस हजार मात्र')"
   }}
   
   If NO year entries are found: Return empty array []

5. address:
   - Find the text that appears AFTER the parent's name and the word "राहणार"
   - Extract until you see: "तहसील" OR "येथील" OR "त्यांचे"
   - Return null if not found

6. validity_date:
   - Look for the pattern: "हे प्रमाणपत्र [DATE] पर्यंतच वैध राहील"
   - Extract ONLY the date after "प्रमाणपत्र" and before "पर्यंतच"
   - Return null if not found

REQUIRED OUTPUT FORMAT:
{{
  "document_type": "income_certificate",
  "parent_name": "string or null",
  "student_name": "string or null",
  "income_years": [{{"year": "...", "income_value": "...", "income_in_words": "..."}}],
  "address": "string or null",
  "validity_date": "string or null"
}}

OCR TEXT (USE ONLY THIS TEXT):
--------------------------------
{truncated_text}
--------------------------------

NOW OUTPUT ONLY THE JSON OBJECT:
"""
    return prompt


def create_passbook_extraction_prompt(raw_text: str, schema: dict) -> str:
    """Create extraction prompt for Bank Passbook"""
    
    truncated_text = raw_text[:3000] if len(raw_text) > 3000 else raw_text
    
    if 'S/D/H/O' in truncated_text or 'S/D/H/o' in truncated_text:
        lines = truncated_text.split('\n')
        for i, line in enumerate(lines):
            if 'S/D/H/O' in line:
                lines[i] = f"\n⭐ PARENT NAME LINE: {line}\n"
        truncated_text = '\n'.join(lines)

    prompt = f"""You are extracting data from a BANK PASSBOOK document.

CRITICAL RULES:
1. Extract ONLY information that is EXPLICITLY written in the OCR text below
2. NEVER use examples, your training data, or information from other documents
3. If a field is not found in the text: set it to null
4. Copy text EXACTLY as it appears (preserve spelling, capitalization, spacing)

FIELDS TO EXTRACT:

**account_holder_name**
- Look for: "NAME(S)", "Account Name", "A/C Name", "ACCOUNT NAME", "Customer Name"
- This is the CUSTOMER'S name, NOT the branch name
- ❌ SKIP: "Br. Name", "Branch Name", "Br. Address"
- Extract the FULL name exactly as written
- If not found: null

**account_number** - CRITICAL FIELD
- MUST be preceded by one of these EXACT labels: "Account No:", "Account No.", "A/C No:", "A/c No:", "Account Number:", "A/C Number:"
- The label MUST appear immediately before or on the same line as the number
- Extract ONLY the digits that appear AFTER these labels
- Usually 9-18 digits long
- Preserve leading zeros
- ❌ STRICT RULES:
  * DO NOT extract numbers that appear BEFORE "Account No:" label
  * DO NOT use: CIF Number, Customer ID, MICR Code, IFSC Code
  * DO NOT use any number that doesn't have the "Account No:" label
  * If you see "CIF Number: 12345" followed by "Account No.: 67890", use ONLY 67890)
- If not found: null

**parent_name**
- Look for: "S/D/H/O", "S/D/W/H/O", "S/D/W/H/o", "S/D/H/o", "Nominee", "Joint Name", "Joint Name 1"
- This appears in the CUSTOMER section, not branch section
- The field may appear as "S/D/H/O :" or "S/D/H/O:" (with or without space before colon)
- IMPORTANT: The name may appear on the SAME line after the colon OR on the NEXT line below it
- Extract the full name that appears immediately after the colon (same line) or on the next line
- Stop extraction at: pipe (|), newline followed by another field name, or next structured field
- Remove leading/trailing whitespace from the extracted name
- The name should typically be in UPPERCASE letters
- If multiple variations exist, prioritize in this order: S/D/H/O > S/D/W/H/O > Nominee > Joint Name
- If not found: null

**address**
- This is the CUSTOMER's address, NOT the branch address
- Usually appears after "Br. Address" (branch address comes first)
- Look for customer-specific locations (house numbers, localities)
- ❌ SKIP: Any address starting with "Br. Address", "Branch Address"
- If the only address is clearly a branch (has "TAL", "DIST", branch codes): set to null
- If not found: null

RESPONSE FORMAT:
- Return ONLY a JSON object
- No markdown (no ``` or ```json)
- No explanations or notes
- Match these exact field names: {list(schema.keys())}

OCR TEXT TO EXTRACT FROM:
{truncated_text}

NOW EXTRACT - USE ONLY THE OCR TEXT ABOVE:"""
    
    return prompt


def create_aadhaar_extraction_prompt(raw_text: str, schema: dict) -> str:
    """Create extraction prompt for Aadhaar Card"""
    
    truncated_text = raw_text[:3000] if len(raw_text) > 3000 else raw_text
    
    prompt = f"""Extract information from this AADHAAR document.

RULES:
1. Extract ONLY data EXPLICITLY present in the OCR text below
2. Copy text exactly as written (don't fix spelling or formatting)
3. If a field is not found: set to null
4. Never calculate or infer values

Expected fields: {list(schema.keys())}

OCR TEXT:
{truncated_text}

Extract to JSON (no markdown, no explanations):"""
    
    return prompt


def create_pan_extraction_prompt(raw_text: str, schema: dict) -> str:
    """Create extraction prompt for PAN Card"""
    
    truncated_text = raw_text[:3000] if len(raw_text) > 3000 else raw_text
    
    prompt = f"""Extract information from this PAN document.

RULES:
1. Extract ONLY data EXPLICITLY present in the OCR text below
2. Copy text exactly as written (don't fix spelling or formatting)
3. If a field is not found: set to null
4. Never calculate or infer values

Expected fields: {list(schema.keys())}

OCR TEXT:
{truncated_text}

Extract to JSON (no markdown, no explanations):"""
    
    return prompt


def create_generic_extraction_prompt(raw_text: str, doc_type: str, schema: dict) -> str:
    """Create generic extraction prompt for unknown document types"""
    
    truncated_text = raw_text[:3000] if len(raw_text) > 3000 else raw_text
    
    prompt = f"""Extract information from this {doc_type.upper()} document.

RULES:
1. Extract ONLY data EXPLICITLY present in the OCR text below
2. Copy text exactly as written (don't fix spelling or formatting)
3. If a field is not found: set to null
4. Never calculate or infer values

Expected fields: {list(schema.keys())}

OCR TEXT:
{truncated_text}

Extract to JSON (no markdown, no explanations):"""
    
    return prompt


def get_system_prompt() -> str:
    """Get the system prompt for the extraction model"""
    return "You are a precise data extraction assistant. Extract ONLY information that is EXPLICITLY present in the provided OCR text. NEVER use examples, training data, or invented information. Respond ONLY with valid JSON - no markdown, no explanations."


def create_extraction_prompt(raw_text: str, doc_type: str, schema: dict = None) -> str:
    """
    Main function to create extraction prompt based on document type
    """
    if schema is None:
        schema = get_schema_for_doc_type(doc_type)
    
    prompt_functions = {
        'pass_book': create_passbook_extraction_prompt,
        'passbook': create_passbook_extraction_prompt,
        'income_certificate': create_income_certificate_extraction_prompt,
        'aadhaar': create_aadhaar_extraction_prompt,
        'pan': create_pan_extraction_prompt,
        'marksheet': create_marksheet_extraction_prompt
    }
    
    prompt_func = prompt_functions.get(doc_type, create_generic_extraction_prompt)
    
    if prompt_func == create_generic_extraction_prompt:
        return prompt_func(raw_text, doc_type, schema)
    else:
        return prompt_func(raw_text, schema)