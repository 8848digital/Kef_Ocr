"""
Document Extraction Prompts
Contains all prompts for extracting structured data from OCR text
"""


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
            "income_years": "array of objects",  # Changed to array
            "address": "string or null",
            "validity_date": "string or null"
        },
        'marksheet':{
            "document_type":"marksheet",
            "student_name":"string or null",
            "mother_name":"string or null",
            "fathers_name":"string or null",
            "exam_year": "string or null",
            "subjects":[
                {
                "subject_name": "string",
                "marks_obtained": "number or null",
                "max_marks": "number or null",
                "grade": "string or null"

                }

            ],
            "total_marks_obtained": "number or null",
            "total_max_marks": "number or null",
            "percentage": "number or null",
            "result": "string or null"

            

            
            
        }
        
    }
    
    return schemas.get(doc_type, {
        "document_type": doc_type,
        "extracted_data": "object"
    })

def create_marksheet_extraction_prompt(raw_text: str, schema: dict) -> str:
    prompt = f"""You are an expert OCR post-processing and document extraction system specializing in Indian educational marksheets.

🔴 CRITICAL: Extract data ONLY from the RAW OCR TEXT provided below. Do NOT use any information from previous examples, previous extractions, or your memory. Each extraction must be completely independent.

The text below was extracted via OCR from a scanned marksheet. It may contain:
- Mixed Devanagari (Marathi/Hindi) and English text
- OCR noise and garbled characters (e.g., "HTEZITHCh", "fATETUT", "THTUTYEL")
- Numbers written as English words (e.g., "NINETYFIVE" = 95, "EIGHTYSEVEN" = 87)
- Inconsistent spacing, merged words, or broken words

RAW OCR TEXT:
{raw_text}

YOUR TASK:
Extract all meaningful information from this marksheet and return a single valid JSON object.

EXTRACTION RULES:

1. STUDENT DETAILS
   - student_name: Full name, typically in "SURNAME FIRSTNAME FATHERNAME" format
   - mother_name: Look for "MOTHER'S NAME", "आईचे नाव", "CANDIDATE'S MOTHER'S NAME"
   - father_name: Look for "FATHER'S NAME", "वडिलांचे नाव" — return null if not found

   RULES:
   - Names appear immediately after these exact labels:
     student_name: after "CANDIDATE'S FULL NAME (SURNAME FIRST)" or "ICANDIDATE'S FULL NAME"
     mother_name: after "CANDIDATE'S MOTHER'S NAME" or "ICANDIDATE'SI MOTHER'S NAME"
   - Extract ONLY the clean name — stop at the FIRST line of gibberish/Devanagari
   - A valid name contains ONLY: letters, spaces, hyphens, apostrophes
   - Any token containing numbers, symbols (@, #, $), or 4+ consecutive consonants
     is gibberish — STOP extraction there
   - Examples:
     "Pangale Saloni Dilip TAgenreat amsa aa" → "Pangale Saloni Dilip"  (stop at TAgenreat)
     "Yadav Janvi Ajit 34caR/ea 31S RIa"     → "Yadav Janvi Ajit"      (stop at 34caR)
     "Khedekar Darshak Nathuram THeeRTent"   → "Khedekar Darshak Nathuram" (stop at THeeRTent)
     "Rohini Ravura Hsde"                    → "Rohini"                 (Ravura/Hsde are garbage)
   - Indian names are typically 2-3 words: SURNAME FIRSTNAME FATHER/HUSBAND NAME
   - If uncertain, keep only the first 2-3 clean words
   

2. SUBJECTS AND MARKS
   - Extract EVERY subject listed on the marksheet, including both numeric and grade-only subjects
   - Subject lines follow pattern: [CODE] [SUBJECT NAME] [MAX] [OBTAINED] [IN WORDS]
   
   🔴 CRITICAL RULE: Extract subjects IN THE ORDER they appear in the raw text
   - Match the EXACT subject name from the raw text (clean up OCR noise, but don't substitute different subjects)
   - Match the EXACT marks shown (don't use marks from other subjects or other documents)
   
   FOR NUMERIC SUBJECTS (subjects with marks):
   - Look for lines containing both a subject name AND numeric marks
   - Common patterns:
     "03 ENGLISH (1ST LANG) 100 084 EIGHTYFOUR"
     "16 MARATHI (2/3 LANG) 100 058 FIFTYEIGHT"
     "151 HINDI (2/3 LANG) 100 076 SEVENTYSIX"
     "71 MATHEMATICS 100 090 NINETY"
   
   - Subject code (01, 03, 16, 27, 71, 72, 73, 151, etc.) helps identify the subject but don't include it in subject_name
   - Language subjects may be marked as (1ST LANG) or (2/3 LANG) - include this in parentheses
   - Convert word-form marks to numbers:
     NINETYNINE=99, NINETYEIGHT=98, NINETYSEVEN=97, NINETYSIX=96, NINETYFIVE=95, 
     NINETYFOUR=94, NINETYTHREE=93, NINETYTWO=92, NINETYONE=91, NINETY=90,
     EIGHTYNINE=89, EIGHTYEIGHT=88, EIGHTYSEVEN=87, EIGHTYSIX=86, EIGHTYFIVE=85,
     EIGHTYFOUR=84, EIGHTYTHREE=83, EIGHTYTWO=82, EIGHTYONE=81, EIGHTY=80,
     SEVENTYNINE=79, SEVENTYEIGHT=78, SEVENTYSEVEN=77, SEVENTYSIX=76, SEVENTYFIVE=75,
     SEVENTYFOUR=74, SEVENTYTHREE=73, SEVENTYTWO=72, SEVENTYONE=71, SEVENTY=70,
     SIXTYNINE=69, SIXTYEIGHT=68, SIXTYSEVEN=67, SIXTYSIX=66, SIXTYFIVE=65,
     SIXTYFOUR=64, SIXTYTHREE=63, SIXTYTWO=62, SIXTYONE=61, SIXTY=60,
     FIFTYNINE=59, FIFTYEIGHT=58, FIFTYSEVEN=57, FIFTYSIX=56, FIFTYFIVE=55,
     FIFTYFOUR=54, FIFTYTHREE=53, FIFTYTWO=52, FIFTYONE=51, FIFTY=50,
     FORTYNINE=49, FORTYEIGHT=48, FORTYSEVEN=47, FORTYSIX=46, FORTYFIVE=45,
     FORTYFOUR=44, FORTYTHREE=43, FORTYTWO=42, FORTYONE=41, FORTY=40,
     THIRTYNINE=39, THIRTYEIGHT=38, THIRTYSEVEN=37, THIRTYSIX=36, THIRTYFIVE=35,
     And so on for THIRTY (30), TWENTY (20), etc.
   - max_marks is almost always 100 per subject for SSC/HSC
   - Set marks_obtained to the numeric value FROM THE RAW TEXT (not from your memory or other documents)
   - Set max_marks to 100 (or the stated maximum)
   - Set grade to null
   
   ⚠️ COMMON ERROR TO AVOID:
   - If raw text shows "16 MARATHI (2/3 LANG) 100 058 FIFTYEIGHT", extract Marathi: 58
   - DO NOT extract "Sanskrit: 58" - there is no Sanskrit in this example
   - DO NOT mix up subject names with marks from other subjects
   
   FOR GRADE-ONLY SUBJECTS (co-curricular subjects with only letter grades):
   - Common grade-only subjects:
     * P1: Health & Physical Education
     * P2: Scouting/Guiding
     * P41: Defence Studies
     * R8: Water Security
     * Art Education
     * Work Experience
   - Pattern to identify: [CODE] [SUBJECT NAME] [*] [GRADE_LETTER]
     Where * indicates "not applicable" or no max marks shown
   - Set marks_obtained to null
   - Set max_marks to null
   - Set grade to the letter grade (A/B/C/D/E)
   - These subjects do NOT contribute to total_marks_obtained or percentage
   
   Example extractions:
   "03 ENGLISH (1ST LANG) 100 084 EIGHTYFOUR"
   → {{"subject_name": "English (1st Lang)", "marks_obtained": 84, "max_marks": 100, "grade": null}}
   
   "16 MARATHI (2/3 LANG) 100 058 FIFTYEIGHT"
   → {{"subject_name": "Marathi (2/3 Lang)", "marks_obtained": 58, "max_marks": 100, "grade": null}}
   
   "P1 HEALTH & PHYSICAL EDUCATION    A"
   → {{"subject_name": "Health & Physical Education", "marks_obtained": null, "max_marks": null, "grade": "A"}}
   
   "P41 DEFENCE STUDIES                A"
   → {{"subject_name": "Defence Studies", "marks_obtained": null, "max_marks": null, "grade": "A"}}
   
   Grade meanings (for reference, don't include in output):
   A = Excellent, B = Very Good, C = Good, D = Average, E = Below Average

3. TOTALS, PERCENTAGE AND RESULT — CRITICAL POSITIONAL EXTRACTION
   
   ⚠️ CRITICAL: The individual subject marks MAY NOT sum to total_marks_obtained
   This is normal because:
   - Marks may be weighted or scaled
   - Only certain subjects may count toward the total
   - There may be additional marks from practical/project categories
   - NEVER calculate totals from subject marks
   - ONLY extract what is explicitly stated in the document
   
   These three values always appear together in a fixed order near "Percentage" / "Total Marks".
   
   PATTERN: [PERCENTAGE] [NOISE] [MAX_MARKS] [OBTAINED_MARKS_NOISY] [IN WORDS] [NOISE] Result [RESULT]
   
   Real OCR examples — study these carefully:
   "ZaRaTi 93.60 Vepuce Tut/ 500 $465+03 FOUR HUNDRED AND SIXTYEIGHT ... Result PASS"
   "Zarharit/ 85.20 Leput TUT/ 500 426 FOUR HUNDRED AND TWENTYSIX ... Result PASS"
   "85.60 LUT JUTI 500 428 FOUR HUNDRED AND TWENTYEIGHT ... Result PASS"
   "टक्केवारी/Percentage 91.40 एकूण गुण/Total Marks 500 457 FOUR HUNDRED AND FIFTYSEVEN Result PASS"
   
   STEP-BY-STEP EXTRACTION (follow this exact order):
   
   ═══════════════════════════════════════════════════════════════════════
   STEP 1 → percentage
   ═══════════════════════════════════════════════════════════════════════
   Find the FIRST decimal number with exactly 2 decimal places (e.g., 93.60, 85.20, 91.40)
   This is ALWAYS the percentage. 
   Pattern: XX.XX where X is a digit
   Common locations: near "टक्केवारी", "Percentage", "ZaRaTi"
   Ignore all other numbers until you find this pattern.
   
   ═══════════════════════════════════════════════════════════════════════
   STEP 2 → total_max_marks
   ═══════════════════════════════════════════════════════════════════════
   After the percentage, find the FIRST clean round number (400, 500, or 600)
   This is ALWAYS total_max_marks.
   Common values:
   - 500 for 5-subject boards or scaled totals
   - 600 for 6-subject boards
   - 400 for 4-subject boards (rare)
   Look near: "Total Marks", "एकूण गुण", "Tut/", "TUT/"
   
   ═══════════════════════════════════════════════════════════════════════
   STEP 3 → total_marks_obtained (USE IN WORDS AS PRIMARY SOURCE)
   ═══════════════════════════════════════════════════════════════════════
   The obtained marks appear in TWO forms:
   a) Numeric form (OFTEN CORRUPTED by OCR): "$465+03", "546+03", "$426", "4Z6", etc.
   b) IN WORDS form (MORE RELIABLE): "FOUR HUNDRED AND SIXTYEIGHT"
   
   🔴 PRIMARY METHOD: ALWAYS use IN WORDS first
   
   Look for pattern: "[HUNDREDS] HUNDRED [AND] [TENS][UNITS]"
   
   The IN WORDS appears immediately after the noisy numeric form.
   
   Examples from real marksheets:
   "$465+03 FOUR HUNDRED AND SIXTYEIGHT"     → extract 468
   "426 FOUR HUNDRED AND TWENTYSIX"          → extract 426  
   "$4Z8 FOUR HUNDRED AND TWENTYEIGHT"       → extract 428
   "504 FIVE HUNDRED AND FOUR"               → extract 504
   "457 FOUR HUNDRED AND FIFTYSEVEN"         → extract 457
   
   WORD-TO-NUMBER CONVERSION TABLE:
   
   Hundreds:
   - ONE HUNDRED = 100
   - TWO HUNDRED = 200
   - THREE HUNDRED = 300
   - FOUR HUNDRED = 400
   - FIVE HUNDRED = 500
   - SIX HUNDRED = 600
   
   Tens:
   - TWENTY = 20, THIRTY = 30, FORTY = 40, FIFTY = 50
   - SIXTY = 60, SEVENTY = 70, EIGHTY = 80, NINETY = 90
   
   Units (1-9):
   - ONE=1, TWO=2, THREE=3, FOUR=4, FIVE=5
   - SIX=6, SEVEN=7, EIGHT=8, NINE=9
   
   Compound parsing rules:
   - "FOUR HUNDRED AND SIXTYEIGHT" = 400 + 68 = 468
   - "FIVE HUNDRED AND FOUR" = 500 + 4 = 504
   - "FOUR HUNDRED AND TWENTYSIX" = 400 + 26 = 426
   - "FOUR HUNDRED SIXTY EIGHT" (no AND) = 400 + 60 + 8 = 468
   
   How to parse compound tens:
   - "SIXTYEIGHT" = 60 + 8 = 68
   - "TWENTYSIX" = 20 + 6 = 26  
   - "NINETYFIVE" = 90 + 5 = 95
   - "FORTYTWO" = 40 + 2 = 42
   
   🟡 FALLBACK METHOD (only if IN WORDS is completely missing/unreadable):
   
   Try to parse the numeric form:
   1. Look for the number immediately after total_max_marks
   2. Clean it: remove all letters and symbols except digits and '+'
   3. If contains '+': split and sum (e.g., "465+03" → 465 + 3 = 468)
   4. If just corrupted digits: try to extract clean number
      "$426" → 426
      "4Z8" → 428 (Z looks like 2)
      "5O4" → 504 (O looks like 0)
   
   ⚠️ WARNING: Use fallback method ONLY when IN WORDS is unavailable
   
   ═══════════════════════════════════════════════════════════════════════
   STEP 4 → result
   ═══════════════════════════════════════════════════════════════════════
   Look for keywords anywhere in the totals section:
   - PASS / पास
   - FAIL / अनुत्तीर्ण  
   - DISTINCTION / विशेष प्रवीणता
   - ATKT (Allowed To Keep Terms - for college marksheets)
   
   Common OCR corruptions:
   "frentrace Result PASS" → PASS
   "Tei/Result PASS" → PASS  
   "Foll/Result PASS" → PASS
   "निकाल/Result PASS" → PASS
   
   Extract the actual result word (PASS/FAIL/DISTINCTION), ignore the noise.
   
   ═══════════════════════════════════════════════════════════════════════
   STEP 5 → Verify the percentage (optional sanity check)
   ═══════════════════════════════════════════════════════════════════════
   After extraction, you can verify:
   calculated_percentage = (total_marks_obtained / total_max_marks) × 100
   
   If your extracted percentage matches calculated_percentage (within ±0.1), 
   you've likely extracted correctly.
   
   Example: 468/500 × 100 = 93.6% ✓ matches extracted 93.60
   
   If there's a mismatch > 0.5%, recheck your extraction.
   However, DO NOT override extracted percentage with calculated value.
   Always use the percentage explicitly printed on the marksheet.

4. ADDITIONAL MARKS / SUPPLEMENTARY SUBJECTS
   
   Some marksheets show additional marks from optional subjects:
   - "Additional Marks Category: DRAWING" 
   - These marks are often shown as "+XX" in the total (e.g., "$465+03")
   - The "+03" means 3 additional marks from Drawing
   
   Do NOT create a separate field for this.
   The total_marks_obtained should include these additional marks.
   
   Example: If shown as "465+03 FOUR HUNDRED AND SIXTYEIGHT"
   → total_marks_obtained = 468 (not 465)

5. OCR NOISE HANDLING
   - Ignore lines that are clearly garbled Devanagari OCR artifacts
   - If a field appears twice (once in Marathi, once in English), use the English value
   - If marks in figures and marks in words conflict, trust the IN WORDS (not figures)
   - If student name appears garbled, use the cleaner/longer version
   - Common OCR confusions:
     * 0 ↔ O (letter O vs zero)
     * 1 ↔ I ↔ l (one vs letter I vs lowercase L)
     * 5 ↔ S
     * 8 ↔ B
     * 2 ↔ Z
     Use context and IN WORDS to resolve these

EXPECTED OUTPUT SCHEMA:
{schema}

OUTPUT RULES:
- Return ONLY a valid JSON object — no explanation, no markdown, no code blocks
- Use null (not "null", not "N/A", not "") for missing fields
- All numeric marks must be integers
- Percentage must be a float with EXACTLY 2 decimal places (e.g., 93.60, not 93.6)
  * Preserve trailing zeros: 85.60 not 85.6, 90.00 not 90, 93.60 not 93.6
  * This matches the format printed on official marksheets
- subject_name should be clean English (e.g., "Mathematics", not "71 MATHEMATICS")
- "subjects" array contains ALL subjects (both numeric and grade-only)
- For numeric subjects: marks_obtained = integer, max_marks = integer, grade = null
- For grade-only subjects: marks_obtained = null, max_marks = null, grade = "A"/"B"/"C"/"D"/"E"
- Do not invent or assume data that is not present in the text
- Do not calculate totals from subject marks — only extract explicit totals
- Ensure total_marks_obtained and percentage are consistent with each other

🔴 BEFORE RETURNING YOUR JSON:
1. Verify each subject name and marks appear in the RAW OCR TEXT above
2. Verify the percentage value appears in the RAW OCR TEXT above
3. Verify the total_marks_obtained matches the "IN WORDS" value in the RAW OCR TEXT
4. If any value doesn't match the RAW OCR TEXT, you have made an error - re-extract from the text
5. Do NOT use data from the example output - that is for format reference only

⚠️ WARNING: The example below uses fictional data. Your output must contain ONLY data from the RAW OCR TEXT at the top of this prompt.

EXAMPLE OUTPUT (for reference only — extract actual values from the text above):
{{
    "document_type": "marksheet",
    "exam_year": "March 2024",
    "centre_no": "8373",
    "student_name": "Pangale Saloni Dilip",
    "mother_name": "Jayashree",
    "father_name": null,
    "subjects": [
        {{"subject_name": "Marathi", "marks_obtained": 82, "max_marks": 100, "grade": null}},
        {{"subject_name": "Sanskrit", "marks_obtained": 98, "max_marks": 100, "grade": null}},
        {{"subject_name": "English", "marks_obtained": 91, "max_marks": 100, "grade": null}},
        {{"subject_name": "Mathematics", "marks_obtained": 86, "max_marks": 100, "grade": null}},
        {{"subject_name": "Science & Technology", "marks_obtained": 95, "max_marks": 100, "grade": null}},
        {{"subject_name": "Social Sciences", "marks_obtained": 95, "max_marks": 100, "grade": null}},
        {{"subject_name": "Health & Physical Education", "marks_obtained": null, "max_marks": null, "grade": "A"}},
        {{"subject_name": "Scouting/Guiding", "marks_obtained": null, "max_marks": null, "grade": "A"}},
        {{"subject_name": "Water Security", "marks_obtained": null, "max_marks": null, "grade": "A"}}
    ],
    "total_marks_obtained": 468,
    "total_max_marks": 500,
    "percentage": 93.60,
    "result": "PASS"
}}

Now extract from the RAW OCR TEXT above and return ONLY the JSON object."""

    return prompt


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
   
   ⚠️ CRITICAL EXTRACTION RULES FOR EACH YEAR ENTRY:
   
   - **year**: Extract the EXACT fiscal year format as it appears
     * Examples: "२०२२ २०२३", "२०२३ २०२४", "२०२४ २०२५"
     * Include the space between years if present
     * Copy EXACTLY - don't change Devanagari to English or vice versa
   
   - **income_value**: Extract ONLY the numeric value from that specific line
     * Copy EXACTLY as written with original punctuation
     * Examples: "१,००,०००", "१ १० ०००", "१ २०,०००"
     * ⚠️ Common mistake: Don't use income from a different year's line
     * If the line shows "२०२२ २०२३ १,००,०००", extract "१,००,०००" NOT any other number
   
   - **income_in_words**: Extract the Marathi words from the SAME EXACT line
     * These words appear AFTER the numeric value on the same line
     * Examples: "एक लाख मात्र", "एक लाख दहा हजार मात्र", "एक लाख वीस हजार मात्र"
     * ⚠️ Each year will have DIFFERENT words - match them correctly to the year
     * Stop at: newline, punctuation, or next sentence
   
   Step 4: Return as JSON array with ALL entries found
   
   EXAMPLE INPUT TEXT:
```
   २०२२ २०२३ १,००,००० एक लाख मात्र
   २०२३ २०२४ १ १० ००० एक लाख दहा हजार मात्र
   २०२४ २०२५ १ २०,००० एक लाख वीस हजार मात्र
```
   
   CORRECT OUTPUT:
```json
   "income_years": [
     {{
       "year": "२०२२ २०२३",
       "income_value": "१,००,०००",
       "income_in_words": "एक लाख मात्र"
     }},
     {{
       "year": "२०२३ २०२४",
       "income_value": "१ १० ०००",
       "income_in_words": "एक लाख दहा हजार मात्र"
     }},
     {{
       "year": "२०२४ २०२५",
       "income_value": "१ २०,०००",
       "income_in_words": "एक लाख वीस हजार मात्र"
     }}
   ]
```
   
   EXAMPLE FOR SINGLE YEAR:
```
   २०२४ २०२५ ८०,००० ऐंशी हजार मात्र
```
   
   CORRECT OUTPUT:
```json
   "income_years": [
     {{
       "year": "२०२४ २०२५",
       "income_value": "८०,०००",
       "income_in_words": "ऐंशी हजार मात्र"
     }}
   ]
```
   
   ⚠️ COMMON MISTAKES TO AVOID:
   - ❌ Mixing income values between different years
   - ❌ Using words from one year with the income of another year
   - ❌ Changing the original script or number format
   - ❌ Returning empty array when years exist in the text
   - ❌ Forgetting to wrap single entry in array brackets []
   
   If NO year entries are found: Return empty array []

5. address:
   - Find the text that appears AFTER the parent's name and the word "राहणार"
   - Extract until you see: "तहसील" OR "येथील" OR "त्यांचे"
   - Include: building names, room numbers, road names, area names
   - Preserve original script exactly as written
   - Return null if not found

6. validity_date:
   - Look for the pattern: "हे प्रमाणपत्र [DATE] पर्यंतच वैध राहील"
   - Extract ONLY the date after "प्रमाणपत्र" and before "पर्यंतच"
   - Common format: "३१ मार्च २०२६" or "31 March 2026"
   - Copy EXACTLY as written in the document
   - Example: "हे प्रमाणपत्र ३१ मार्च २०२६ पर्यंतच वैध राहील" → Extract: "३१ मार्च २०२६"
   - Return null if not found

⚠️ VERIFICATION CHECKLIST BEFORE RESPONDING:
- Did I find ALL year entries in the income table?
- Did I create a separate object for EACH year found?
- Did I copy each year's income_value EXACTLY from its specific line?
- Did I copy each year's income_in_words EXACTLY from the SAME line as its income_value?
- Did I avoid mixing values between different years?
- Did I wrap the results in square brackets [] to make it an array?
- Did I check that these values actually appear in the text below?

REQUIRED OUTPUT FORMAT - JSON OBJECT WITH THESE EXACT KEYS:
{{
  "document_type": "income_certificate",
  "parent_name": "string or null",
  "student_name": "string or null",
  "income_years": [
    {{
      "year": "string (fiscal year)",
      "income_value": "string (numeric value)",
      "income_in_words": "string (Marathi words)"
    }}
  ],
  "address": "string or null",
  "validity_date": "string or null"
}}

OCR TEXT (USE ONLY THIS TEXT):
--------------------------------
{truncated_text}
--------------------------------

NOW OUTPUT ONLY THE JSON OBJECT (curly braces for main object, square brackets for income_years array):
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
    
    Args:
        raw_text: OCR extracted text
        doc_type: Type of document (pass_book, income_certificate, aadhaar, pan, etc.)
        schema: JSON schema for the document (optional, will be fetched if not provided)
    
    Returns:
        str: Formatted prompt for the extraction model
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
    
    # Check if we're using the generic prompt (which needs doc_type as an argument)
    if prompt_func == create_generic_extraction_prompt:
        return prompt_func(raw_text, doc_type, schema)  # Generic needs 3 args
    else:
        return prompt_func(raw_text, schema)  # Specific prompts need 2 args