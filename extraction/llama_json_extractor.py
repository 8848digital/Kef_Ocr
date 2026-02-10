import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

from extraction.prompts import (
    get_schema_for_doc_type,
    create_extraction_prompt,
    get_system_prompt
)


class LlamaJSONExtractor:
    """Extract structured JSON from OCR text using Llama 3.1B"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        print(f" Loading Llama model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
            print(f" GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            dtype = torch.float32
            print("  No GPU detected, using CPU (this will be slower)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        if torch.cuda.is_available():
            print(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        else:
            print("✓ Model loaded on CPU")
    
    def extract_json(self, raw_text: str, doc_type: str, temperature=0.1, max_tokens=1536):
        """
        Convert raw OCR text to structured JSON
        
        CRITICAL: Uses temperature=0.1 (was 0.2) for more deterministic output
        """
        schema = get_schema_for_doc_type(doc_type)
        prompt = create_extraction_prompt(raw_text, doc_type, schema)

        if 'S/D/H/O' in raw_text or 'S/D/H/o' in raw_text:
            print(" DEBUG: Found S/D/H/O pattern in raw text")
            lines = raw_text.split('\n')
            for i, line in enumerate(lines):
                if 'S/D/H/O' in line or 'S/D/H/o' in line:
                    print(f"    Line {i}: {line}")
                    if i + 1 < len(lines):
                        print(f"    Next line: {lines[i+1]}")
        
        messages = [
            {
                "role": "system",
                "content": get_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,  # Lower temperature = more deterministic
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        print(f" DEBUG: LLM Response: {response[:500]}")
        
        return self._parse_json_response(response, doc_type, raw_text)
    
    def _validate_and_fix_extraction(self, data: dict, raw_text: str) -> dict:
        """Validate and fix extracted data"""
        
        # Clean up income certificate data
        if data.get('document_type') == 'income_certificate':
            data = self._cleanup_income_certificate(data, raw_text)
        
        if data.get('document_type') not in ['pass_book', 'passbook']:
            return data

        # Validate parent_name
        if 'parent_name' in data and data['parent_name']:
            parent_name_value = str(data['parent_name']).strip()

            print(f" Validating parent_name: '{parent_name_value}'")
            print(f" parent_name repr: {repr(parent_name_value)}")
            print(f" parent_name length: {len(parent_name_value)}")

            if not parent_name_value:  # Empty after strip
                data['parent_name'] = None
                return data
            
            parent_name_upper = parent_name_value.upper()
        
            # Check for location keywords
            location_keywords = ['NAGAR', 'COLONY', 'STREET', 'ROAD', 'MARG', 
                                'CHAWL', 'TOWNSHIP', 'AREA', 'SOCIETY', 'COMPLEX',
                                'FLOOR', 'BUILDING', 'APARTMENT', 'CHS', 'COMPOUND']
        
            has_location = any(keyword in parent_name_upper for keyword in location_keywords)

            if has_location:
                matched_keywords = [kw for kw in location_keywords if kw in parent_name_upper]
                print(f"     Found location keywords: {matched_keywords}")

            # Check if parent_name appears in address (only if address is not empty)
            address = data.get('address', '')
            in_address = False
            if address and len(parent_name_value) > 10:
                in_address = parent_name_upper in address.upper()
                if in_address:
                    print(f"     Found in address: '{address}'")                
        
            special_char_count = len(re.findall(r'[^A-Za-z\s]', parent_name_value))
            special_char_ratio = special_char_count / max(len(parent_name_value), 1)
        
            is_too_short = len(parent_name_value) < 4
            has_too_many_special_chars = special_char_ratio > 0.3
            has_garbage = bool(re.search(r'(\d{5,}|[^A-Za-z\s]{3,})', parent_name_value))
        
            is_corrupted = is_too_short or has_too_many_special_chars or has_garbage
            if is_corrupted:
                print(f"     Corruption check failed:")
                print(f"       - Too short ({len(parent_name_value)} chars): {is_too_short}")
                print(f"       - Too many special chars ({special_char_ratio:.2%}): {has_too_many_special_chars}")
                print(f"       - Has garbage pattern: {has_garbage}")

            if has_location or in_address or is_corrupted:
                print(f"  VALIDATION: Rejecting parent_name='{data['parent_name']}'")
                if has_location:
                    print(f"    Reason: Contains location keyword")
                if in_address:
                    print(f"    Reason: Found in address field")
                if is_corrupted:
                    print(f"    Reason: Appears corrupted (short={is_too_short}, special_chars={has_too_many_special_chars}, garbage={has_garbage})")
                data['parent_name'] = None
    
        return data
    
    def _cleanup_income_certificate(self, data: dict, raw_text: str = "") -> dict:
        """Clean up common prefixes/suffixes in income certificate fields"""
        
        print(f" _cleanup_income_certificate called")
        print(f"   - Has raw_text: {bool(raw_text)} (length: {len(raw_text) if raw_text else 0})")
        print(f"   - Current student_name: {repr(data.get('student_name'))}")
        
        # Clean parent_name
        if data.get('parent_name'):
            name = str(data['parent_name'])
            # Remove prefixes
            name = re.sub(r'^(श्री॰\s*|श्री\s*)', '', name)
            # Remove suffixes
            name = re.sub(r'\s*(यांना|यांची|यांचा|राहणार)$', '', name)
            data['parent_name'] = name.strip() if name.strip() else None
            
        # Clean student_name
        if data.get('student_name'):
            name = str(data['student_name'])
            # Remove prefixes
            name = re.sub(r'^(कुमारी\s*|कुमार\s*)', '', name)
            # Remove suffixes
            name = re.sub(r'\s*(यांना|याना)$', '', name)
            data['student_name'] = name.strip() if name.strip() else None
        
        # If student_name is null but text mentions मुलगा/मुलगी, try to extract it
        if raw_text and not data.get('student_name'):
            # Check for student indicators
            indicators = ['मुलगा कुमार', 'मुलगी कुमारी', 'मुलगा', 'मुलगी', 'यांचा मुलगा', 'यांची मुलगी']
            found_indicators = [ind for ind in indicators if ind in raw_text]
            has_student_indicator = len(found_indicators) > 0
            
            if has_student_indicator:
                print(f"  WARNING: Document mentions student but student_name is null")
                print(f"    Found indicators: {found_indicators}")
                print(f"    Attempting to extract student name...")
                
                # Show relevant portion of text for debugging (increased window)
                student_section = re.search(r'सदरचा दाखला[^\n]{0,300}', raw_text)
                if student_section:
                    print(f"    Text section: {student_section.group(0)}")
                else:
                    # Try alternate search
                    student_section = re.search(r'मुलग[ाी][^\n]{0,250}', raw_text)
                    if student_section:
                        print(f"    Text section (alt): {student_section.group(0)}")
                
                # Try to extract student name using multiple regex patterns
                # Order matters - try most specific patterns first
                # IMPROVED: Allow for name to extend across line breaks and be more flexible with ending
                patterns = [
                    # Pattern 1: "यांचा मुलगा कुमार [NAME] यांना" - capture everything until यांना
                    (r'यांची\s+मुलगी\s+कुमारी\s+([^\n]{3,80})$', 'यांची मुलगी कुमारी (end of line)'),
                    (r'यांचा\s+मुलगा\s+कुमार\s+([^\n]{3,80})$', 'यांचा मुलगा कुमार (end of line)'),
                    (r'कुमारी\s+([^\n]{3,80}?)(?:\s+यांना|\s+शैक्षणिक|$)', 'कुमारी + flexible end'),
                    (r'कुमार\s+([^\n]{3,80}?)(?:\s+यांना|\s+शैक्षणिक|$)', 'कुमार + flexible end'),
                    (r'यांचा\s+मुलगा\s+कुमार\s+([^\n]{3,80}?)\s+यांना', 'यांचा मुलगा कुमार + यांना'),
                    (r'यांचा\s+मुलगा\s+कुमार\s+([^\n]{3,80}?)\s+यांना', 'यांचा मुलगा कुमार + यांना'),
                    # Pattern 2: "यांची मुलगी कुमारी [NAME] यांना"
                    (r'यांची\s+मुलगी\s+कुमारी\s+([^\n]{3,80}?)\s+यांना', 'यांची मुलगी कुमारी + यांना'),
                    # Pattern 3: More flexible - allow newlines and look for शैक्षणिक as terminator
                    (r'यांचा\s+मुलगा\s+कुमार\s+(.{3,100}?)\s+यांना', 'यांचा मुलगा कुमार (flexible)'),
                    (r'यांची\s+मुलगी\s+कुमारी\s+(.{3,100}?)\s+यांना', 'यांची मुलगी कुमारी (flexible)'),
                    # Pattern 4: Handle OCR error where यांचा is merged with previous word
                    (r'[ाी]चा\s+मुलगा\s+कुमार\s+(.{3,100}?)\s+यांना', 'merged word + मुलगा कुमार'),
                    (r'[ाी]ची\s+मुलगी\s+कुमारी\s+(.{3,100}?)\s+यांना', 'merged word + मुलगी कुमारी'),
                    # Pattern 5: "मुलगा कुमार [NAME] यांना"
                    (r'मुलगा\s+कुमार\s+(.{3,100}?)\s+यांना', 'मुलगा कुमार'),
                    # Pattern 6: "मुलगी कुमारी [NAME] यांना"
                    (r'मुलगी\s+कुमारी\s+(.{3,100}?)\s+यांना', 'मुलगी कुमारी'),
                    # Pattern 7: If यांना is missing, try capturing until शैक्षणिक or other terminators
                    (r'यांची\s+मुलगी\s+कुमारी\s+([^\n]{3,80}?)(?:\s+यांना|\s+(?:यांच|शैक्षणिक))', 'यांची मुलगी (no यांना)'),
                    (r'यांचा\s+मुलगा\s+कुमार\s+([^\n]{3,80}?)(?:\s+यांना|\s+(?:यांच|शैक्षणिक))', 'यांचा मुलगा (no यांना)'),
                    # Pattern 8: Without कुमार/कुमारी prefix
                    (r'यांचा\s+मुलगा\s+(.{3,100}?)\s+यांना', 'यांचा मुलगा'),
                    (r'यांची\s+मुलगी\s+(.{3,100}?)\s+यांना', 'यांची मुलगी'),
                    # Pattern 9: Most general - just मुलगा/मुलगी followed by name
                    (r'मुलगा\s+(.{3,100}?)\s+यांना', 'मुलगा only'),
                    (r'मुलगी\s+(.{3,100}?)\s+यांना', 'मुलगी only'),
                ]
                
                for i, (pattern, description) in enumerate(patterns, 1):
                    match = re.search(pattern, raw_text)
                    if match:
                        student_name = match.group(1).strip()
                        print(f"    Pattern {i} ({description}) matched: '{student_name}'")
                        
                        # Clean up the extracted name
                        # Remove "कुमार" or "कुमारी" if still present at the start
                        student_name = re.sub(r'^(कुमारी\s*|कुमार\s*)', '', student_name)
                        # Remove any trailing text after the actual name
                        student_name = re.sub(r'\s+(यांचा|यांची|यांना|राहणार).*$', '', student_name)
                        # Clean up extra whitespace
                        student_name = ' '.join(student_name.split())
                        
                        # Validate the extracted name
                        if len(student_name) >= 2 and not re.search(r'[0-9०-९]{5,}', student_name):
                            print(f"    ✓ Extracted student_name: '{student_name}'")
                            data['student_name'] = student_name
                            break
                        else:
                            print(f"    ✗ Name looks invalid (len={len(student_name)}): '{student_name}'")
                
                if not data.get('student_name'):
                    print(f"    ✗ Could not extract valid student_name from text")
            
        # IMPROVED validation for income_value with mixed numeral support
        if raw_text and data.get('income_value'):
            income_val = str(data['income_value'])

            raw_text_normalized = raw_text.replace('o', '०').replace('O', '०')
            
            # Normalize both the extracted value and raw text for comparison
            # Convert Devanagari digits to English for comparison
            devanagari_to_english = str.maketrans('०१२३४५६७८९', '0123456789')
            
            # Normalize the extracted income value
            income_normalized = income_val.translate(devanagari_to_english)
            income_normalized = income_normalized.replace(',', '').replace('.', '').replace(' ', '')
            
            # Normalize the raw text for comparison
            raw_text_normalized = raw_text_normalized.translate(devanagari_to_english)

            
            
            # Try to find the income in raw text
            found_in_text = False
            
            # Check 1: Look for the normalized digits in normalized text
            if income_normalized in raw_text_normalized.replace(',', '').replace('.', '').replace(' ', ''):
                found_in_text = True
                print(f"✓ income_value '{income_val}' found in text (normalized match)")
            
            # Check 2: Look for the year line and verify income appears there
            if not found_in_text:
                year_match = re.search(r'२०२[३४५].*?२०२[४५६]?[^\n]{0,100}', raw_text)
                if year_match:
                    year_line = year_match.group(0)
                    year_line_normalized = year_line.translate(devanagari_to_english)
                    
                    # Check if our income value appears in this line (after normalization)
                    if income_normalized in year_line_normalized.replace(',', '').replace('.', '').replace(' ', ''):
                        found_in_text = True
                        print(f"✓ income_value '{income_val}' found in year line (normalized)")
                    else:
                        # Extract what's actually on that line
                        actual_numbers = re.findall(r'[०-९0-9,.-]+', year_line)
                        if actual_numbers:
                            print(f"  income_value '{income_val}' not exactly matched")
                            print(f"   Year line contains: {year_line}")
                            print(f"   Numbers found: {actual_numbers}")
                            
                            # Try fuzzy match - check if the significant digits are similar
                            for num in actual_numbers:
                                num_normalized = num.translate(devanagari_to_english).replace(',', '').replace('.', '')
                                # If the core digits match (ignoring minor OCR differences)
                                if len(num_normalized) > 3 and (
                                    income_normalized == num_normalized or  # Exact match
                                    income_normalized in num_normalized or   # Income is substring
                                    num_normalized in income_normalized or   # Num is substring
                                    income_normalized[:5] == num_normalized[:5]  # First 5 digits match
                                ):
                                    found_in_text = True
                                    print(f"   ✓ Found fuzzy match with '{num}' (normalized: {num_normalized})")
                                    break
                            
                            if not found_in_text:
                                print(f"   Setting income_value to null")
                                data['income_value'] = None
            
            if not found_in_text and data.get('income_value'):
                print(f" income_value '{income_val}' does NOT exist in raw text - setting to null")
                data['income_value'] = None
        
        # STRICT validation for income_in_word
        if raw_text and data.get('income_in_word'):
            income_word = str(data['income_in_word'])
            
            # Check if the words actually appear in the text
            # Look for key components
            words_in_text = False
            
            # Split into individual words and check if they exist
            word_parts = income_word.split()
            matches = 0
            for word in word_parts:
                if len(word) > 2 and word in raw_text:
                    matches += 1
            
            # If most words are found, consider it valid
            if matches >= len(word_parts) * 0.7:  # At least 70% of words must exist
                words_in_text = True
                print(f"✓ income_in_word '{income_word}' verified ({matches}/{len(word_parts)} words found)")
            else:
                # Try to find what's actually written
                year_match = re.search(r'२०२[३४५].*?[०-९0-9,.-]+\s+([^\n]{10,50})', raw_text)
                if year_match:
                    actual_words = year_match.group(1).strip()
                    print(f" income_in_word '{income_word}' NOT found in text!")
                    print(f"   Text actually says: '{actual_words}'")
                    print(f"   Setting income_in_word to null")
                    data['income_in_word'] = None
                else:
                    print(f" income_in_word '{income_word}' NOT verified - setting to null")
                    data['income_in_word'] = None
        
        # Clean income_in_word - remove any stray digits or incorrect prefixes
        if data.get('income_in_word'):
            words = str(data['income_in_word'])
            # Remove leading digits/numbers and common OCR errors
            words = re.sub(r'^[०-९0-9,.\s]+', '', words)
            data['income_in_word'] = words.strip() if words.strip() else None
        
        return data
    
    def _parse_json_response(self, response: str, doc_type: str, raw_text: str = "") -> dict:
        """Parse and clean JSON response from Llama"""
        
        try:
            cleaned = response.strip()
            
            # Remove markdown fences
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Try to find JSON object first
            start_obj = cleaned.find('{')
            end_obj = cleaned.rfind('}')
            
            if start_obj != -1 and end_obj != -1:
                json_str = cleaned[start_obj:end_obj + 1]
                parsed = json.loads(json_str)
                
                # Ensure document_type is set
                if 'document_type' not in parsed:
                    parsed['document_type'] = doc_type
                parsed = self._validate_and_fix_extraction(parsed, raw_text)
                
                return parsed
            
            # If no object found, try to find JSON array
            start_arr = cleaned.find('[')
            end_arr = cleaned.rfind(']')
            
            if start_arr != -1 and end_arr != -1:
                json_str = cleaned[start_arr:end_arr + 1]
                parsed_array = json.loads(json_str)
                
                print(f"  LLM returned array instead of object. Converting...")
                
                # Convert array to object based on schema
                schema = get_schema_for_doc_type(doc_type)
                schema_keys = list(schema.keys())
                
                parsed = {}
                for i, key in enumerate(schema_keys):
                    if i < len(parsed_array):
                        parsed[key] = parsed_array[i]
                    else:
                        parsed[key] = None
                
                # Ensure document_type is set
                if 'document_type' not in parsed:
                    parsed['document_type'] = doc_type
                    
                print(f"✓ Converted array to object: {list(parsed.keys())}")
                
                parsed = self._validate_and_fix_extraction(parsed, raw_text)
                return parsed
            
            raise ValueError("No JSON object or array found in response")
                
        except json.JSONDecodeError as e:
            print(f" JSON parsing error: {e}")
            print(f"Raw response: {response[:500]}")
            return {
                "error": "JSON parsing failed",
                "document_type": doc_type,
                "raw_response": response[:1000]
            }
        except Exception as e:
            print(f" Unexpected error: {e}")
            return {
                "error": str(e),
                "document_type": doc_type,
                "raw_response": response[:1000]
            }