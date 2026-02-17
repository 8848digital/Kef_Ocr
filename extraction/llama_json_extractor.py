import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

from extraction.prompts import (
    get_schema_for_doc_type,
    create_extraction_prompt,
    get_system_prompt,
    detect_marksheet_board,
)
from .validators.ssc_validator import validate_and_fix_ssc_marksheet as validate_ssc
from .validators.cbse_validator import validate_and_fix_cbse_marksheet as validate_cbse




class LlamaJSONExtractor:
    """Extract structured JSON from OCR text using Llama 3.1B"""

    def __init__(self, model_name="/app/models/qwen"):
        print(f" Loading Llama model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if torch.cuda.is_available():
            dtype = torch.float16
            print(f" GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            dtype = torch.float32
            print("  No GPU detected, using CPU (this will be slower)")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )

        if torch.cuda.is_available():
            print(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        else:
            print("✓ Model loaded on CPU")

    # ──────────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ──────────────────────────────────────────────────────────────────────────

    def extract_json(self, raw_text: str, doc_type: str, temperature=0.1, max_tokens=1536):
        """
        Convert raw OCR text to structured JSON.
        CRITICAL: temperature=0.1 for more deterministic output.
        """
        schema = get_schema_for_doc_type(doc_type)
        prompt = create_extraction_prompt(raw_text, doc_type, schema)

        if "S/D/H/O" in raw_text or "S/D/H/o" in raw_text:
            print(" DEBUG: Found S/D/H/O pattern in raw text")
            lines = raw_text.split("\n")
            for i, line in enumerate(lines):
                if "S/D/H/O" in line or "S/D/H/o" in line:
                    print(f"    Line {i}: {line}")
                    if i + 1 < len(lines):
                        print(f"    Next line: {lines[i + 1]}")

        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user",   "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        print(f" DEBUG: LLM Response: {response[0:]}")

        return self._parse_json_response(response, doc_type, raw_text)

    # ──────────────────────────────────────────────────────────────────────────
    # DISPATCH — routes each document type to the correct validator
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_and_fix_extraction(self, data: dict, raw_text: str) -> dict:
        """Route extracted data to the appropriate board/doc-type validator."""

        if data.get("document_type") == "marksheet":
            board = detect_marksheet_board(raw_text)
            if board == "ssc":
                data = validate_ssc(data, raw_text)
            elif board == "cbse":
                data = validate_cbse(data, raw_text)
            
            return data

        if data.get("document_type") == "income_certificate":
            data = self._cleanup_income_certificate(data, raw_text)

        if data.get("document_type") not in ["pass_book", "passbook"]:
            return data

        # ── Passbook: validate parent_name ───────────────────────────────────
        if "parent_name" in data and data["parent_name"]:
            parent_name_value = str(data["parent_name"]).strip()

            print(f" Validating parent_name: '{parent_name_value}'")
            print(f" parent_name repr: {repr(parent_name_value)}")
            print(f" parent_name length: {len(parent_name_value)}")

            if not parent_name_value:
                data["parent_name"] = None
                return data

            parent_name_upper = parent_name_value.upper()

            location_keywords = [
                "NAGAR", "COLONY", "STREET", "ROAD", "MARG",
                "CHAWL", "TOWNSHIP", "AREA", "SOCIETY", "COMPLEX",
                "FLOOR", "BUILDING", "APARTMENT", "CHS", "COMPOUND",
            ]
            has_location = any(kw in parent_name_upper for kw in location_keywords)
            if has_location:
                matched = [kw for kw in location_keywords if kw in parent_name_upper]
                print(f"     Found location keywords: {matched}")

            address = data.get("address", "")
            in_address = False
            if address and len(parent_name_value) > 10:
                in_address = parent_name_upper in address.upper()
                if in_address:
                    print(f"     Found in address: '{address}'")

            special_char_count = len(re.findall(r"[^A-Za-z\s]", parent_name_value))
            special_char_ratio = special_char_count / max(len(parent_name_value), 1)
            is_too_short          = len(parent_name_value) < 4
            has_too_many_specials = special_char_ratio > 0.3
            has_garbage           = bool(re.search(r"(\d{5,}|[^A-Za-z\s]{3,})", parent_name_value))
            is_corrupted          = is_too_short or has_too_many_specials or has_garbage

            if is_corrupted:
                print(f"     Corruption check failed:")
                print(f"       - Too short ({len(parent_name_value)} chars): {is_too_short}")
                print(f"       - Too many special chars ({special_char_ratio:.2%}): {has_too_many_specials}")
                print(f"       - Has garbage pattern: {has_garbage}")

            if has_location or in_address or is_corrupted:
                print(f"  VALIDATION: Rejecting parent_name='{data['parent_name']}'")
                data["parent_name"] = None

        return data

    # ──────────────────────────────────────────────────────────────────────────
    # INCOME CERTIFICATE CLEANUP  (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _cleanup_income_certificate(self, data: dict, raw_text: str = "") -> dict:
        """Clean up common prefixes/suffixes in income certificate fields."""

        print(f" _cleanup_income_certificate called")
        print(f"   - Has raw_text: {bool(raw_text)} (length: {len(raw_text) if raw_text else 0})")
        print(f"   - Current student_name: {repr(data.get('student_name'))}")

        if data.get("parent_name"):
            name = str(data["parent_name"])
            name = re.sub(r"^(श्री॰\s*|श्री\s*)", "", name)
            name = re.sub(r"\s*(यांना|यांची|यांचा|राहणार)$", "", name)
            data["parent_name"] = name.strip() if name.strip() else None

        if data.get("student_name"):
            name = str(data["student_name"])
            name = re.sub(r"^(कुमारी\s*|कुमार\s*)", "", name)
            name = re.sub(r"\s*(यांना|याना)$", "", name)
            data["student_name"] = name.strip() if name.strip() else None

        if raw_text and not data.get("student_name"):
            indicators = ["मुलगा कुमार", "मुलगी कुमारी", "मुलगा", "मुलगी", "यांचा मुलगा", "यांची मुलगी"]
            found_indicators = [ind for ind in indicators if ind in raw_text]

            if found_indicators:
                print(f"  WARNING: Document mentions student but student_name is null")
                print(f"    Found indicators: {found_indicators}")
                print(f"    Attempting to extract student name...")

                student_section = re.search(r"सदरचा दाखला[^\n]{0,300}", raw_text)
                if student_section:
                    print(f"    Text section: {student_section.group(0)}")
                else:
                    student_section = re.search(r"मुलग[ाी][^\n]{0,250}", raw_text)
                    if student_section:
                        print(f"    Text section (alt): {student_section.group(0)}")

                patterns = [
                    (r"यांची\s+मुलगी\s+कुमारी\s+([^\n]{3,80})$",                    "यांची मुलगी कुमारी (end of line)"),
                    (r"यांचा\s+मुलगा\s+कुमार\s+([^\n]{3,80})$",                     "यांचा मुलगा कुमार (end of line)"),
                    (r"कुमारी\s+([^\n]{3,80}?)(?:\s+यांना|\s+शैक्षणिक|$)",          "कुमारी + flexible end"),
                    (r"कुमार\s+([^\n]{3,80}?)(?:\s+यांना|\s+शैक्षणिक|$)",           "कुमार + flexible end"),
                    (r"यांचा\s+मुलगा\s+कुमार\s+([^\n]{3,80}?)\s+यांना",            "यांचा मुलगा कुमार + यांना"),
                    (r"यांची\s+मुलगी\s+कुमारी\s+([^\n]{3,80}?)\s+यांना",           "यांची मुलगी कुमारी + यांना"),
                    (r"यांचा\s+मुलगा\s+कुमार\s+(.{3,100}?)\s+यांना",               "यांचा मुलगा कुमार (flexible)"),
                    (r"यांची\s+मुलगी\s+कुमारी\s+(.{3,100}?)\s+यांना",              "यांची मुलगी कुमारी (flexible)"),
                    (r"[ाी]चा\s+मुलगा\s+कुमार\s+(.{3,100}?)\s+यांना",              "merged word + मुलगा कुमार"),
                    (r"[ाी]ची\s+मुलगी\s+कुमारी\s+(.{3,100}?)\s+यांना",             "merged word + मुलगी कुमारी"),
                    (r"मुलगा\s+कुमार\s+(.{3,100}?)\s+यांना",                        "मुलगा कुमार"),
                    (r"मुलगी\s+कुमारी\s+(.{3,100}?)\s+यांना",                       "मुलगी कुमारी"),
                    (r"यांची\s+मुलगी\s+कुमारी\s+([^\n]{3,80}?)(?:\s+यांना|\s+(?:यांच|शैक्षणिक))", "यांची मुलगी (no यांना)"),
                    (r"यांचा\s+मुलगा\s+कुमार\s+([^\n]{3,80}?)(?:\s+यांना|\s+(?:यांच|शैक्षणिक))", "यांचा मुलगा (no यांना)"),
                    (r"यांचा\s+मुलगा\s+(.{3,100}?)\s+यांना",                        "यांचा मुलगा"),
                    (r"यांची\s+मुलगी\s+(.{3,100}?)\s+यांना",                        "यांची मुलगी"),
                    (r"मुलगा\s+(.{3,100}?)\s+यांना",                                "मुलगा only"),
                    (r"मुलगी\s+(.{3,100}?)\s+यांना",                                "मुलगी only"),
                ]

                for i, (pattern, description) in enumerate(patterns, 1):
                    match = re.search(pattern, raw_text)
                    if match:
                        student_name = match.group(1).strip()
                        print(f"    Pattern {i} ({description}) matched: '{student_name}'")
                        student_name = re.sub(r"^(कुमारी\s*|कुमार\s*)", "", student_name)
                        student_name = re.sub(r"\s+(यांचा|यांची|यांना|राहणार).*$", "", student_name)
                        student_name = " ".join(student_name.split())
                        if len(student_name) >= 2 and not re.search(r"[0-9०-९]{5,}", student_name):
                            print(f"    ✓ Extracted student_name: '{student_name}'")
                            data["student_name"] = student_name
                            break
                        else:
                            print(f"    ✗ Name looks invalid (len={len(student_name)}): '{student_name}'")

                if not data.get("student_name"):
                    print(f"    ✗ Could not extract valid student_name from text")

        if raw_text and data.get("income_value"):
            income_val = str(data["income_value"])
            raw_text_normalized = raw_text.replace("o", "०").replace("O", "०")
            dev_to_eng = str.maketrans("०१२३४५६७८९", "0123456789")
            income_normalized = (
                income_val.translate(dev_to_eng).replace(",", "").replace(".", "").replace(" ", "")
            )
            raw_text_normalized = raw_text_normalized.translate(dev_to_eng)
            found_in_text = False

            if income_normalized in raw_text_normalized.replace(",", "").replace(".", "").replace(" ", ""):
                found_in_text = True
                print(f"✓ income_value '{income_val}' found in text (normalized match)")

            if not found_in_text:
                year_match = re.search(r"२०२[३४५].*?२०२[४५६]?[^\n]{0,100}", raw_text)
                if year_match:
                    year_line = year_match.group(0)
                    year_line_normalized = year_line.translate(dev_to_eng)

                    if income_normalized in year_line_normalized.replace(",", "").replace(".", "").replace(" ", ""):
                        found_in_text = True
                        print(f"✓ income_value '{income_val}' found in year line (normalized)")
                    else:
                        actual_numbers = re.findall(r"[०-९0-9,.-]+", year_line)
                        if actual_numbers:
                            print(f"  income_value '{income_val}' not exactly matched")
                            for num in actual_numbers:
                                num_norm = num.translate(dev_to_eng).replace(",", "").replace(".", "")
                                if len(num_norm) > 3 and (
                                    income_normalized == num_norm
                                    or income_normalized in num_norm
                                    or num_norm in income_normalized
                                    or income_normalized[:5] == num_norm[:5]
                                ):
                                    found_in_text = True
                                    print(f"   ✓ Found fuzzy match with '{num}'")
                                    break
                            if not found_in_text:
                                print(f"   Setting income_value to null")
                                data["income_value"] = None

            if not found_in_text and data.get("income_value"):
                print(f" income_value '{income_val}' does NOT exist in raw text - setting to null")
                data["income_value"] = None

        if raw_text and data.get("income_in_word"):
            income_word = str(data["income_in_word"])
            word_parts = income_word.split()
            matches = sum(1 for w in word_parts if len(w) > 2 and w in raw_text)

            if matches >= len(word_parts) * 0.7:
                print(f"✓ income_in_word '{income_word}' verified ({matches}/{len(word_parts)} words found)")
            else:
                year_match = re.search(r"२०२[३४५].*?[०-९0-9,.-]+\s+([^\n]{10,50})", raw_text)
                if year_match:
                    actual_words = year_match.group(1).strip()
                    print(f" income_in_word NOT found. Text says: '{actual_words}'. Setting to null.")
                    data["income_in_word"] = None
                else:
                    print(f" income_in_word '{income_word}' NOT verified - setting to null")
                    data["income_in_word"] = None

        if data.get("income_in_word"):
            words = re.sub(r"^[०-९0-9,.\s]+", "", str(data["income_in_word"]))
            data["income_in_word"] = words.strip() if words.strip() else None

        return data

    # ──────────────────────────────────────────────────────────────────────────
    # JSON PARSING
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_json_response(self, response: str, doc_type: str, raw_text: str = "") -> dict:
        """Parse and clean JSON response from LLM, then dispatch to validator."""
        try:
            cleaned = response.strip()

            # Strip markdown fences
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Try JSON object
            start_obj, end_obj = cleaned.find("{"), cleaned.rfind("}")
            if start_obj != -1 and end_obj != -1:
                parsed = json.loads(cleaned[start_obj : end_obj + 1])
                if "document_type" not in parsed:
                    parsed["document_type"] = doc_type
                return self._validate_and_fix_extraction(parsed, raw_text)

            # Try JSON array (LLM occasionally returns one)
            start_arr, end_arr = cleaned.find("["), cleaned.rfind("]")
            if start_arr != -1 and end_arr != -1:
                parsed_array = json.loads(cleaned[start_arr : end_arr + 1])
                print("  LLM returned array instead of object. Converting...")
                schema_keys = list(get_schema_for_doc_type(doc_type).keys())
                parsed = {key: parsed_array[i] if i < len(parsed_array) else None
                          for i, key in enumerate(schema_keys)}
                parsed.setdefault("document_type", doc_type)
                print(f"✓ Converted array to object: {list(parsed.keys())}")
                return self._validate_and_fix_extraction(parsed, raw_text)

            raise ValueError("No JSON object or array found in response")

        except json.JSONDecodeError as e:
            print(f" JSON parsing error: {e}")
            print(f"Raw response: {response[:500]}")
            return {"error": "JSON parsing failed", "document_type": doc_type, "raw_response": response[:1000]}

        except Exception as e:
            print(f" Unexpected error: {e}")
            return {"error": str(e), "document_type": doc_type, "raw_response": response[:1000]}
