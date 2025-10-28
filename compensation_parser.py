import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompensationParser:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        Hybrid parser that uses regex first, and falls back to a local LLM
        (runs fully offline, no bitsandbytes, works on ARM Macs via MPS).
        """
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Loading model '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device)

        self.re_pattern = re.compile(
            r"(?i)([A-Z]*\$|€|£)?\s*\$?([\d.,]+)([KMB])?(?:\s*[–-]\s*([A-Z]*\$|€|£)?\s*\$?([\d.,]+)([KMB])?)?"
        )
        self.currency_map = {
            "CA$": "CAD",
            "AUD$": "AUD",
            "NZ$": "NZD",
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",            
            "$": "USD",
        }

    def parse(self, text: str) -> dict:
        """Try regex first, then fallback to LLM if uncertain."""
        data = self._parse_with_regex(text)
        if not data or data["min_salary"] is None:
            llm_data = self._parse_with_local_llm(text)
            if llm_data:
                data.update({k: v for k, v in llm_data.items() if v is not None})
        return data

    def _parse_with_regex(self, s: str) -> dict:
        logger.info(f"Parsing compensation with regex: '{s}'")
        match = self.re_pattern.search(s)
        offers_equity = "equity" in s.lower()

        if not match:
            logger.warning(f"No regex match found for: '{s}'")
            return {"currency": None, "currency_symbol": None, "min_salary": None, "max_salary": None, "offers_equity": offers_equity, "text": s}

        def parse_amount(val: str, scale: str) -> int | None:
            if not val:
                return None
            val = val.replace(",", "")
            try:
                num = float(val)
                if scale:
                    scale = scale.upper()
                    if scale == "K":
                        num *= 1_000
                    elif scale == "M":
                        num *= 1_000_000
                    elif scale == "B":
                        num *= 1_000_000_000
                return int(num)
            except ValueError:
                return None

        currency = match.group(1) or match.group(4)
        logger.info(f"Extracted currency symbol: '{currency}'")
        currency_code = self._normalize_currency(currency)
        logger.info(f"Normalized currency code: '{currency_code}'")

        min_salary = parse_amount(match.group(2), match.group(3))
        logger.info(f"Extracted min_salary: {min_salary}")
        max_salary = parse_amount(match.group(5), match.group(6)) or min_salary
        logger.info(f"Extracted max_salary: {max_salary}")

        return {
            "currency": currency_code,
            "currency_symbol": currency,
            "min_salary": min_salary,
            "max_salary": max_salary,
            "offers_equity": offers_equity,
            "text": s,
        }

    def _normalize_currency(self, currency: str | None) -> str | None:
        logger.info(f"Normalizing currency: '{currency}'")
        if not currency:
            logger.warning(f"No currency found for: '{currency}'")
            return None
        currency = currency.strip().upper()
        logger.info(f"Stripped and uppercased currency: '{currency}'")
        for symbol, code in self.currency_map.items():
            logger.info(f"Checking currency symbol: '{symbol}'")
            if currency.startswith(symbol.replace("$", "")) or currency == symbol.upper():
                logger.info(f"Matched currency symbol '{symbol}' to code '{code}'")
                return code
            logger.info(f"No match found for currency symbol: '{symbol}'")

        if "CA$" in currency:
            logger.info(f"Direct Matched currency symbol 'CA$' to code 'CAD'")
            return "CAD"
        if currency in ("$", "USD$"):
            logger.info(f"Direct Matched currency symbol '$' to code 'USD'")
            return "USD"
        return currency

    def _parse_with_local_llm(self, text: str) -> dict:
        prompt = f"""
Extract structured compensation info from this text:
"{text}"

Return a valid JSON object with these fields:
{{
  "currency": (string or null),
  "currency_symbol": (string or null),
  "min_salary": (integer or null),
  "max_salary": (integer or null),
  "offers_equity": (boolean)
}}
"""
        logger.info(f"Generated prompt for LLM: {prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200)
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try to extract JSON
        start, end = raw_output.find("{"), raw_output.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                parsed = json.loads(raw_output[start:end])
                return {
                    "currency": parsed.get("currency"),
                    "currency_symbol": parsed.get("currency_symbol"),
                    "min_salary": parsed.get("min_salary"),
                    "max_salary": parsed.get("max_salary"),
                    "offers_equity": parsed.get("offers_equity", False),
                    "text": text,
                }
            except json.JSONDecodeError:
                pass
        return {}

    def parse_batch(self, texts: list[str]) -> list[dict]:
        """Batch parse a list of compensation strings."""
        return [self.parse(t) for t in texts]

