"""
backend/agents/decompose.py

Production-ready decomposition utility that works with both OpenAI and Gemini LLMs.

Requirements:
- backend.services.llm_client.call_llm(prompt, provider=None, **kwargs)
  The call_llm function should:
    - Accept a `provider` keyword ("openai" or "gemini") or None to use default.
    - Optionally accept `function_schema` (a dict) when using function-calling style.
    - Return either:
       * a string (raw LLM text)
       * or a dict-like structured response (OpenAI/Gemini function-calling-like).
  If your call_llm implementation differs, adapt the keyword names accordingly.

Behavior:
- Produce deterministic baseline decomposition always.
- If LLM available and use_llm=True, attempt one guarded function-calling call
  (best-effort). If the LLM response is invalid or parsing fails, fall back to baseline.
"""

from __future__ import annotations
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from functools import lru_cache
import time
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try importing call_llm from your LLM service wrapper.
try:
    from backend.services.llm_client import call_llm  # expected signature: call_llm(prompt, provider=None, **kwargs)
    LLM_AVAILABLE = True
except Exception:
    call_llm = None  # type: ignore
    LLM_AVAILABLE = False

# Default fields for market/vendor scans (tweak as needed)
DEFAULT_FIELDS = [
    "pricing",
    "key_features",
    "target_users",
    "notable_clients",
    "extracted_text",
    "limits",
    "deployment_options",
    "support_and_sla",
    "security_compliance",
    "region_availability",
]


@dataclass
class DecomposeResult:
    fields: List[str]
    subqueries: List[str]
    meta: Dict[str, Any]


def _normalize_field_name(s: str) -> str:
    s = s.strip().lower().replace("/", " ").replace("-", " ")
    s = " ".join(s.split())
    # prefer snake_case for multi-word fields
    s = s.replace(" ", "_")
    return s


@lru_cache(maxsize=1024)
def _default_decompose(topic: str, constraints: Optional[str]) -> DecomposeResult:
    t = topic.strip()
    constraints_text = (constraints.strip() if constraints else "").strip()
    fields = DEFAULT_FIELDS.copy()

    # heuristics: if topic mentions something specific, ensure it appears early
    lc = topic.lower()
    if "pricing" in lc and "pricing" not in fields:
        fields.insert(0, "pricing")
    if ("security" in lc or "compliance" in lc) and "security_compliance" not in fields:
        fields.append("security_compliance")

    subqueries = []
    for f in fields:
        if constraints_text:
            q = f"{t} {f} {constraints_text}"
        else:
            q = f"{t} {f}"
        subqueries.append(q)

    meta = {
        "source": "default",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic_length_words": len(t.split()),
    }
    return DecomposeResult(fields=fields, subqueries=subqueries, meta=meta)


def _build_llm_prompt(topic: str, constraints: Optional[str], default_fields: List[str]) -> str:
    constraints_text = constraints.strip() if constraints else ""
    prompt = f"""
You are a concise market-research assistant.

Task:
Given the user topic below, propose a compact list of fields that a market scan should collect (for example: pricing, key_features,
target_users, integrations). Then for each field provide a focused sub-query (<=12 words) that will be given to a web searcher.

Respond ONLY in JSON with this exact structure (no surrounding explanation):

{{
  "fields": ["pricing", "key_features", ...],
  "subqueries": ["<topic> pricing", "<topic> key_features", ...],
  "meta": {{
    "notes": "one-line explanation of choices",
    "priority_fields": ["pricing", "security_compliance"]
  }}
}}

Rules:
- Use at most 12 fields.
- Fields should be concise, lowercase, ideally single-word or snake_case.
- Subqueries must include the original topic phrase and be short.
- If constraints are provided, include them in subqueries to narrow search.
- Do NOT invent sources or claims. This step only outputs fields & subqueries.
- If uncertain, favor these default fields: {default_fields}

Input:
topic: \"\"\"{topic}\"\"\"
constraints: \"\"\"{constraints_text}\"\"\"
"""
    return prompt.strip()


def _extract_json_from_text(text: str) -> Any:
    """
    Find the first JSON object in text and parse it.
    Returns parsed object or raises ValueError.
    """
    if not text or not isinstance(text, str):
        raise ValueError("No text to parse for JSON")

    # common technique: find first { and last } reasonably
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # fallback: try to find array as top-level
        start_a = text.find("[")
        end_a = text.rfind("]")
        if start_a != -1 and end_a != -1 and end_a > start_a:
            fragment = text[start_a:end_a + 1]
            return json.loads(fragment)
        raise ValueError("No JSON object/array found in text")

    fragment = text[start:end + 1]

    # sanitize common LLM noise (trailing commas etc.)
    # Remove control characters
    fragment = re.sub(r"[\x00-\x1f]", "", fragment)
    # Attempt to load; if fails, try a looser approach by removing trailing commas
    try:
        return json.loads(fragment)
    except Exception:
        try:
            cleaned = re.sub(r",\s*}", "}", fragment)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            return json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON fragment: {e}") from e


def _parse_llm_response(resp: Any) -> Dict[str, Any]:
    """
    Accepts LLM response from backend.services.llm_client and extracts JSON payload
    matching our expected decomposition schema.
    Returns a dict parsed from JSON.
    """
    # Handle the new llm_client.py response format
    if isinstance(resp, dict):
        # Check if this is the new llm_client format with 'structured' field
        if "structured" in resp and resp["structured"] is not None:
            # Use the pre-parsed structured output
            structured = resp["structured"]
            if isinstance(structured, dict) and "fields" in structured and "subqueries" in structured:
                return structured
            # If structured is a list or other format, try to extract what we need
            if isinstance(structured, dict):
                return structured
        
        # Fallback: try to extract from text field
        if "text" in resp and resp["text"]:
            try:
                return _extract_json_from_text(resp["text"])
            except Exception:
                pass
        
        # Legacy support: check for direct fields (backward compatibility)
        if "fields" in resp and "subqueries" in resp:
            return resp
        
        # Legacy support: check for older response formats
        if "message" in resp:
            msg = resp.get("message")
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("text") or ""
                if content:
                    try:
                        return _extract_json_from_text(content)
                    except Exception:
                        pass
        
        # Legacy support: OpenAI choices format
        if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
            first = resp["choices"][0]
            msg = first.get("message") or first.get("text") or first.get("content")
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("text") or ""
            else:
                content = msg or ""
            if content:
                try:
                    return _extract_json_from_text(content)
                except Exception:
                    pass
        
        raise ValueError("Could not parse LLM response into required JSON format")
    
    # If resp is a string: search for JSON
    elif isinstance(resp, str):
        return _extract_json_from_text(resp)
    else:
        raise ValueError("Unsupported LLM response type")


def decompose_topic(topic: str,
                    constraints: Optional[str] = None,
                    use_llm: bool = True,
                    llm_provider: Optional[str] = None,
                    llm_timeout_secs: int = 8) -> Dict[str, Any]:
    """
    Decompose topic into fields & subqueries.

    Parameters:
      topic: user topic string
      constraints: optional narrowing constraints, e.g., "India, SMB"
      use_llm: when True and LLM available, attempt LLM refinement
      llm_provider: "openai" or "gemini" or None to use default
      llm_timeout_secs: max seconds to wait for LLM (best-effort; actual timeout governed by call_llm)

    Returns dict with keys: fields (list), subqueries (list), meta (dict)
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be a non-empty string")

    base = _default_decompose(topic, constraints)

    if not use_llm or not LLM_AVAILABLE:
        logger.debug("LLM not used; returning default decomposition")
        return asdict(base)

    # build prompt / function_schema
    prompt = _build_llm_prompt(topic, constraints, DEFAULT_FIELDS)

    # Define a simple function schema — many LLM wrappers accept a 'function_schema' or 'json_schema' argument.
    # This is a best-effort schema to encourage structured output.
    function_schema = {
        "name": "decompose_schema",
        "description": "Return fields and subqueries for market scan as JSON",
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of short field keywords (snake_case preferred)"
                },
                "subqueries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subqueries corresponding to fields"
                },
                "meta": {
                    "type": "object",
                    "description": "Optional meta info"
                }
            },
            "required": ["fields", "subqueries"]
        }
    }

    try:
        # Attempt to call LLM with a structured schema (if call_llm supports it).
        # The call_llm implementation in backend.services.llm_client should accept function_schema=...
        logger.debug("Calling LLM provider=%s for decomposition", llm_provider or "default")
        raw_resp = call_llm(prompt, provider=llm_provider, function_schema=function_schema, timeout=llm_timeout_secs)
    except TypeError:
        # call_llm doesn't accept function_schema; call without it
        try:
            logger.debug("call_llm does not accept function_schema; calling without it")
            raw_resp = call_llm(prompt, provider=llm_provider, timeout=llm_timeout_secs)
        except Exception as e:
            logger.exception("LLM call failed: %s", str(e))
            return asdict(base)
    except Exception as e:
        logger.exception("LLM call failed: %s", str(e))
        return asdict(base)

    # raw_resp may be str or dict; parse robustly
    try:
        parsed = _parse_llm_response(raw_resp)
        # Validate & normalize parsed content
        fields = parsed.get("fields", [])
        subqueries = parsed.get("subqueries", [])
        meta = parsed.get("meta", {})
        # Normalize fields
        norm_fields = []
        for f in fields:
            if not isinstance(f, str):
                continue
            nf = _normalize_field_name(f)
            if nf and nf not in norm_fields:
                norm_fields.append(nf)
        # If subqueries mismatch length, rebuild conservatively
        if not isinstance(subqueries, list) or len(subqueries) != len(norm_fields):
            subqueries = []
            for f in norm_fields:
                if constraints:
                    q = f"{topic} {f} {constraints}"
                else:
                    q = f"{topic} {f}"
                subqueries.append(q)
        if not norm_fields:
            logger.warning("LLM returned empty/invalid fields; falling back to default")
            return asdict(base)
        result = DecomposeResult(fields=norm_fields, subqueries=subqueries, meta={"source": "llm", **(meta or {})})
        return asdict(result)
    except Exception as e:
        logger.exception("Failed to parse LLM response to JSON: %s", str(e))
        # fallback to baseline
        return asdict(base)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Decompose topic into fields & subqueries")
    parser.add_argument("--topic", "-t", required=True, help="Topic phrase")
    parser.add_argument("--constraints", "-c", default=None, help="Optional constraints")
    parser.add_argument("--no-llm", action="store_true", help="Do not call the LLM")
    parser.add_argument("--provider", "-p", default=None, help="LLM provider to use (openai|gemini)")
    args = parser.parse_args()
    out = decompose_topic(args.topic, args.constraints, use_llm=not args.no_llm, llm_provider=args.provider)
    print(json.dumps(out, indent=2))
