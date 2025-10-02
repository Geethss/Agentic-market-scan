"""
backend/services/llm_client.py

Unified LLM client for OpenAI and Google Gemini (GenAI) with optional function-calling.

Design goals:
- Single call_llm() API used across the project.
- Support both providers (openai | gemini) and a sensible default if provider is None.
- Attempt function-calling / JSON schema when requested; gracefully fall back if provider/SDK doesn't support it.
- Return consistent structure:
    {
      "text": "<best text output>",
      "raw": <raw provider response object>,
      "structured": <parsed JSON if LLM returned a JSON blob or function arguments, else None>,
      "provider": "openai" | "gemini"
    }

Requirements / Notes:
- Expects your project config to have OPENAI_API_KEY and GEMINI_API_KEY set (see backend.config).
- This file tries common SDK call patterns used by current OpenAI and Google GenAI Python libs.
  SDKs evolve—if your installed SDK uses different function names, adapt the small call blocks accordingly.
"""

from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, Optional

from backend.config import cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Lazy imports to avoid hard dependency at module import time.
_openai_client = None
_genai_client = None


def _init_openai():
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    try:
        # New OpenAI Python SDK uses `OpenAI` class:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.OPENAI_API_KEY) if cfg.OPENAI_API_KEY else OpenAI()
        _openai_client = client
        logger.debug("OpenAI client initialized")
        return _openai_client
    except Exception as e:
        logger.debug("OpenAI client not initialized: %s", e)
        _openai_client = None
        return None


def _init_genai():
    global _genai_client
    if _genai_client is not None:
        return _genai_client

    # Two common variants in docs: `google.generativeai` or `google.genai`.
    # Try both import paths.
    try:
        try:
            import google.generativeai as genai  # older package name
            genai.configure(api_key=cfg.GEMINI_API_KEY)
            _genai_client = genai
            logger.debug("google.generativeai client initialized")
            return _genai_client
        except Exception:
            # try alternative package (google-genai)
            from google import genai  # newer google.genai client import style
            client = genai.Client()
            # Some wrappers ask to set api key via environment; if cfg exposed, try configure as well
            try:
                genai.configure(api_key=cfg.GEMINI_API_KEY)  # harmless if not supported
            except Exception:
                pass
            _genai_client = client
            logger.debug("google.genai client initialized")
            return _genai_client
    except Exception as e:
        logger.debug("Gemini client not initialized: %s", e)
        _genai_client = None
        return None


def _extract_json_from_text(text: str) -> Optional[Any]:
    """
    Try to extract a JSON object/array from the given text.
    Returns parsed JSON or None.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    # find first { and last }
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        fragment = s[start:end + 1]
        try:
            return json.loads(fragment)
        except Exception:
            try:
                # Remove trailing commas
                cleaned = fragment.replace(",}", "}").replace(",]", "]")
                return json.loads(cleaned)
            except Exception:
                return None
    # try array
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        fragment = s[start:end + 1]
        try:
            return json.loads(fragment)
        except Exception:
            return None
    return None


def call_llm(prompt: str,
             provider: Optional[str] = None,
             model: Optional[str] = None,
             function_schema: Optional[Dict] = None,
             max_tokens: Optional[int] = 1024,
             temperature: float = 0.0,
             timeout: Optional[int] = None,
             return_raw: bool = False) -> Dict[str, Any]:
    """
    Unified LLM call.

    Parameters
    ----------
    prompt: str
        The user/system prompt text (for simple text generation).
    provider: Optional[str]
        "openai" or "gemini". If None, prefer Gemini if key present, else OpenAI.
    model: Optional[str]
        Provider-specific model override (e.g., "gpt-4o", "gemini-1.5-flash"). If None, a sensible default is used.
    function_schema: Optional[dict]
        Schema guiding function-calling (JSON Schema-like) - passed if provider supports it.
    max_tokens: int
        Maximum tokens for response.
    temperature: float
        Sampling temperature.
    timeout: Optional[int]
        Best-effort timeout in seconds (passed to underlying SDK where supported).
    return_raw: bool
        If True, include raw provider response in "raw" field. Default False (still include raw where available).

    Returns
    -------
    dict: { "text": str, "raw": Any, "structured": Any, "provider": str }
    """
    chosen_provider = provider
    # Pick provider default: prefer Gemini if configured (free tier), else OpenAI
    if not chosen_provider:
        if cfg.GEMINI_API_KEY:
            chosen_provider = "gemini"
        elif cfg.OPENAI_API_KEY:
            chosen_provider = "openai"
        else:
            raise RuntimeError("No LLM provider configured (set GEMINI_API_KEY or OPENAI_API_KEY in env)")

    chosen_provider = chosen_provider.lower()

    # Normalize model defaults
    if chosen_provider == "openai":
        model = model or "gpt-4o-mini"
    else:
            model = model or "gemini-2.0-flash-exp"

    # Common return shape
    result = {"text": "", "raw": None, "structured": None, "provider": chosen_provider}

    # Try provider-specific calls with robust fallbacks
    if chosen_provider == "openai":
        client = _init_openai()
        if not client:
            raise RuntimeError("OpenAI SDK not available or OPENAI_API_KEY missing")
        try:
            # 1) Try Chat Completions / function calling if schema provided
            if function_schema:
                try:
                    # Many OpenAI SDKs accept 'functions' param. Map function_schema into functions list:
                    functions = [{
                        "name": function_schema.get("name", "fn"),
                        "description": function_schema.get("description", ""),
                        "parameters": function_schema.get("parameters", {})
                    }]
                    # chat/completions create
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": prompt}],
                        functions=functions,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout
                    )
                    result["raw"] = resp
                    # parse structured output if present (function_call arguments)
                    try:
                        # new SDK shape: resp.choices[0].message.function_call.arguments
                        msg = resp.choices[0].message
                        if getattr(msg, "function_call", None):
                            args_raw = msg.function_call.arguments
                            try:
                                parsed = json.loads(args_raw)
                                result["structured"] = parsed
                            except Exception:
                                result["structured"] = _extract_json_from_text(args_raw)
                        else:
                            # fallback parse text
                            content = getattr(msg, "content", None) or getattr(msg, "text", None) or ""
                            result["text"] = content if isinstance(content, str) else str(content)
                            result["structured"] = _extract_json_from_text(result["text"])
                    except Exception:
                        # best-effort: try to stringify and parse
                        try:
                            txt = resp.choices[0].message.content
                            result["text"] = txt
                            result["structured"] = _extract_json_from_text(txt)
                        except Exception:
                            pass
                    if not result["text"]:
                        # populate text if empty
                        try:
                            result["text"] = getattr(resp.choices[0].message, "content", "") or ""
                        except Exception:
                            result["text"] = ""
                    return result
                except TypeError as te:
                    logger.debug("OpenAI client didn't accept 'functions' or create signature: %s", te)
                    # fall through to simpler call
                except Exception as e:
                    logger.exception("OpenAI function-calling attempt failed: %s", e)
                    # fall through to plain call

            # 2) Try Responses.create (newer unified API)
            try:
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    timeout=timeout
                )
                result["raw"] = resp
                # resp.output_text helps get plain text
                text = getattr(resp, "output_text", None)
                if text:
                    result["text"] = text
                    result["structured"] = _extract_json_from_text(text)
                    return result
                # Try parsing choices
                try:
                    # Some responses have .output[0].content[0].text
                    els = getattr(resp, "output", None)
                    if els and isinstance(els, list) and len(els) > 0:
                        # flatten to string
                        combined = ""
                        for el in els:
                            # many shapes exist
                            if isinstance(el, dict):
                                for k in ("content", "text", "output_text"):
                                    if k in el:
                                        combined += str(el[k]) + "\n"
                            else:
                                combined += str(el) + "\n"
                        result["text"] = combined.strip()
                        result["structured"] = _extract_json_from_text(result["text"])
                        return result
                except Exception:
                    pass
            except Exception as e:
                logger.debug("OpenAI responses.create failed or not supported: %s", e)

            # 3) Fallback: chat.completions.create simple call without function schema
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
                result["raw"] = resp
                # parse text
                try:
                    # many SDKs: resp.choices[0].message.content
                    content = resp.choices[0].message.content
                    result["text"] = content if isinstance(content, str) else str(content)
                    result["structured"] = _extract_json_from_text(result["text"])
                except Exception:
                    # as a last resort stringify resp
                    result["text"] = str(resp)
                return result
            except Exception as e:
                logger.exception("OpenAI fallback chat completion failed: %s", e)
                raise
        except Exception as e:
            logger.exception("OpenAI call failed entirely: %s", e)
            raise

    elif chosen_provider == "gemini":
        genai = _init_genai()
        if not genai:
            raise RuntimeError("Gemini/GenAI SDK not available or GEMINI_API_KEY missing")

        try:
            # Modern Google Generative AI SDK (0.8.5+) uses GenerativeModel
            try:
                # Create a GenerativeModel instance
                model_instance = genai.GenerativeModel(model)
                
                # Generate content with the prompt
                resp = model_instance.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                
                result["raw"] = resp
                
                # Extract text from response
                if hasattr(resp, 'text') and resp.text:
                    result["text"] = resp.text
                    result["structured"] = _extract_json_from_text(resp.text)
                    return result
                else:
                    # Fallback: try to extract text from candidates
                    if hasattr(resp, 'candidates') and resp.candidates:
                        candidate = resp.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            text_parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    text_parts.append(part.text)
                            if text_parts:
                                result["text"] = ''.join(text_parts)
                                result["structured"] = _extract_json_from_text(result["text"])
                                return result
                    
                    # Last resort: stringify the response
                    result["text"] = str(resp)
                    result["structured"] = _extract_json_from_text(result["text"])
                    return result
                    
            except Exception as e:
                logger.debug("Modern Gemini API attempt failed: %s", e)
                raise
        except Exception as e:
            logger.exception("Gemini call failed: %s", e)
            raise

    else:
        raise ValueError(f"Unsupported provider: {chosen_provider}")

    # Fallback - should not reach here
    return result
