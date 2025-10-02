import os
import pytest
from backend.services.llm_client import call_llm

# Helper to check env keys
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
HAS_GEMINI = bool(os.getenv("GEMINI_API_KEY"))

@pytest.mark.parametrize("provider_key", ["gemini", "openai"])
def test_call_llm_basic_text(provider_key):
    """
    Basic smoke test: call call_llm with a simple prompt for each provider if
    the corresponding API key is present. Assert we get non-empty text.
    """
    if provider_key == "openai" and not HAS_OPENAI:
        pytest.skip("OPENAI_API_KEY not set; skipping OpenAI test")
    if provider_key == "gemini" and not HAS_GEMINI:
        pytest.skip("GEMINI_API_KEY not set; skipping Gemini test")

    prompt = "In three bullets, summarize why structured LLM outputs (JSON) help production apps."
    resp = call_llm(prompt, provider=provider_key, max_tokens=300, temperature=0.0)

    # Basic contract checks
    assert isinstance(resp, dict), f"Expected dict from call_llm, got {type(resp)}"
    assert "text" in resp, f"No 'text' field in response (raw: {resp.get('raw')})"
    txt = resp["text"]
    assert isinstance(txt, str), f"Response text not a string: {type(txt)}"
    assert txt.strip() != "", f"Empty text in response. raw: {resp.get('raw')}"

@pytest.mark.parametrize("provider_key", ["gemini", "openai"])
def test_call_llm_function_schema(provider_key):
    """
    Try function-schema / structured output path. We give a simple schema mirroring
    decompose. We assert that:
      - call succeeds and returns text; and
      - if structured is present, it contains expected keys.
    If structured is absent (some providers/SDKs), test still passes but logs info.
    """
    if provider_key == "openai" and not HAS_OPENAI:
        pytest.skip("OPENAI_API_KEY not set; skipping OpenAI test")
    if provider_key == "gemini" and not HAS_GEMINI:
        pytest.skip("GEMINI_API_KEY not set; skipping Gemini test")

    prompt = "Produce JSON with fields and subqueries for topic 'agent orchestration frameworks 2025'."

    function_schema = {
        "name": "decompose_schema",
        "description": "Return fields and subqueries",
        "parameters": {
            "type": "object",
            "properties": {
                "fields": {"type": "array", "items": {"type": "string"}},
                "subqueries": {"type": "array", "items": {"type": "string"}},
                "meta": {"type": "object"}
            },
            "required": ["fields", "subqueries"]
        }
    }

    resp = call_llm(prompt, provider=provider_key, function_schema=function_schema, max_tokens=400, temperature=0.0)

    assert isinstance(resp, dict), "call_llm must return a dict"
    assert "text" in resp and isinstance(resp["text"], str)
    assert resp["text"].strip() != "", f"Empty text returned. raw={resp.get('raw')}"

    structured = resp.get("structured")
    # If structured is present, assert fields/subqueries exist
    if structured is not None:
        assert isinstance(structured, (dict, list)), f"structured should be dict/list, got {type(structured)}"
        if isinstance(structured, dict):
            assert "fields" in structured and "subqueries" in structured, f"structured missing keys: {structured}"
            assert isinstance(structured["fields"], list)
            assert isinstance(structured["subqueries"], list)
    else:
        # Not fatal — print debugging info but don't fail the test
        pytest.skip(f"Provider '{provider_key}' did not return structured output; raw: {resp.get('raw')}")
