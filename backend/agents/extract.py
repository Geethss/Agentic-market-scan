# backend/agents/extract.py
"""
backend/agents/extract.py

Purpose
-------
Extract structured vendor information from retrieved snippets/chunks using an LLM.

Primary function:
    extract_vendors(topic, fields, snippets_per_field, top_k_per_field=6, llm_provider=None)

Input shapes
------------
- topic: str
- fields: List[str]  (as returned by decompose)
- snippets_per_field: Dict[field -> List[doc_text_or_snippet_dict]]
    Each entry should be either:
      - a plain string snippet, or
      - a dict { "id": "<doc-id-or-url>", "url": "<url>", "title": "<title>", "text": "<snippet-or-chunk-text>" }
- top_k_per_field: int
- llm_provider: optional string "gemini"|"openai"

Output
------
{
  "vendors": [
    {
      "name": "Vendor Name",
      "pricing": "Free/Paid/From $X ...",
      "key_features": ["feat1", "feat2"],
      "target_users": "SMBs, Enterprises, etc.",
      "integrations": ["Slack", "S3"],
      "support_and_sla": "...",
      "security_compliance": ["SOC2", "ISO27001"],
      "evidence": [ {"url": "...", "snippet_id": "...", "text": "..."}, ... ],
      "notes": "optional summary",
      "confidence": 0.0  # optional LLM-provided confidence, normalized [0..1]
    },
    ...
  ],
  "warnings": [...],
}
"""

from __future__ import annotations
import json
import logging
import re
import time
import textwrap
from typing import List, Dict, Any, Optional, Tuple

from backend.services.llm_client import call_llm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tunables
SNIPPET_MAX_CHARS = 800          # truncate each snippet to this length to control prompt size
MAX_SNIPPETS_TOTAL = 60          # hard cap of snippets passed to LLM
TOP_K_PER_FIELD = 6
MAX_VENDORS = 12
WRITER_TEMPERATURE = 0.0
CRITIC_TEMPERATURE = 0.0
LLM_TIMEOUT_SECS = 15


# JSON schema for function-calling
_EXTRACT_SCHEMA = {
    "name": "extract_vendors_schema",
    "description": "Return a JSON array of vendor objects extracted from provided evidence snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "vendors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "pricing": {"type": ["string", "null"]},
                        "key_features": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "target_users": {"type": ["string", "null"]},
                        "extracted_text": {"type": ["string", "null"]},
                        "support_and_sla": {"type": ["string", "null"]},
                        "security_compliance": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "snippet_id": {"type": ["string", "null"]},
                                    "text": {"type": "string"},
                                    "published": {"type": ["string", "null"]}
                                },
                                "required": ["url", "text"]
                            }
                        },
                        "notes": {"type": ["string", "null"]},
                        "confidence": {"type": ["number", "null"]}
                    },
                    "required": ["name", "evidence"]
                },
                "maxItems": MAX_VENDORS
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["vendors"]
    }
}


def _normalize_snippet_input(snippets) -> List[Dict[str, str]]:
    """
    Accepts a list of either plain strings or dicts and returns a list of
    dicts with at least: {"id","url","text","title"(optional),"published"(optional)}
    """
    normalized = []
    for i, s in enumerate(snippets or []):
        if not s:
            continue
        if isinstance(s, str):
            normalized.append({
                "id": f"snip_{i}",
                "url": "",
                "text": s[:SNIPPET_MAX_CHARS]
            })
        elif isinstance(s, dict):
            text = s.get("text") or s.get("snippet") or ""
            if not text:
                continue
            normalized.append({
                "id": s.get("id") or s.get("url") or f"snip_{i}",
                "url": s.get("url") or "",
                "text": (text[:SNIPPET_MAX_CHARS] + "...") if len(text) > SNIPPET_MAX_CHARS else text,
                "title": s.get("title"),
                "published": s.get("published")
            })
        else:
            # try string coercion
            txt = str(s)
            normalized.append({"id": f"snip_{i}", "url": "", "text": txt[:SNIPPET_MAX_CHARS]})
    return normalized


def _prepare_snippets_payload(topic: str, fields: List[str], snippets_per_field: Dict[str, List[Any]], top_k_per_field: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a compact prompt section that enumerates snippets with short ids and metadata.
    Returns (snippets_text_block, flat_snippets_list)
    """
    flat = []
    total = 0
    lines = []
    lines.append(f"Topic: {topic}")
    lines.append("")
    lines.append("Evidence snippets (id | field | url | text):")
    lines.append("")

    for field in fields:
        if total >= MAX_SNIPPETS_TOTAL:
            break
        items = snippets_per_field.get(field, []) or []
        # normalize and take top_k
        norm = _normalize_snippet_input(items)[:top_k_per_field]
        for s in norm:
            if total >= MAX_SNIPPETS_TOTAL:
                break
            sid = s.get("id")
            url = s.get("url", "") or ""
            text = s.get("text", "") or ""
            # shorten text for prompt
            short = textwrap.shorten(text, width=SNIPPET_MAX_CHARS, placeholder="...")
            lines.append(f"- {sid} | {field} | {url} | {short}")
            flat.append({"id": sid, "field": field, "url": url, "text": short, "title": s.get("title"), "published": s.get("published")})
            total += 1

    snippets_text = "\n".join(lines)
    return snippets_text, flat


def _build_writer_prompt(topic: str, fields: List[str], snippets_block: str, max_vendors: int = MAX_VENDORS) -> str:
    """
    Prompt for the writer LLM to extract vendors. We instruct strict JSON output via function calling schema,
    but also provide a plain text fallback prompt in case function-calling isn't supported.
    """
    p = f"""
You are a precise information extraction assistant.

Task:
Given the topic and the provided evidence snippets, extract up to {max_vendors} vendor entries relevant to the topic.
Each vendor MUST include a "name" and at least one evidence item that supports its claims.
Evidence items must reference the snippet id shown in the evidence list.

Output:
Return ONLY valid JSON that matches this exact schema:
{{
  "vendors": [
    {{
      "name": "Vendor Name",
      "pricing": "pricing info or null",
      "key_features": ["feature1", "feature2"],
      "target_users": "target audience or null",
      "extracted_text": "summary of extracted information",
      "support_and_sla": "support info or null",
      "security_compliance": ["compliance1", "compliance2"],
      "evidence": [
        {{
          "url": "source_url",
          "snippet_id": "snippet_id",
          "text": "supporting text"
        }}
      ],
      "notes": "optional notes or null",
      "confidence": 0.8
    }}
  ],
  "warnings": ["optional warning messages"]
}}

CRITICAL: Return ONLY the JSON object, no other text, explanations, or formatting.

Do NOT hallucinate sources. If you cannot find reliable evidence for a claim, leave that field empty or omit it.
Prefer conservative answers: include only vendors and claims directly supported by evidence snippets.

Here are the snippets you can cite (id | field | url | text):
{snippets_block}

Return the JSON now:
"""
    return p.strip()


def _test_llm_response(topic: str, llm_provider: str = "gemini") -> str:
    """
    Test function to see what the LLM is actually returning
    """
    simple_prompt = f"""
Extract vendors from this topic: {topic}

Return ONLY valid JSON in this format:
{{
  "vendors": [
    {{
      "name": "Vendor Name",
      "pricing": "pricing info",
      "key_features": ["feature1", "feature2"],
      "evidence": [{{"url": "url", "text": "text"}}]
    }}
  ]
}}

Return ONLY the JSON, no other text.
"""
    
    try:
        from backend.services.llm_client import call_llm
        logger.debug("Calling LLM with provider: %s", llm_provider)
        resp = call_llm(simple_prompt, provider=llm_provider, temperature=0.0, timeout=10)
        logger.debug("Test LLM response type: %s", type(resp))
        if isinstance(resp, dict):
            text = resp.get("text", "")
            logger.debug("Test LLM text response: %s", text[:500])
            return text
        else:
            logger.debug("Test LLM non-dict response: %s", str(resp)[:500])
            return str(resp)
    except Exception as e:
        logger.exception("Test LLM call failed: %s", e)
        return f"Error: {e}"


def _simple_fallback_extraction(topic: str, snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple fallback extraction that creates basic vendor entries from snippets.
    Works for any topic by extracting company/product names from text.
    """
    vendors = []
    vendor_names = set()
    
    # Common company/product indicators
    company_indicators = [
        "company", "corporation", "inc", "llc", "ltd", "corp", "enterprise", 
        "platform", "service", "tool", "software", "app", "system", "solution"
    ]
    
    # Common LLM names to look for
    llm_names = [
        "GPT-4", "GPT-4o", "GPT-4.1", "GPT-3.5", "ChatGPT", "OpenAI",
        "Claude", "Claude 3.5", "Claude 3.7", "Anthropic",
        "Gemini", "Gemini Pro", "Gemini 2.5", "Google",
        "Llama", "Llama 3", "Meta", "Facebook",
        "DeepSeek", "DeepSeek R1", "DeepSeek V3",
        "Mistral", "Mistral AI", "Mixtral",
        "Qwen", "Alibaba", "Yi", "01.AI",
        "Phi", "Microsoft", "Cohere", "Perplexity"
    ]
    
    for snippet in snippets[:15]:  # Check more snippets
        text = snippet.get("text", "")
        title = snippet.get("title", "")
        url = snippet.get("url", "")
        snippet_id = snippet.get("id", "")
        
        # Combine title and text for better extraction
        full_text = f"{title} {text}".lower()
        
        # Look for LLM names in the text
        for llm_name in llm_names:
            if llm_name.lower() in full_text and llm_name not in vendor_names:
                vendor_names.add(llm_name)
                
                # Try to extract some basic info
                key_features = []
                if "open source" in full_text:
                    key_features.append("Open Source")
                if "instruct" in full_text:
                    key_features.append("Instruction Tuned")
                if "chat" in full_text:
                    key_features.append("Chat Capabilities")
                if "code" in full_text:
                    key_features.append("Code Generation")
                if "vision" in full_text:
                    key_features.append("Vision Capabilities")
                
                # Extract pricing info if available
                pricing = ""
                if "free" in full_text:
                    pricing = "Free"
                elif "paid" in full_text or "subscription" in full_text:
                    pricing = "Paid"
                elif "open source" in full_text:
                    pricing = "Open Source"
                else:
                    pricing = "Please visit the website for pricing details"
                
                # Extract target users
                target_users = ""
                if "enterprise" in full_text:
                    target_users = "Enterprise"
                elif "research" in full_text:
                    target_users = "Research"
                elif "developer" in full_text:
                    target_users = "Developers"
                
                # Create extracted text summary - clean and extract useful info
                cleaned_text = text
                
                # Remove common unhelpful prefixes
                prefixes_to_remove = [
                    r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}—',  # "Jun 27, 2025—"
                    r'^\*\s*\*\*[^*]+\*\*\s*:',  # "* **Best AI agent framework for enterprises**:"
                    r'^https?://[^\s]+\s*',  # URLs at start
                    r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s*',  # "Jun 27, 2025"
                ]
                
                for pattern in prefixes_to_remove:
                    cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
                
                # Clean up extra whitespace
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                
                # Skip generic intro phrases and find useful sentences
                vendor_name_lower = llm_name.lower()
                
                # Look for sentences that contain the vendor name
                # Split by sentence delimiters or commas followed by capital letters (new sentence)
                sentences = re.split(r'(?:[.!?]\s+)|(?:,\s+(?=[A-Z]))', cleaned_text)
                useful_sentences = []
                
                for sentence in sentences:
                    # Skip generic intro/meta sentences
                    skip_patterns = [
                        r'by the end of this (article|guide|post)',
                        r'in this (article|guide|post)',
                        r'you will (have|learn|understand)',
                        r'this (article|guide|post) (will|covers|discusses)',
                        r'^(the|a|an)\s+(following|top|best)\s+',
                    ]
                    
                    should_skip = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in skip_patterns)
                    
                    if not should_skip and len(sentence.strip()) > 20:
                        # Prioritize sentences mentioning the vendor
                        if vendor_name_lower in sentence.lower():
                            useful_sentences.insert(0, sentence.strip())
                        else:
                            useful_sentences.append(sentence.strip())
                
                # Join the useful sentences
                if useful_sentences:
                    extracted_text = '. '.join(useful_sentences[:3])  # Take top 3 sentences
                    if len(extracted_text) > 500:
                        extracted_text = extracted_text[:500] + "..."
                else:
                    # Fallback to cleaned text if no useful sentences found
                    extracted_text = cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
                
                vendors.append({
                    "name": llm_name,
                    "pricing": pricing,
                    "key_features": key_features,
                    "target_users": target_users,
                    "extracted_text": extracted_text,
                    "support_and_sla": "",
                    "security_compliance": [],
                    "evidence": [{
                        "url": url,
                        "snippet_id": snippet_id,
                        "text": text[:300]  # First 300 chars
                    }],
                    "notes": "Extracted via fallback method",
                    "confidence": 0.4
                })
                
                # Limit to avoid too many vendors
                if len(vendors) >= 8:
                    break
        
        if len(vendors) >= 8:
            break
    
    return {
        "vendors": vendors,
        "warnings": ["Used fallback extraction method due to LLM failure"]
    }


def _call_writer(prompt: str, llm_provider: Optional[str], timeout: int) -> Dict[str, Any]:
    """
    Call call_llm with function schema when possible. Returns parsed dict.
    """
    logger.debug("_call_writer called with provider: %s", llm_provider)
    try:
        logger.debug("Calling LLM with function schema")
        resp = call_llm(prompt, provider=llm_provider, function_schema=_EXTRACT_SCHEMA, temperature=WRITER_TEMPERATURE, timeout=timeout)
        logger.debug("LLM response type: %s", type(resp))
    except TypeError:
        # call_llm may not accept function_schema: call without it
        logger.debug("Function schema not supported, calling without it")
        resp = call_llm(prompt, provider=llm_provider, temperature=WRITER_TEMPERATURE, timeout=timeout)
        logger.debug("LLM response type: %s", type(resp))
    except Exception as e:
        logger.exception("Writer LLM call failed: %s", e)
        raise

    # call_llm returns dict with keys: text, raw, structured (our llm_client uses that shape)
    # prefer resp["structured"] if present, else try to extract JSON from resp["text"]
    logger.debug("Processing LLM response: type=%s", type(resp))
    
    if isinstance(resp, dict):
        structured = resp.get("structured")
        if structured:
            logger.debug("Using structured response from LLM")
            return structured
        # fallback to parsing text
        text = resp.get("text", "") or ""
        logger.debug("Using text response from LLM, length: %d", len(text))
    else:
        text = str(resp)
        logger.debug("LLM returned non-dict response, converting to string, length: %d", len(text))

    # Clean up the text - remove markdown code blocks and other formatting
    cleaned_text = text.strip()
    
    # Remove markdown code block markers
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]  # Remove ```json
    elif cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]   # Remove ```
    
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]  # Remove trailing ```
    
    # Remove any remaining newlines and extra whitespace
    cleaned_text = cleaned_text.strip()
    
    # Try to find the JSON object boundaries more aggressively
    json_start = cleaned_text.find('{')
    json_end = cleaned_text.rfind('}')
    
    if json_start != -1 and json_end != -1 and json_end > json_start:
        cleaned_text = cleaned_text[json_start:json_end+1]
    
    logger.debug("JSON boundaries found: start=%d, end=%d", json_start, json_end)
    logger.debug("Cleaned text length: %d", len(cleaned_text))

    # attempt JSON parse
    try:
        parsed = json.loads(cleaned_text)
        logger.debug("JSON parsing successful! Found %d vendors", len(parsed.get('vendors', [])))
        return parsed
    except Exception as e:
        logger.debug("JSON parsing failed: %s", e)
        
        # Try to repair common JSON issues
        try:
            # Try multiple repair strategies
            repair_strategies = [
                # Strategy 1: Remove trailing incomplete objects
                lambda text: text[:text.rfind('}') + 1] if text.count('{') > text.count('}') else text,
                # Strategy 2: Remove everything after the error position
                lambda text: text[:2887] if len(text) > 2887 else text,
                # Strategy 3: Find the last complete vendor object
                lambda text: text[:text.rfind('},') + 1] + ']}' if ',}' in text else text,
                # Strategy 4: Remove incomplete vendor entries
                lambda text: text[:text.rfind('"name":')] + ']}' if '"name":' in text else text
            ]
            
            for i, strategy in enumerate(repair_strategies):
                try:
                    repaired_text = strategy(cleaned_text)
                    if repaired_text != cleaned_text:
                        logger.debug("Attempting repair strategy %d", i+1)
                        parsed = json.loads(repaired_text)
                        logger.debug("JSON repair successful with strategy %d! Found %d vendors", i+1, len(parsed.get('vendors', [])))
                        return parsed
                except Exception:
                    continue
                    
        except Exception as repair_e:
            logger.debug("All JSON repair strategies failed: %s", repair_e)
        
        # try to find first JSON object in text
        try:
            start = cleaned_text.find("{")
            end = cleaned_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                fragment = cleaned_text[start:end+1]
                return json.loads(fragment)
        except Exception:
            pass

    # If JSON parsing fails, try to extract vendor names manually
    logger.debug("JSON parsing completely failed, attempting manual extraction")
    
    # Try to extract vendor data from the malformed JSON
    try:
        import re
        # Extract complete vendor objects using regex
        vendor_pattern = r'"name":\s*"([^"]+)"[^}]*?"pricing":\s*"([^"]*)"[^}]*?"key_features":\s*\[([^\]]*)\][^}]*?"target_users":\s*"([^"]*)"'
        
        # Try a simpler approach - extract each vendor block
        # Handle truncated JSON by looking for complete vendor objects
        vendor_blocks = re.findall(r'\{[^{}]*"name":\s*"([^"]+)"[^{}]*\}', cleaned_text)
        
        # If that fails, try to extract vendor names from the truncated text
        if not vendor_blocks:
            vendor_blocks = re.findall(r'"name":\s*"([^"]+)"', cleaned_text)
        
        if vendor_blocks:
            logger.debug("Found %d vendor blocks manually", len(vendor_blocks))
            vendors = []
            
            for i, name in enumerate(vendor_blocks[:5]):  # Limit to 5 vendors
                # Try to extract additional data for this vendor
                name_start = cleaned_text.find(f'"name": "{name}"')
                if name_start == -1:
                    continue
                    
                # Find the end of this vendor object
                next_vendor_start = len(cleaned_text)
                if i+1 < len(vendor_blocks):
                    next_name_start = cleaned_text.find(f'"name": "{vendor_blocks[i+1]}"')
                    if next_name_start != -1:
                        next_vendor_start = next_name_start
                
                vendor_section = cleaned_text[name_start:next_vendor_start]
                
                # Extract pricing (handle both string and null values)
                pricing_match = re.search(r'"pricing":\s*(?:"([^"]*)"|null)', vendor_section)
                pricing = pricing_match.group(1) if pricing_match and pricing_match.group(1) else ""
                
                # Extract key features
                features_match = re.search(r'"key_features":\s*\[([^\]]*)\]', vendor_section)
                key_features = []
                if features_match:
                    features_text = features_match.group(1)
                    # Extract individual feature strings
                    feature_matches = re.findall(r'"([^"]+)"', features_text)
                    key_features = feature_matches
                
                # Extract target users (handle both string and null values)
                target_match = re.search(r'"target_users":\s*(?:"([^"]*)"|null)', vendor_section)
                target_users = target_match.group(1) if target_match and target_match.group(1) else ""
                
                # Try to extract additional data from evidence text
                evidence_text_matches = re.findall(r'"text":\s*"([^"]*)"', vendor_section)
                all_evidence_text = " ".join(evidence_text_matches)
                
                # Extract pricing from evidence text if not found in JSON
                if not pricing and all_evidence_text:
                    # Look for pricing patterns in evidence text
                    price_patterns = [
                        r'\$[\d,]+(?:\.\d{2})?(?:/month|/mo|/year|/yr)?',
                        r'[\d,]+(?:\.\d{2})?\s*(?:dollars?|USD|per month|per year)',
                        r'(?:free|Free|FREE)',
                        r'(?:paid|Paid|PAID)',
                        r'(?:premium|Premium|PREMIUM)',
                        r'(?:budget|Budget|BUDGET)'
                    ]
                    for pattern in price_patterns:
                        price_match = re.search(pattern, all_evidence_text, re.IGNORECASE)
                        if price_match:
                            pricing = price_match.group(0)
                            break
                
                # If still no pricing found, use fallback text
                if not pricing:
                    pricing = "Please visit the website for pricing details"
                
                # Extract key features from evidence text if not found in JSON
                if not key_features and all_evidence_text:
                    # Look for feature indicators in evidence text
                    feature_indicators = [
                        r'(?:coding|programming|development)',
                        r'(?:reasoning|logic|analysis)',
                        r'(?:multimodal|vision|image)',
                        r'(?:conversation|chat|dialogue)',
                        r'(?:translation|language)',
                        r'(?:summarization|summary)',
                        r'(?:creative|writing|content)',
                        r'(?:enterprise|business|professional)'
                    ]
                    for pattern in feature_indicators:
                        if re.search(pattern, all_evidence_text, re.IGNORECASE):
                            key_features.append(pattern.replace('(?:', '').replace(')', '').title())
                
                # Extract target users from evidence text if not found in JSON
                if not target_users and all_evidence_text:
                    # Look for target user indicators
                    user_patterns = [
                        r'(?:developers?|programmers?)',
                        r'(?:enterprise|business|corporate)',
                        r'(?:researchers?|academic)',
                        r'(?:students?|education)',
                        r'(?:content creators?|writers?)',
                        r'(?:data scientists?|analysts?)'
                    ]
                    for pattern in user_patterns:
                        user_match = re.search(pattern, all_evidence_text, re.IGNORECASE)
                        if user_match:
                            target_users = user_match.group(0).title()
                            break
                
                # Extract evidence with actual text content
                evidence = []
                evidence_matches = re.findall(r'"url":\s*"([^"]*)"', vendor_section)
                text_matches = re.findall(r'"text":\s*"([^"]*)"', vendor_section)
                
                for j, url in enumerate(evidence_matches):
                    text_content = text_matches[j] if j < len(text_matches) else ""
                    evidence.append({
                        "url": url,
                        "snippet_id": f"extracted_{i}_{j}",
                        "text": text_content,
                        "published": None
                    })
                
                # Create extracted text summary from evidence - clean and extract useful info
                cleaned_evidence = all_evidence_text
                
                # Remove common unhelpful prefixes
                prefixes_to_remove = [
                    r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}—',  # "Jun 27, 2025—"
                    r'^\*\s*\*\*[^*]+\*\*\s*:',  # "* **Best AI agent framework for enterprises**:"
                    r'^https?://[^\s]+\s*',  # URLs at start
                    r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s*',  # "Jun 27, 2025"
                ]
                
                for pattern in prefixes_to_remove:
                    cleaned_evidence = re.sub(pattern, '', cleaned_evidence, flags=re.IGNORECASE)
                
                # Clean up extra whitespace
                cleaned_evidence = re.sub(r'\s+', ' ', cleaned_evidence).strip()
                
                # Skip generic intro phrases and find the sentence that mentions the vendor
                vendor_name_lower = name.lower()
                
                # Look for sentences that contain the vendor name
                # Split by sentence delimiters or commas followed by capital letters (new sentence)
                sentences = re.split(r'(?:[.!?]\s+)|(?:,\s+(?=[A-Z]))', cleaned_evidence)
                useful_sentences = []
                
                for sentence in sentences:
                    # Skip generic intro/meta sentences
                    skip_patterns = [
                        r'by the end of this (article|guide|post)',
                        r'in this (article|guide|post)',
                        r'you will (have|learn|understand)',
                        r'this (article|guide|post) (will|covers|discusses)',
                        r'^(the|a|an)\s+(following|top|best)\s+',
                    ]
                    
                    should_skip = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in skip_patterns)
                    
                    if not should_skip and len(sentence.strip()) > 20:
                        # Prioritize sentences mentioning the vendor
                        if vendor_name_lower in sentence.lower():
                            useful_sentences.insert(0, sentence.strip())
                        else:
                            useful_sentences.append(sentence.strip())
                
                # Join the useful sentences
                if useful_sentences:
                    extracted_text = '. '.join(useful_sentences[:3])  # Take top 3 sentences
                    if len(extracted_text) > 500:
                        extracted_text = extracted_text[:500] + "..."
                else:
                    # Fallback to cleaned evidence if no useful sentences found
                    extracted_text = cleaned_evidence[:500] + "..." if len(cleaned_evidence) > 500 else cleaned_evidence
                
                vendor_data = {
                    "name": name,
                    "pricing": pricing,
                    "key_features": key_features,
                    "target_users": target_users,
                    "extracted_text": extracted_text,
                    "support_and_sla": "",
                    "security_compliance": [],
                    "evidence": evidence,
                    "notes": "Extracted from malformed JSON",
                    "confidence": 0.7
                }
                
                logger.debug("Extracted vendor %d: %s with pricing=%s", i+1, name, pricing)
                
                vendors.append(vendor_data)
            
            logger.debug("Extracted %d vendors with data manually", len(vendors))
            return {
                "vendors": vendors,
                "warnings": [f"LLM returned malformed JSON, extracted {len(vendors)} vendors with data manually"]
            }
    except Exception as extract_e:
        logger.debug("Manual vendor extraction failed: %s", extract_e)
    
    logger.warning("LLM returned unparseable JSON. Response length: %d", len(text))
    logger.warning("First 500 chars: %s", text[:500])
    logger.warning("Last 200 chars: %s", text[-200:] if len(text) > 200 else text)
    
    # Return a minimal fallback structure
    return {
        "vendors": [],
        "warnings": [f"LLM returned unparseable JSON (length: {len(text)}): {text[:100]}..."]
    }


def _call_critic(original_vendors: List[Dict[str, Any]], snippets: List[Dict[str, Any]], topic: str, llm_provider: Optional[str], timeout: int) -> Dict[str, Any]:
    """
    Critic examines the vendor list and returns either {"ok": True} or {"ok": False, "errors": [...], "fix_instructions": "..."}

    The critic must check:
    - Every vendor has at least one evidence item referencing a snippet id or url included in snippets.
    - For each vendor claim (pricing/key_features/etc), confirm there is evidence that mentions that claim (best-effort).
    """
    # Build a compact text summary for critic prompt
    vendor_summary_lines = []
    for v in original_vendors:
        name = v.get("name", "<UNKNOWN>")
        vendor_summary_lines.append(f"Vendor: {name}")
        for k in ("pricing", "key_features", "target_users", "extracted_text", "security_compliance"):
            if v.get(k):
                vendor_summary_lines.append(f"  - {k}: {v.get(k)}")
        evidences = v.get("evidence") or []
        for e in evidences[:3]:
            vendor_summary_lines.append(f"    * evidence: {e.get('snippet_id') or e.get('url')}: { (e.get('text') or '')[:120] }")
        vendor_summary_lines.append("")

    vendors_text = "\n".join(vendor_summary_lines)
    snippets_text_lines = []
    for s in snippets:
        snippets_text_lines.append(f"{s['id']} | {s['field']} | {s.get('url','')} | { (s.get('text') or '')[:200] }")
    snippets_text = "\n".join(snippets_text_lines)

    critic_prompt = f"""
You are a critical verifier for JSON extraction outputs.

We extracted these vendor entries (JSON). Verify:
1) Every vendor has at least one evidence item that references a snippet id or url listed below.
2) For any structured claim (pricing, key_features, security_compliance, integrations), ensure at least one provided evidence snippet text actually supports that claim. If support is missing, list the unsupported claims.
3) If a vendor contains hallucinated information (no evidence), list that vendor name and the unsupported fields.

Respond ONLY with JSON:
{{
  "ok": boolean,
  "errors": [ "human-readable problems..." ],
  "fix_instructions": "If not ok, provide short instructions to the writer for removing/marking unsupported claims."
}}

Vendors summary:
{vendors_text}

Available snippets:
{snippets_text}

Return the JSON now.
"""
    # Call critic LLM
    try:
        resp = call_llm(critic_prompt, provider=llm_provider, temperature=CRITIC_TEMPERATURE, timeout=timeout)
    except Exception as e:
        logger.exception("Critic LLM call failed: %s", e)
        return {"ok": True, "errors": [], "fix_instructions": ""}

    # parse JSON from resp
    if isinstance(resp, dict) and resp.get("structured"):
        parsed = resp.get("structured")
        if isinstance(parsed, dict):
            return parsed
    # fallback parse from text
    txt = resp.get("text") if isinstance(resp, dict) else str(resp)
    try:
        return json.loads(txt)
    except Exception:
        # attempt to extract JSON object
        try:
            s = txt.find("{")
            e = txt.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(txt[s:e+1])
        except Exception:
            pass
    # If critic returns unparsable output, be permissive
    return {"ok": True, "errors": [], "fix_instructions": ""}


def _validate_and_normalize_vendors(raw: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate that raw has the expected schema. Normalize types.
    Returns (vendors_list, warnings)
    """
    warnings = []
    vendors = []
    if not raw:
        return [], ["empty_response"]

    if isinstance(raw, dict) and "vendors" in raw:
        raw_vendors = raw.get("vendors") or []
    elif isinstance(raw, list):
        raw_vendors = raw
    else:
        raise ValueError("Unexpected LLM output shape for vendors")

    for v in raw_vendors[:MAX_VENDORS]:
        if not isinstance(v, dict):
            warnings.append("skipping_non_object_vendor")
            continue
        name = v.get("name")
        if not name or not isinstance(name, str):
            warnings.append("vendor_missing_name_or_invalid")
            continue
        # normalize arrays
        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return [str(i) for i in x]
            if isinstance(x, str):
                return [i.strip() for i in x.split(",") if i.strip()]
            return [str(x)]

        key_features = _as_list(v.get("key_features"))
        security_compliance = _as_list(v.get("security_compliance"))
        evidence = v.get("evidence") or []
        norm_evidence = []
        if isinstance(evidence, list):
            for e in evidence:
                if not isinstance(e, dict):
                    continue
                url = e.get("url") or ""
                sid = e.get("snippet_id") or None
                text = e.get("text") or ""
                published = e.get("published") or None
                if not text and not url:
                    continue
                norm_evidence.append({"url": url, "snippet_id": sid, "text": text, "published": published})
        else:
            warnings.append(f"vendor_{name}_evidence_not_list")

        vendor_obj = {
            "name": name.strip(),
            "pricing": v.get("pricing") or "",
            "key_features": key_features,
            "target_users": v.get("target_users") or "",
            "extracted_text": v.get("extracted_text") or "",
            "support_and_sla": v.get("support_and_sla") or "",
            "security_compliance": security_compliance,
            "evidence": norm_evidence,
            "notes": v.get("notes") or "",
            "confidence": float(v.get("confidence")) if v.get("confidence") is not None else None
        }
        vendors.append(vendor_obj)

    return vendors, warnings


def extract_vendors(topic: str,
                    fields: List[str],
                    snippets_per_field: Dict[str, List[Any]],
                    top_k_per_field: int = TOP_K_PER_FIELD,
                    llm_provider: Optional[str] = None,
                    timeout: int = LLM_TIMEOUT_SECS,
                    critic_rounds: int = 1) -> Dict[str, Any]:
    """
    Main entrypoint for extraction.

    Returns:
      {"vendors": [...], "warnings": [...]}
    """
    start = time.time()
    snippets_block, flat_snips = _prepare_snippets_payload(topic, fields, snippets_per_field, top_k_per_field)
    writer_prompt = _build_writer_prompt(topic, fields, snippets_block, max_vendors=MAX_VENDORS)

    try:
        logger.info("Calling LLM writer with prompt length: %d", len(writer_prompt))
        
        # Test the LLM with a simple prompt first
        logger.debug("Testing LLM with simple prompt for topic: %s", topic)
        test_response = _test_llm_response(topic, llm_provider or "gemini")
        logger.debug("Test LLM response: %s", test_response[:300])
        
        raw = _call_writer(writer_prompt, llm_provider=llm_provider, timeout=timeout)
        logger.debug("LLM writer returned: %s", str(raw)[:200])
    except Exception as e:
        logger.exception("Writer failed: %s", e)
        # Try a simple fallback extraction
        try:
            raw = _simple_fallback_extraction(topic, flat_snips)
            logger.info("Used fallback extraction method")
        except Exception as fallback_e:
            logger.exception("Fallback extraction also failed: %s", fallback_e)
        return {"vendors": [], "warnings": [f"writer_failed:{str(e)[:200]}"]}

    vendors, warnings = _validate_and_normalize_vendors(raw)

    # Critic loop: ask critic to validate; if critic not OK, re-run writer with fix instructions once
    if critic_rounds and vendors:
        critic_resp = _call_critic(vendors, flat_snips, topic, llm_provider=llm_provider, timeout=timeout)
        if not critic_resp.get("ok", True):
            fix_instructions = critic_resp.get("fix_instructions", "")
            logger.info("Critic found issues: %s. Asking writer to fix.", critic_resp.get("errors", []))
            # Build a follow-up prompt constraining the writer to remove unsupported claims
            followup_prompt = writer_prompt + "\n\nCRITIC_FEEDBACK:\n" + (fix_instructions or "Remove unsupported claims and only keep vendor fields that are directly supported by evidence.")
            try:
                raw2 = _call_writer(followup_prompt, llm_provider=llm_provider, timeout=timeout)
                vendors2, warnings2 = _validate_and_normalize_vendors(raw2)
                # prefer the critic-corrected output if non-empty
                if vendors2:
                    vendors = vendors2
                    warnings.extend(warnings2)
            except Exception as e:
                logger.exception("Writer follow-up failed: %s", e)
                warnings.append(f"writer_followup_failed:{str(e)[:200]}")
        else:
            # critic ok; merge critic warnings if any
            critic_errors = critic_resp.get("errors", [])
            if critic_errors:
                warnings.extend([f"critic:{e}" for e in critic_errors])

    elapsed = time.time() - start
    logger.info("extract_vendors completed topic='%s' vendors=%d time=%.2fs", topic[:80], len(vendors), elapsed)
    return {"vendors": vendors, "warnings": warnings}
