import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from backend.agents.extract import (
    _normalize_snippet_input,
    _prepare_snippets_payload,
    _build_writer_prompt,
    _call_writer,
    _call_critic,
    _validate_and_normalize_vendors,
    extract_vendors,
    SNIPPET_MAX_CHARS,
    MAX_SNIPPETS_TOTAL,
    TOP_K_PER_FIELD,
    MAX_VENDORS,
    WRITER_TEMPERATURE,
    CRITIC_TEMPERATURE,
    LLM_TIMEOUT_SECS,
    _EXTRACT_SCHEMA
)


class TestSnippetProcessing:
    """Test snippet input processing and normalization"""

    def test_normalize_snippet_input_strings(self):
        """Test normalizing string snippets"""
        snippets = ["First snippet", "Second snippet", "Third snippet"]
        result = _normalize_snippet_input(snippets)
        
        assert len(result) == 3
        assert result[0]["id"] == "snip_0"
        assert result[0]["url"] == ""
        assert result[0]["text"] == "First snippet"
        assert result[1]["id"] == "snip_1"
        assert result[2]["id"] == "snip_2"

    def test_normalize_snippet_input_dicts(self):
        """Test normalizing dict snippets"""
        snippets = [
            {
                "id": "doc1",
                "url": "https://example.com",
                "text": "Document content",
                "title": "Example Doc",
                "published": "2024-01-01"
            },
            {
                "url": "https://example2.com",
                "snippet": "Another snippet",
                "title": "Doc 2"
            }
        ]
        result = _normalize_snippet_input(snippets)
        
        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[0]["url"] == "https://example.com"
        assert result[0]["text"] == "Document content"
        assert result[0]["title"] == "Example Doc"
        assert result[0]["published"] == "2024-01-01"
        
        assert result[1]["id"] == "https://example2.com"
        assert result[1]["url"] == "https://example2.com"
        assert result[1]["text"] == "Another snippet"
        assert result[1]["title"] == "Doc 2"

    def test_normalize_snippet_input_mixed(self):
        """Test normalizing mixed string and dict snippets"""
        snippets = [
            "String snippet",
            {"id": "doc1", "text": "Dict snippet", "url": "https://example.com"},
            "Another string"
        ]
        result = _normalize_snippet_input(snippets)
        
        assert len(result) == 3
        assert result[0]["id"] == "snip_0"
        assert result[0]["text"] == "String snippet"
        assert result[1]["id"] == "doc1"
        assert result[1]["text"] == "Dict snippet"
        assert result[2]["id"] == "snip_2"

    def test_normalize_snippet_input_empty_and_none(self):
        """Test handling of empty and None snippets"""
        snippets = ["Valid snippet", "", None, "Another valid"]
        result = _normalize_snippet_input(snippets)
        
        assert len(result) == 2
        assert result[0]["text"] == "Valid snippet"
        assert result[1]["text"] == "Another valid"

    def test_normalize_snippet_input_text_truncation(self):
        """Test text truncation for long snippets"""
        long_text = "x" * (SNIPPET_MAX_CHARS + 100)
        snippets = [long_text]
        result = _normalize_snippet_input(snippets)
        
        assert len(result) == 1
        assert len(result[0]["text"]) == SNIPPET_MAX_CHARS  # Truncated to exact limit
        assert result[0]["text"] == "x" * SNIPPET_MAX_CHARS

    def test_normalize_snippet_input_other_types(self):
        """Test normalizing other types (coerced to string)"""
        snippets = [123, {"text": "Dict snippet"}, 45.6]
        result = _normalize_snippet_input(snippets)
        
        assert len(result) == 3
        assert result[0]["text"] == "123"
        assert result[1]["text"] == "Dict snippet"
        assert result[2]["text"] == "45.6"

    def test_prepare_snippets_payload_basic(self):
        """Test basic snippet payload preparation"""
        topic = "AI Tools"
        fields = ["pricing", "features"]
        snippets_per_field = {
            "pricing": ["Free tier available", "Paid plans start at $10"],
            "features": ["Machine learning", "API access"]
        }
        
        snippets_text, flat_snippets = _prepare_snippets_payload(
            topic, fields, snippets_per_field, top_k_per_field=2
        )
        
        assert "Topic: AI Tools" in snippets_text
        assert "Evidence snippets" in snippets_text
        assert len(flat_snippets) == 4
        
        # Check that snippets are properly formatted
        assert any("snip_0" in line for line in snippets_text.split("\n"))
        assert any("pricing" in line for line in snippets_text.split("\n"))
        assert any("features" in line for line in snippets_text.split("\n"))

    def test_prepare_snippets_payload_max_limit(self):
        """Test that snippet count respects MAX_SNIPPETS_TOTAL limit"""
        topic = "Test"
        fields = ["field1", "field2"]
        # Create more snippets than the limit
        snippets_per_field = {
            "field1": [f"snippet_{i}" for i in range(MAX_SNIPPETS_TOTAL + 10)],
            "field2": [f"snippet_{i}" for i in range(MAX_SNIPPETS_TOTAL + 10)]
        }
        
        snippets_text, flat_snippets = _prepare_snippets_payload(
            topic, fields, snippets_per_field, top_k_per_field=MAX_SNIPPETS_TOTAL
        )
        
        assert len(flat_snippets) <= MAX_SNIPPETS_TOTAL

    def test_prepare_snippets_payload_top_k_limit(self):
        """Test that top_k_per_field limit is respected"""
        topic = "Test"
        fields = ["field1"]
        snippets_per_field = {
            "field1": [f"snippet_{i}" for i in range(10)]
        }
        
        snippets_text, flat_snippets = _prepare_snippets_payload(
            topic, fields, snippets_per_field, top_k_per_field=3
        )
        
        # Should only have 3 snippets from field1
        field1_snippets = [s for s in flat_snippets if s["field"] == "field1"]
        assert len(field1_snippets) == 3


class TestPromptBuilding:
    """Test prompt building functions"""

    def test_build_writer_prompt_basic(self):
        """Test basic writer prompt building"""
        topic = "AI Tools"
        fields = ["pricing", "features"]
        snippets_block = "snippet_0 | pricing | https://example.com | Free tier available"
        
        prompt = _build_writer_prompt(topic, fields, snippets_block, max_vendors=5)
        
        assert "extract up to 5 vendor entries" in prompt
        assert "snippet_0 | pricing | https://example.com | Free tier available" in prompt
        assert "Return ONLY JSON" in prompt
        assert "evidence" in prompt

    def test_build_writer_prompt_max_vendors(self):
        """Test that max_vendors parameter is used correctly"""
        topic = "Test"
        fields = ["field1"]
        snippets_block = "test snippet"
        
        prompt = _build_writer_prompt(topic, fields, snippets_block, max_vendors=3)
        
        assert "extract up to 3 vendor entries" in prompt

    def test_build_writer_prompt_instructions(self):
        """Test that prompt contains proper instructions"""
        topic = "Test"
        fields = ["field1"]
        snippets_block = "test snippet"
        
        prompt = _build_writer_prompt(topic, fields, snippets_block)
        
        assert "Do NOT hallucinate sources" in prompt
        assert "Prefer conservative answers" in prompt
        assert "name" in prompt
        assert "evidence" in prompt
        assert "pricing" in prompt
        assert "key_features" in prompt


class TestLLMCalls:
    """Test LLM calling functions"""

    def test_call_writer_with_function_schema(self):
        """Test writer call with function schema"""
        prompt = "Test prompt"
        mock_response = {
            "structured": {
                "vendors": [
                    {
                        "name": "Test Vendor",
                        "evidence": [{"url": "https://example.com", "text": "test"}]
                    }
                ]
            }
        }
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            mock_call_llm.return_value = mock_response
            
            result = _call_writer(prompt, llm_provider="gemini", timeout=15)
            
            mock_call_llm.assert_called_once_with(
                prompt, 
                provider="gemini", 
                function_schema=_EXTRACT_SCHEMA, 
                temperature=WRITER_TEMPERATURE, 
                timeout=15
            )
            assert result == mock_response["structured"]

    def test_call_writer_without_function_schema(self):
        """Test writer call when function schema is not supported"""
        prompt = "Test prompt"
        mock_response = {
            "text": '{"vendors": [{"name": "Test Vendor", "evidence": [{"url": "https://example.com", "text": "test"}]}]}'
        }
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            # First call raises TypeError (function_schema not supported)
            mock_call_llm.side_effect = [TypeError("function_schema not supported"), mock_response]
            
            result = _call_writer(prompt, llm_provider="gemini", timeout=15)
            
            # Should be called twice - once with schema, once without
            assert mock_call_llm.call_count == 2
            assert result == {"vendors": [{"name": "Test Vendor", "evidence": [{"url": "https://example.com", "text": "test"}]}]}

    def test_call_writer_json_parsing_fallback(self):
        """Test writer call with JSON parsing fallback"""
        prompt = "Test prompt"
        mock_response = {
            "text": 'Some text before {"vendors": [{"name": "Test Vendor", "evidence": [{"url": "https://example.com", "text": "test"}]}]} some text after'
        }
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            mock_call_llm.return_value = mock_response
            
            result = _call_writer(prompt, llm_provider="gemini", timeout=15)
            
            assert result == {"vendors": [{"name": "Test Vendor", "evidence": [{"url": "https://example.com", "text": "test"}]}]}

    def test_call_writer_parse_error(self):
        """Test writer call with unparseable response"""
        prompt = "Test prompt"
        mock_response = {"text": "This is not JSON at all"}
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            mock_call_llm.return_value = mock_response
            
            with pytest.raises(ValueError, match="Writer LLM returned no parseable JSON"):
                _call_writer(prompt, llm_provider="gemini", timeout=15)

    def test_call_critic_success(self):
        """Test successful critic call"""
        vendors = [
            {
                "name": "Test Vendor",
                "pricing": "Free",
                "evidence": [{"snippet_id": "snip_0", "text": "test evidence"}]
            }
        ]
        snippets = [
            {"id": "snip_0", "field": "pricing", "url": "https://example.com", "text": "Free tier available"}
        ]
        
        mock_response = {
            "structured": {
                "ok": True,
                "errors": [],
                "fix_instructions": ""
            }
        }
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            mock_call_llm.return_value = mock_response
            
            result = _call_critic(vendors, snippets, "AI Tools", llm_provider="gemini", timeout=15)
            
            assert result["ok"] is True
            assert result["errors"] == []
            assert result["fix_instructions"] == ""

    def test_call_critic_with_errors(self):
        """Test critic call that finds errors"""
        vendors = [
            {
                "name": "Test Vendor",
                "pricing": "Free",
                "evidence": [{"snippet_id": "nonexistent", "text": "test evidence"}]
            }
        ]
        snippets = [
            {"id": "snip_0", "field": "pricing", "url": "https://example.com", "text": "Free tier available"}
        ]
        
        mock_response = {
            "structured": {
                "ok": False,
                "errors": ["Evidence snippet_id 'nonexistent' not found"],
                "fix_instructions": "Remove unsupported claims"
            }
        }
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            mock_call_llm.return_value = mock_response
            
            result = _call_critic(vendors, snippets, "AI Tools", llm_provider="gemini", timeout=15)
            
            assert result["ok"] is False
            assert "nonexistent" in result["errors"][0]
            assert "Remove unsupported claims" in result["fix_instructions"]

    def test_call_critic_fallback_parsing(self):
        """Test critic call with fallback JSON parsing"""
        vendors = [{"name": "Test Vendor", "evidence": []}]
        snippets = [{"id": "snip_0", "field": "pricing", "text": "test"}]
        
        mock_response = {
            "text": '{"ok": true, "errors": [], "fix_instructions": ""}'
        }
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            mock_call_llm.return_value = mock_response
            
            result = _call_critic(vendors, snippets, "AI Tools", llm_provider="gemini", timeout=15)
            
            assert result["ok"] is True

    def test_call_critic_exception_handling(self):
        """Test critic call with exception handling"""
        vendors = [{"name": "Test Vendor", "evidence": []}]
        snippets = [{"id": "snip_0", "field": "pricing", "text": "test"}]
        
        with patch('backend.agents.extract.call_llm') as mock_call_llm:
            mock_call_llm.side_effect = Exception("LLM call failed")
            
            result = _call_critic(vendors, snippets, "AI Tools", llm_provider="gemini", timeout=15)
            
            # Should return permissive result on exception
            assert result["ok"] is True
            assert result["errors"] == []
            assert result["fix_instructions"] == ""


class TestValidation:
    """Test vendor validation and normalization"""

    def test_validate_and_normalize_vendors_basic(self):
        """Test basic vendor validation"""
        raw = {
            "vendors": [
                {
                    "name": "Test Vendor",
                    "pricing": "Free",
                    "key_features": ["Feature 1", "Feature 2"],
                    "evidence": [
                        {"url": "https://example.com", "text": "Evidence text"}
                    ]
                }
            ]
        }
        
        vendors, warnings = _validate_and_normalize_vendors(raw)
        
        assert len(vendors) == 1
        assert vendors[0]["name"] == "Test Vendor"
        assert vendors[0]["pricing"] == "Free"
        assert vendors[0]["key_features"] == ["Feature 1", "Feature 2"]
        assert len(vendors[0]["evidence"]) == 1
        assert warnings == []

    def test_validate_and_normalize_vendors_list_input(self):
        """Test validation with list input instead of dict"""
        raw = [
            {
                "name": "Test Vendor",
                "evidence": [{"url": "https://example.com", "text": "Evidence"}]
            }
        ]
        
        vendors, warnings = _validate_and_normalize_vendors(raw)
        
        assert len(vendors) == 1
        assert vendors[0]["name"] == "Test Vendor"

    def test_validate_and_normalize_vendors_empty_response(self):
        """Test validation with empty response"""
        vendors, warnings = _validate_and_normalize_vendors(None)
        
        assert vendors == []
        assert warnings == ["empty_response"]

    def test_validate_and_normalize_vendors_missing_name(self):
        """Test validation with vendor missing name"""
        raw = {
            "vendors": [
                {
                    "pricing": "Free",
                    "evidence": [{"url": "https://example.com", "text": "Evidence"}]
                }
            ]
        }
        
        vendors, warnings = _validate_and_normalize_vendors(raw)
        
        assert vendors == []
        assert "vendor_missing_name_or_invalid" in warnings

    def test_validate_and_normalize_vendors_array_normalization(self):
        """Test array field normalization"""
        raw = {
            "vendors": [
                {
                    "name": "Test Vendor",
                    "key_features": "Feature 1, Feature 2",  # String instead of array
                    "integrations": ["Integration 1"],  # Already array
                    "security_compliance": None,  # None value
                    "evidence": [{"url": "https://example.com", "text": "Evidence"}]
                }
            ]
        }
        
        vendors, warnings = _validate_and_normalize_vendors(raw)
        
        assert vendors[0]["key_features"] == ["Feature 1", "Feature 2"]
        assert vendors[0]["integrations"] == ["Integration 1"]
        assert vendors[0]["security_compliance"] == []

    def test_validate_and_normalize_vendors_evidence_validation(self):
        """Test evidence validation and normalization"""
        raw = {
            "vendors": [
                {
                    "name": "Test Vendor",
                    "evidence": [
                        {"url": "https://example.com", "text": "Valid evidence"},
                        {"snippet_id": "snip_0", "text": "Another valid"},
                        {"text": "No URL"},  # Should be kept (has text)
                        "Invalid evidence",  # Should be skipped
                        {"url": "https://example2.com", "text": "Valid", "published": "2024-01-01"}
                    ]
                }
            ]
        }
        
        vendors, warnings = _validate_and_normalize_vendors(raw)
        
        assert len(vendors[0]["evidence"]) == 4  # All valid evidence items (text-only is kept)
        assert vendors[0]["evidence"][0]["url"] == "https://example.com"
        assert vendors[0]["evidence"][1]["snippet_id"] == "snip_0"
        assert vendors[0]["evidence"][2]["text"] == "No URL"
        assert vendors[0]["evidence"][3]["published"] == "2024-01-01"

    def test_validate_and_normalize_vendors_max_limit(self):
        """Test that vendor count respects MAX_VENDORS limit"""
        raw = {
            "vendors": [
                {
                    "name": f"Vendor {i}",
                    "evidence": [{"url": f"https://example{i}.com", "text": f"Evidence {i}"}]
                }
                for i in range(MAX_VENDORS + 5)
            ]
        }
        
        vendors, warnings = _validate_and_normalize_vendors(raw)
        
        assert len(vendors) == MAX_VENDORS

    def test_validate_and_normalize_vendors_confidence_handling(self):
        """Test confidence score handling"""
        raw = {
            "vendors": [
                {
                    "name": "Test Vendor",
                    "confidence": 0.85,
                    "evidence": [{"url": "https://example.com", "text": "Evidence"}]
                },
                {
                    "name": "Test Vendor 2",
                    "confidence": None,
                    "evidence": [{"url": "https://example2.com", "text": "Evidence"}]
                }
            ]
        }
        
        vendors, warnings = _validate_and_normalize_vendors(raw)
        
        assert vendors[0]["confidence"] == 0.85
        assert vendors[1]["confidence"] is None


class TestMainExtractFunction:
    """Test the main extract_vendors function"""

    def test_extract_vendors_success(self):
        """Test successful vendor extraction"""
        topic = "AI Tools"
        fields = ["pricing", "features"]
        snippets_per_field = {
            "pricing": ["Free tier available", "Paid plans start at $10"],
            "features": ["Machine learning", "API access"]
        }
        
        mock_writer_response = {
            "vendors": [
                {
                    "name": "Test Vendor",
                    "pricing": "Free",
                    "key_features": ["ML", "API"],
                    "evidence": [{"url": "https://example.com", "text": "Free tier available"}]
                }
            ]
        }
        
        mock_critic_response = {"ok": True, "errors": [], "fix_instructions": ""}
        
        with patch('backend.agents.extract._prepare_snippets_payload') as mock_prepare, \
             patch('backend.agents.extract._call_writer') as mock_writer, \
             patch('backend.agents.extract._call_critic') as mock_critic, \
             patch('backend.agents.extract._validate_and_normalize_vendors') as mock_validate:
            
            mock_prepare.return_value = ("snippets_block", [])
            mock_writer.return_value = mock_writer_response
            mock_critic.return_value = mock_critic_response
            mock_validate.return_value = ([{"name": "Test Vendor", "pricing": "Free"}], [])
            
            result = extract_vendors(topic, fields, snippets_per_field)
            
            assert "vendors" in result
            assert "warnings" in result
            assert len(result["vendors"]) == 1
            assert result["vendors"][0]["name"] == "Test Vendor"

    def test_extract_vendors_writer_failure(self):
        """Test extraction when writer fails"""
        topic = "AI Tools"
        fields = ["pricing"]
        snippets_per_field = {"pricing": ["Free tier"]}
        
        with patch('backend.agents.extract._prepare_snippets_payload') as mock_prepare, \
             patch('backend.agents.extract._call_writer') as mock_writer:
            
            mock_prepare.return_value = ("snippets_block", [])
            mock_writer.side_effect = Exception("Writer failed")
            
            result = extract_vendors(topic, fields, snippets_per_field)
            
            assert result["vendors"] == []
            assert "writer_failed" in result["warnings"][0]

    def test_extract_vendors_critic_feedback(self):
        """Test extraction with critic feedback and correction"""
        topic = "AI Tools"
        fields = ["pricing"]
        snippets_per_field = {"pricing": ["Free tier"]}
        
        mock_writer_response = {
            "vendors": [
                {
                    "name": "Test Vendor",
                    "pricing": "Free",
                    "evidence": [{"url": "https://example.com", "text": "Free tier"}]
                }
            ]
        }
        
        mock_critic_response = {
            "ok": False,
            "errors": ["Unsupported claim"],
            "fix_instructions": "Remove unsupported claims"
        }
        
        mock_corrected_response = {
            "vendors": [
                {
                    "name": "Test Vendor",
                    "evidence": [{"url": "https://example.com", "text": "Free tier"}]
                }
            ]
        }
        
        with patch('backend.agents.extract._prepare_snippets_payload') as mock_prepare, \
             patch('backend.agents.extract._call_writer') as mock_writer, \
             patch('backend.agents.extract._call_critic') as mock_critic, \
             patch('backend.agents.extract._validate_and_normalize_vendors') as mock_validate:
            
            mock_prepare.return_value = ("snippets_block", [])
            mock_writer.side_effect = [mock_writer_response, mock_corrected_response]
            mock_critic.return_value = mock_critic_response
            mock_validate.side_effect = [
                ([{"name": "Test Vendor", "pricing": "Free"}], []),
                ([{"name": "Test Vendor"}], [])
            ]
            
            result = extract_vendors(topic, fields, snippets_per_field, critic_rounds=1)
            
            # Should call writer twice (original + correction)
            assert mock_writer.call_count == 2
            assert len(result["vendors"]) == 1
            assert result["vendors"][0]["name"] == "Test Vendor"

    def test_extract_vendors_critic_followup_failure(self):
        """Test extraction when critic followup fails"""
        topic = "AI Tools"
        fields = ["pricing"]
        snippets_per_field = {"pricing": ["Free tier"]}
        
        mock_writer_response = {
            "vendors": [{"name": "Test Vendor", "evidence": []}]
        }
        
        mock_critic_response = {
            "ok": False,
            "errors": ["Unsupported claim"],
            "fix_instructions": "Remove unsupported claims"
        }
        
        with patch('backend.agents.extract._prepare_snippets_payload') as mock_prepare, \
             patch('backend.agents.extract._call_writer') as mock_writer, \
             patch('backend.agents.extract._call_critic') as mock_critic, \
             patch('backend.agents.extract._validate_and_normalize_vendors') as mock_validate:
            
            mock_prepare.return_value = ("snippets_block", [])
            mock_writer.side_effect = [mock_writer_response, Exception("Followup failed")]
            mock_critic.return_value = mock_critic_response
            mock_validate.return_value = ([{"name": "Test Vendor"}], [])
            
            result = extract_vendors(topic, fields, snippets_per_field, critic_rounds=1)
            
            assert "writer_followup_failed" in result["warnings"][0]

    def test_extract_vendors_no_critic_rounds(self):
        """Test extraction with critic_rounds=0"""
        topic = "AI Tools"
        fields = ["pricing"]
        snippets_per_field = {"pricing": ["Free tier"]}
        
        mock_writer_response = {
            "vendors": [{"name": "Test Vendor", "evidence": []}]
        }
        
        with patch('backend.agents.extract._prepare_snippets_payload') as mock_prepare, \
             patch('backend.agents.extract._call_writer') as mock_writer, \
             patch('backend.agents.extract._call_critic') as mock_critic, \
             patch('backend.agents.extract._validate_and_normalize_vendors') as mock_validate:
            
            mock_prepare.return_value = ("snippets_block", [])
            mock_writer.return_value = mock_writer_response
            mock_validate.return_value = ([{"name": "Test Vendor"}], [])
            
            result = extract_vendors(topic, fields, snippets_per_field, critic_rounds=0)
            
            # Critic should not be called
            mock_critic.assert_not_called()
            assert len(result["vendors"]) == 1

    def test_extract_vendors_custom_parameters(self):
        """Test extraction with custom parameters"""
        topic = "AI Tools"
        fields = ["pricing"]
        snippets_per_field = {"pricing": ["Free tier"]}
        
        with patch('backend.agents.extract._prepare_snippets_payload') as mock_prepare, \
             patch('backend.agents.extract._call_writer') as mock_writer, \
             patch('backend.agents.extract._validate_and_normalize_vendors') as mock_validate:
            
            mock_prepare.return_value = ("snippets_block", [])
            mock_writer.return_value = {"vendors": []}
            mock_validate.return_value = ([], [])
            
            extract_vendors(
                topic, 
                fields, 
                snippets_per_field, 
                top_k_per_field=3,
                llm_provider="openai",
                timeout=30,
                critic_rounds=2
            )
            
            # Verify custom parameters were passed
            mock_prepare.assert_called_once_with(topic, fields, snippets_per_field, 3)
            mock_writer.assert_called_once()
            call_args = mock_writer.call_args
            assert call_args[1]["llm_provider"] == "openai"
            assert call_args[1]["timeout"] == 30


class TestConstants:
    """Test module constants"""
    
    def test_constants_values(self):
        """Test that constants have expected values"""
        assert SNIPPET_MAX_CHARS == 800
        assert MAX_SNIPPETS_TOTAL == 60
        assert TOP_K_PER_FIELD == 6
        assert MAX_VENDORS == 12
        assert WRITER_TEMPERATURE == 0.0
        assert CRITIC_TEMPERATURE == 0.0
        assert LLM_TIMEOUT_SECS == 15

    def test_extract_schema_structure(self):
        """Test that _EXTRACT_SCHEMA has expected structure"""
        schema = _EXTRACT_SCHEMA
        assert schema["name"] == "extract_vendors_schema"
        assert "parameters" in schema
        assert "properties" in schema["parameters"]
        assert "vendors" in schema["parameters"]["properties"]
        assert "warnings" in schema["parameters"]["properties"]
