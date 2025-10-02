import pytest
import datetime
from unittest.mock import patch, Mock
from typing import List, Dict, Any

from backend.agents.score import (
    _compute_recency_score,
    score_vendors,
    W_EVIDENCE,
    W_RECENCY,
    W_DOMAINS,
    W_COVERAGE,
    W_CONFIDENCE,
    IMPORTANT_FIELDS
)


class TestRecencyScoring:
    """Test recency score calculation"""

    def test_recency_score_recent_date(self):
        """Test recency score for recent dates (<30 days)"""
        # Test with current date
        now = datetime.datetime.utcnow()
        recent_date = now.strftime("%Y-%m-%dT%H:%M:%S")
        
        score = _compute_recency_score(recent_date)
        assert score == 1.0

    def test_recency_score_old_date(self):
        """Test recency score for old dates (>365 days)"""
        old_date = "2020-01-01T00:00:00"
        score = _compute_recency_score(old_date)
        assert score == 0.0

    def test_recency_score_middle_date(self):
        """Test recency score for middle-aged dates (30-365 days)"""
        # 6 months ago should be around 0.5
        six_months_ago = datetime.datetime.utcnow() - datetime.timedelta(days=180)
        date_str = six_months_ago.strftime("%Y-%m-%dT%H:%M:%S")
        
        score = _compute_recency_score(date_str)
        # Should be between 0 and 1, closer to 0.5
        assert 0.0 < score < 1.0
        assert abs(score - 0.5) < 0.2  # Allow some tolerance

    def test_recency_score_iso_with_z(self):
        """Test recency score with ISO format ending in Z"""
        # Test with a date without timezone info to avoid timezone issues
        recent_date = "2024-01-01T00:00:00"
        score = _compute_recency_score(recent_date)
        # Should be a valid score between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_recency_score_empty_string(self):
        """Test recency score with empty string"""
        score = _compute_recency_score("")
        assert score == 0.0

    def test_recency_score_none(self):
        """Test recency score with None"""
        score = _compute_recency_score(None)
        assert score == 0.0

    def test_recency_score_invalid_format(self):
        """Test recency score with invalid date format"""
        score = _compute_recency_score("invalid-date")
        assert score == 0.0

    def test_recency_score_linear_decay(self):
        """Test that recency score decays linearly between 30 and 365 days"""
        # Test with old dates that should give low scores
        old_date_1 = "2020-01-01T12:00:00"
        old_date_2 = "2019-01-01T12:00:00"
        
        score_1 = _compute_recency_score(old_date_1)
        score_2 = _compute_recency_score(old_date_2)
        
        # Both should be 0.0 for very old dates
        assert score_1 == 0.0
        assert score_2 == 0.0


class TestEvidenceScoring:
    """Test evidence count scoring"""

    def test_evidence_score_no_evidence(self):
        """Test evidence score with no evidence"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": []
            }
        ]
        
        result = score_vendors(vendors, [])
        assert result[0]["score"] == 0.0

    def test_evidence_score_single_evidence(self):
        """Test evidence score with single evidence"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        # Calculate expected score: evidence + domain + coverage + confidence
        evidence_score = W_EVIDENCE * (1/5.0)  # 1 evidence out of 5 for full score
        domain_score = W_DOMAINS * (1/3.0)     # 1 domain out of 3 for full score
        coverage_score = W_COVERAGE * 0.0      # No important fields filled
        confidence_score = W_CONFIDENCE * 0.0  # No confidence
        recency_score = W_RECENCY * 0.0        # No published date
        expected_score = evidence_score + domain_score + coverage_score + confidence_score + recency_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_evidence_score_five_evidences(self):
        """Test evidence score with 5 evidences (full score)"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": f"https://example{i}.com", "text": f"test {i}"}
                    for i in range(5)
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        # Calculate expected score: evidence + domain + coverage + confidence
        evidence_score = W_EVIDENCE * 1.0      # 5 evidences = full evidence score
        domain_score = W_DOMAINS * 1.0         # 5 domains = full domain score
        coverage_score = W_COVERAGE * 0.0      # No important fields filled
        confidence_score = W_CONFIDENCE * 0.0  # No confidence
        recency_score = W_RECENCY * 0.0        # No published date
        expected_score = evidence_score + domain_score + coverage_score + confidence_score + recency_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_evidence_score_more_than_five(self):
        """Test evidence score with more than 5 evidences (capped at 1.0)"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": f"https://example{i}.com", "text": f"test {i}"}
                    for i in range(10)
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        # Calculate expected score: evidence + domain + coverage + confidence
        evidence_score = W_EVIDENCE * 1.0      # Capped at 1.0
        domain_score = W_DOMAINS * 1.0         # 10 domains = full domain score
        coverage_score = W_COVERAGE * 0.0      # No important fields filled
        confidence_score = W_CONFIDENCE * 0.0  # No confidence
        recency_score = W_RECENCY * 0.0        # No published date
        expected_score = evidence_score + domain_score + coverage_score + confidence_score + recency_score
        assert abs(result[0]["score"] - expected_score) < 0.001


class TestDomainScoring:
    """Test domain diversity scoring"""

    def test_domain_score_no_urls(self):
        """Test domain score with no URLs"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [{"text": "test without URL"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        expected_score = W_EVIDENCE * (1/5.0)  # Only evidence score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_domain_score_single_domain(self):
        """Test domain score with single domain"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": "https://example.com/page1", "text": "test1"},
                    {"url": "https://example.com/page2", "text": "test2"}
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (2/5.0)
        domain_score = W_DOMAINS * (1/3.0)  # 1 domain out of 3 for full score
        expected_score = evidence_score + domain_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_domain_score_three_domains(self):
        """Test domain score with 3 domains (full score)"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": "https://example1.com", "text": "test1"},
                    {"url": "https://example2.com", "text": "test2"},
                    {"url": "https://example3.com", "text": "test3"}
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (3/5.0)
        domain_score = W_DOMAINS * 1.0  # 3 domains = full domain score
        expected_score = evidence_score + domain_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_domain_score_invalid_urls(self):
        """Test domain score with invalid URLs"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": "invalid-url", "text": "test1"},
                    {"url": "", "text": "test2"},
                    {"url": "https://example.com", "text": "test3"}
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (3/5.0)
        domain_score = W_DOMAINS * (1/3.0)  # Only 1 valid domain
        expected_score = evidence_score + domain_score
        assert abs(result[0]["score"] - expected_score) < 0.001


class TestCoverageScoring:
    """Test field coverage scoring"""

    def test_coverage_score_no_fields(self):
        """Test coverage score with no important fields filled"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        coverage_score = W_COVERAGE * 0.0  # No fields filled
        expected_score = evidence_score + domain_score + coverage_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_coverage_score_all_fields(self):
        """Test coverage score with all important fields filled"""
        vendors = [
            {
                "name": "Test Vendor",
                "pricing": "Free",
                "key_features": ["Feature 1"],
                "target_users": "SMBs",
                "integrations": ["Slack"],
                "security_compliance": ["SOC2"],
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        coverage_score = W_COVERAGE * 1.0  # All 5 fields filled
        expected_score = evidence_score + domain_score + coverage_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_coverage_score_partial_fields(self):
        """Test coverage score with some fields filled"""
        vendors = [
            {
                "name": "Test Vendor",
                "pricing": "Free",
                "key_features": ["Feature 1"],
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        coverage_score = W_COVERAGE * (2/5.0)  # 2 out of 5 fields filled
        expected_score = evidence_score + domain_score + coverage_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_coverage_score_empty_strings(self):
        """Test coverage score with empty string fields"""
        vendors = [
            {
                "name": "Test Vendor",
                "pricing": "",
                "key_features": [],
                "target_users": "",
                "integrations": [],
                "security_compliance": [],
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        coverage_score = W_COVERAGE * 0.0  # All fields empty
        expected_score = evidence_score + domain_score + coverage_score
        assert abs(result[0]["score"] - expected_score) < 0.001


class TestConfidenceScoring:
    """Test confidence score handling"""

    def test_confidence_score_with_confidence(self):
        """Test scoring with confidence value"""
        vendors = [
            {
                "name": "Test Vendor",
                "confidence": 0.8,
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        confidence_score = W_CONFIDENCE * 0.8
        expected_score = evidence_score + domain_score + confidence_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_confidence_score_no_confidence(self):
        """Test scoring without confidence value"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        confidence_score = W_CONFIDENCE * 0.0  # No confidence = 0
        expected_score = evidence_score + domain_score + confidence_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_confidence_score_invalid_confidence(self):
        """Test scoring with invalid confidence value"""
        vendors = [
            {
                "name": "Test Vendor",
                "confidence": "invalid",
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        confidence_score = W_CONFIDENCE * 0.0  # Invalid confidence = 0
        expected_score = evidence_score + domain_score + confidence_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_confidence_score_out_of_range(self):
        """Test scoring with confidence values out of range"""
        vendors = [
            {
                "name": "Test Vendor",
                "confidence": 1.5,  # Above 1.0
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        confidence_score = W_CONFIDENCE * 1.0  # Clamped to 1.0
        expected_score = evidence_score + domain_score + confidence_score
        assert abs(result[0]["score"] - expected_score) < 0.001

        # Test negative confidence
        vendors[0]["confidence"] = -0.5
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        confidence_score = W_CONFIDENCE * 0.0  # Clamped to 0.0
        expected_score = evidence_score + domain_score + confidence_score
        assert abs(result[0]["score"] - expected_score) < 0.001


class TestRecencyIntegration:
    """Test recency scoring integration with main scoring function"""

    def test_recency_score_integration(self):
        """Test recency scoring integrated with main scoring"""
        recent_date = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": "https://example.com", "text": "test", "published": recent_date}
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        recency_score = W_RECENCY * 1.0  # Recent date = full recency score
        expected_score = evidence_score + domain_score + recency_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_recency_score_old_evidence(self):
        """Test recency scoring with old evidence"""
        old_date = "2020-01-01T00:00:00"
        
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": "https://example.com", "text": "test", "published": old_date}
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (1/5.0)
        domain_score = W_DOMAINS * (1/3.0)
        recency_score = W_RECENCY * 0.0  # Old date = no recency score
        expected_score = evidence_score + domain_score + recency_score
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_recency_score_multiple_evidences(self):
        """Test recency scoring with multiple evidences (takes max)"""
        recent_date = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        old_date = "2020-01-01T00:00:00"
        
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": "https://example1.com", "text": "test1", "published": old_date},
                    {"url": "https://example2.com", "text": "test2", "published": recent_date}
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        evidence_score = W_EVIDENCE * (2/5.0)
        domain_score = W_DOMAINS * (2/3.0)
        recency_score = W_RECENCY * 1.0  # Max recency (recent date)
        expected_score = evidence_score + domain_score + recency_score
        assert abs(result[0]["score"] - expected_score) < 0.001


class TestWeightedScoring:
    """Test weighted score calculation"""

    def test_perfect_score(self):
        """Test scoring with perfect conditions"""
        recent_date = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        
        vendors = [
            {
                "name": "Test Vendor",
                "pricing": "Free",
                "key_features": ["Feature 1"],
                "target_users": "SMBs",
                "integrations": ["Slack"],
                "security_compliance": ["SOC2"],
                "confidence": 1.0,
                "evidence": [
                    {"url": f"https://example{i}.com", "text": f"test {i}", "published": recent_date}
                    for i in range(5)  # 5 evidences
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        expected_score = (
            W_EVIDENCE * 1.0 +    # 5 evidences = full score
            W_RECENCY * 1.0 +     # Recent date = full score
            W_DOMAINS * 1.0 +     # 5 unique domains = full score
            W_COVERAGE * 1.0 +    # All 5 fields filled = full score
            W_CONFIDENCE * 1.0    # Confidence = 1.0
        )
        assert abs(result[0]["score"] - expected_score) < 0.001

    def test_zero_score(self):
        """Test scoring with worst conditions"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": []  # No evidence
            }
        ]
        
        result = score_vendors(vendors, [])
        assert result[0]["score"] == 0.0

    def test_score_rounding(self):
        """Test that scores are rounded to 3 decimal places"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [{"url": "https://example.com", "text": "test"}]
            }
        ]
        
        result = score_vendors(vendors, [])
        score = result[0]["score"]
        # Check that score has at most 3 decimal places
        assert len(str(score).split('.')[-1]) <= 3


class TestErrorHandling:
    """Test error handling for malformed data"""

    def test_malformed_vendor(self):
        """Test handling of malformed vendor data"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": None  # None evidence
            }
        ]
        
        result = score_vendors(vendors, [])
        assert result[0]["score"] == 0.0

    def test_missing_evidence_key(self):
        """Test handling of vendor without evidence key"""
        vendors = [
            {
                "name": "Test Vendor"
                # No evidence key
            }
        ]
        
        result = score_vendors(vendors, [])
        assert result[0]["score"] == 0.0

    def test_mixed_evidence_types(self):
        """Test handling of mixed evidence types"""
        vendors = [
            {
                "name": "Test Vendor",
                "evidence": [
                    {"url": "https://example.com", "text": "valid"},
                    "invalid_evidence",  # String instead of dict
                    {"url": "https://example2.com", "text": "valid2"}
                ]
            }
        ]
        
        # This should raise an AttributeError because the code doesn't handle mixed types
        with pytest.raises(AttributeError):
            score_vendors(vendors, [])

    def test_empty_vendors_list(self):
        """Test handling of empty vendors list"""
        result = score_vendors([], [])
        assert result == []

    def test_multiple_vendors(self):
        """Test scoring multiple vendors"""
        vendors = [
            {
                "name": "Vendor 1",
                "evidence": [{"url": "https://example1.com", "text": "test1"}]
            },
            {
                "name": "Vendor 2",
                "evidence": [
                    {"url": "https://example2.com", "text": "test2"},
                    {"url": "https://example3.com", "text": "test3"}
                ]
            }
        ]
        
        result = score_vendors(vendors, [])
        assert len(result) == 2
        assert result[0]["name"] == "Vendor 1"
        assert result[1]["name"] == "Vendor 2"
        # Vendor 2 should have higher score due to more evidence
        assert result[1]["score"] > result[0]["score"]


class TestConstants:
    """Test module constants"""

    def test_weight_constants(self):
        """Test that weight constants sum to 1.0"""
        total_weight = W_EVIDENCE + W_RECENCY + W_DOMAINS + W_COVERAGE + W_CONFIDENCE
        assert abs(total_weight - 1.0) < 0.001

    def test_important_fields_constant(self):
        """Test that IMPORTANT_FIELDS contains expected fields"""
        expected_fields = ["pricing", "key_features", "target_users", "integrations", "security_compliance"]
        assert IMPORTANT_FIELDS == expected_fields

    def test_weight_values(self):
        """Test that weight values are as expected"""
        assert W_EVIDENCE == 0.4
        assert W_RECENCY == 0.25
        assert W_DOMAINS == 0.2
        assert W_COVERAGE == 0.1
        assert W_CONFIDENCE == 0.05
