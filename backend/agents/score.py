# backend/agents/score.py
"""
Vendor scoring module.

Purpose:
Take vendor objects (from extract.py) + sources and assign a normalized score [0..1]
based on evidence count, recency, domain trust, and coverage of important fields.
"""

from __future__ import annotations
import datetime
import logging
from typing import List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tunable weights - adjusted for better balance
W_EVIDENCE = 0.30      # Evidence count is important but not everything
W_RECENCY = 0.15       # Recency matters but shouldn't dominate
W_DOMAINS = 0.25       # Domain diversity is crucial for reliability
W_COVERAGE = 0.15      # Field coverage shows completeness
W_TEXT_QUALITY = 0.10  # Quality/length of extracted text
W_CONFIDENCE = 0.05    # Confidence is a nice-to-have

IMPORTANT_FIELDS = ["pricing", "key_features", "target_users", "extracted_text", "security_compliance"]


def _compute_recency_score(published: str) -> float:
    """
    Convert an ISO date string to a recency score [0..1].
    - 1.0 for <30 days old
    - linearly decay to 0.0 at 365 days
    """
    if not published:
        return 0.0
    try:
        dt = datetime.datetime.fromisoformat(published.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    delta_days = (datetime.datetime.utcnow() - dt).days
    if delta_days <= 30:
        return 1.0
    if delta_days >= 365:
        return 0.0
    return max(0.0, 1.0 - (delta_days - 30) / (365 - 30))


def _compute_text_quality_score(extracted_text: str) -> float:
    """
    Score the quality of extracted text based on length and content.
    - 1.0 for >=200 chars of meaningful text
    - 0.5 for 100-199 chars
    - 0.0 for <100 chars or generic fallback text
    """
    if not extracted_text:
        return 0.0
    
    # Penalize generic fallback messages
    if "Please visit the website" in extracted_text:
        return 0.1
    
    length = len(extracted_text.strip())
    
    if length >= 200:
        return 1.0
    elif length >= 100:
        return 0.5 + (length - 100) / 200  # Linear scale from 0.5 to 1.0
    else:
        return max(0.0, length / 100)  # Linear scale from 0.0 to 0.5


def score_vendors(vendors: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich vendors with a "score" field (0..100).

    Inputs:
    - vendors: list of vendor dicts (from extract.py)
    - sources: list of raw source dicts (from scout.py)

    Returns: updated vendors
    """
    now = datetime.datetime.utcnow()

    for v in vendors:
        evidences = v.get("evidence") or []
        evidence_count = len(evidences)

        # Evidence count normalized [0..1] (>=3 evidences = full score)
        score_evidence = min(1.0, evidence_count / 3.0)

        # Recency: take max recency across evidences
        recency_score = 0.0
        for e in evidences:
            recency_score = max(recency_score, _compute_recency_score(e.get("published", "")))

        # Domain diversity
        domains = set()
        for e in evidences:
            url = e.get("url") or ""
            try:
                dom = urlparse(url).netloc
                if dom:
                    domains.add(dom)
            except Exception:
                continue
        domain_score = min(1.0, len(domains) / 2.0)  # 2+ unique domains = full score

        # Field coverage: fraction of important fields filled
        coverage_fields = sum(1 for f in IMPORTANT_FIELDS if v.get(f))
        coverage_score = coverage_fields / len(IMPORTANT_FIELDS)

        # Text quality: score the extracted_text quality
        text_quality_score = _compute_text_quality_score(v.get("extracted_text", ""))

        # Confidence (optional, from extract.py)
        conf_score = v.get("confidence") or 0.0
        try:
            conf_score = float(conf_score)
        except Exception:
            conf_score = 0.0
        conf_score = max(0.0, min(conf_score, 1.0))

        # Weighted sum (0..1 scale)
        final_score_normalized = (
            W_EVIDENCE * score_evidence +
            W_RECENCY * recency_score +
            W_DOMAINS * domain_score +
            W_COVERAGE * coverage_score +
            W_TEXT_QUALITY * text_quality_score +
            W_CONFIDENCE * conf_score
        )
        
        # Convert to 0-100 scale
        final_score = round(final_score_normalized * 100, 1)
        v["score"] = final_score

    logger.info("Scored %d vendors", len(vendors))
    return vendors
