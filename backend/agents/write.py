# backend/agents/write.py
"""
backend/agents/write.py

Responsible for writing pipeline artifacts:
- matrix.csv
- brief.md
- vendors.json
- report.html (simple styled HTML)
- optionally report.pdf (if weasyprint is installed)

Primary API:
    write_reports(run_dir: str, topic: str, vendors: List[Dict], sources: List[Dict]) -> Dict[str,str]

Returns a dict mapping artifact names to filesystem paths.

Notes:
- Produces stable, human-friendly output suitable for demos and downloads.
- Keeps HTML/CSS minimal and offline-friendly.
- PDF export is optional and attempted only if 'weasyprint' is installed; failure doesn't break the pipeline.
"""

from __future__ import annotations
import os
import json
import csv
import logging
import html
from typing import List, Dict, Any, Optional

from backend.services.llm_client import call_llm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_recommendation(topic: str, vendors: List[Dict[str, Any]], llm_provider: str = "gemini") -> str:
    """
    Generate a 4-5 line recommendation for the best vendor and why.
    """
    if not vendors:
        return "No vendors found to recommend."
    
    # Sort vendors by score (highest first)
    sorted_vendors = sorted(vendors, key=lambda v: v.get("score", 0), reverse=True)
    
    # Prepare vendor summary for LLM
    vendor_summaries = []
    for i, vendor in enumerate(sorted_vendors[:5]):  # Top 5 vendors
        name = vendor.get("name", "Unknown")
        score = vendor.get("score", 0)
        pricing = vendor.get("pricing", "Not specified")
        key_features = vendor.get("key_features", [])
        target_users = vendor.get("target_users", "Not specified")
        
        features_text = "; ".join(key_features) if key_features else "Not specified"
        
        vendor_summaries.append(f"""
{i+1}. {name} (Score: {score:.1f}/100)
   - Pricing: {pricing}
   - Key Features: {features_text}
   - Target Users: {target_users}
""")
    
    vendors_text = "\n".join(vendor_summaries)
    
    prompt = f"""
You are a market research analyst providing a concise recommendation.

Topic: {topic}

Based on the following vendor analysis, provide a 4-5 line recommendation for the BEST vendor and explain WHY they are the top choice. Focus on practical business value, key differentiators, and target market fit.

Vendor Analysis:
{vendors_text}

Provide a clear, actionable recommendation in 4-5 lines that a business decision-maker would find valuable.
"""
    
    try:
        response = call_llm(prompt, provider=llm_provider, temperature=0.3)
        if isinstance(response, dict):
            text = response.get("text", "")
        else:
            text = str(response)
        
        # Clean up the response
        text = text.strip()
        if not text:
            return "Unable to generate recommendation at this time."
        
        return text
        
    except Exception as e:
        logger.exception("Failed to generate recommendation: %s", e)
        # Fallback recommendation
        if sorted_vendors:
            best = sorted_vendors[0]
            return f"Based on our analysis, {best.get('name', 'the top vendor')} appears to be the best choice with a score of {best.get('score', 0):.1f}/100. This vendor offers {best.get('pricing', 'competitive pricing')} and targets {best.get('target_users', 'your market segment')}. Their key features include {', '.join(best.get('key_features', [])[:3]) if best.get('key_features') else 'comprehensive functionality'}."
        return "Unable to generate recommendation at this time."


def write_matrix_csv(vendors: List[Dict[str, Any]], path: str) -> None:
    """
    Write a CSV matrix:
    columns: vendor, score, pricing, key_features, integrations, evidence_count
    """
    fieldnames = ["vendor", "score", "pricing", "key_features", "extracted_text", "evidence_count"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for v in vendors:
            extracted_text = v.get("extracted_text", "") or ""
            # Truncate extracted text for CSV readability
            if len(extracted_text) > 200:
                extracted_text = extracted_text[:200] + "..."
            
            w.writerow({
                "vendor": v.get("name", ""),
                "score": v.get("score", 0),
                "pricing": v.get("pricing", "") or "",
                "key_features": "; ".join(v.get("key_features", []) or []),
                "extracted_text": extracted_text,
                "evidence_count": len(v.get("evidence", []))
            })


def write_vendors_json(vendors: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vendors, f, indent=2, ensure_ascii=False)


def _escape_md(s: Optional[str]) -> str:
    if s is None:
        return ""
    # minimal escaping for markdown, keep it readable
    return str(s).replace("\r", " ").replace("\n", " ").strip()


def write_brief_md(topic: str, vendors: List[Dict[str, Any]], path: str, top_k: int = 10) -> None:
    """
    Writes a readable markdown brief summarizing top vendors and citations.
    """
    lines = []
    lines.append(f"# Market Scan Brief")
    lines.append("")
    lines.append(f"**Topic:** {topic}")
    lines.append("")
    lines.append(f"**Generated:**")
    lines.append("")
    
    # Generate recommendation
    try:
        recommendation = generate_recommendation(topic, vendors)
        lines.append("## Recommendation")
        lines.append("")
        lines.append(recommendation)
        lines.append("")
    except Exception as e:
        logger.exception("Failed to generate recommendation for brief: %s", e)
    
    # top vendors sorted by score
    sorted_vs = sorted(vendors, key=lambda x: x.get("score", 0), reverse=True)
    lines.append("## Top Vendors")
    lines.append("")
    for v in sorted_vs[:top_k]:
        name = v.get("name", "Unknown")
        score = v.get("score", 0)
        lines.append(f"### {name} — score: {score:.1f}/100")
        if v.get("pricing"):
            lines.append(f"- **Pricing:** {_escape_md(v.get('pricing'))}")
        if v.get("key_features"):
            lines.append(f"- **Key features:** {_escape_md(', '.join(v.get('key_features', [])))}")
        if v.get("target_users"):
            lines.append(f"- **Target users:** {_escape_md(v.get('target_users'))}")
        if v.get("integrations"):
            lines.append(f"- **Integrations:** {_escape_md(', '.join(v.get('integrations', [])))}")
        if v.get("security_compliance"):
            lines.append(f"- **Security / Compliance:** {_escape_md(', '.join(v.get('security_compliance', [])))}")
        lines.append(f"- **Evidence count:** {len(v.get('evidence') or [])}")
        # citations (up to 3)
        for i, e in enumerate((v.get("evidence") or [])[:3]):
            title = e.get("title") or e.get("url") or ""
            text = e.get("text") or ""
            url = e.get("url") or ""
            # include snippet_id if available
            sid = e.get("snippet_id")
            id_part = f" (snippet: {sid})" if sid else ""
            lines.append(f"  - [{_escape_md(title)}]({url}){id_part}")
            excerpt = text.strip()
            if excerpt:
                if len(excerpt) > 300:
                    excerpt = excerpt[:300].rstrip() + "..."
                lines.append(f"    > {_escape_md(excerpt)}")
        lines.append("")

    # full vendor list (short)
    lines.append("## All Vendors (summary)")
    lines.append("")
    for v in sorted_vs:
        lines.append(f"- {v.get('name')} — score: {v.get('score', 0):.1f}/100")
    lines.append("")

    # write file
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_html_report(topic: str, vendors: List[Dict[str, Any]]) -> str:
    """
    Return a minimal standalone HTML page string representing the brief.
    CSS is inline so the HTML is portable for demos.
    """
    safe_topic = html.escape(topic or "")
    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'/>",
        "<title>Market Scan Report</title>",
        "<style>",
        "body{font-family:Inter,Segoe UI,Roboto,Arial,sans-serif; padding:24px; color:#111}",
        "h1{font-size:24px; margin-bottom:8px}",
        "h2{font-size:18px; margin-top:20px}",
        ".meta{color:#666; font-size:13px}",
        "a{color:#0b66c3}",
        "blockquote{color:#333; margin:8px 0; padding-left:12px; border-left:3px solid #eee}",
        "table{border-collapse:collapse; width:100%; margin-top:20px; background-color:white}",
        "th,td{border:1px solid #ddd; padding:12px; text-align:left; vertical-align:top; background-color:white}",
        "th{background-color:white; font-weight:bold}",
        "tr:nth-child(even){background-color:white}",
        "tr:hover{background-color:white}",
        ".score{font-weight:bold; color:#0b66c3}",
        ".evidence-count{text-align:center}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Market Scan — {safe_topic}</h1>",
        "<p class='meta'>Generated by Agentic Market Scan</p>"
    ]
    
    # Add recommendation section
    try:
        recommendation = generate_recommendation(topic, vendors)
        safe_recommendation = html.escape(recommendation).replace('\n', '<br>')
        html_lines.extend([
            "<h2>Recommendation</h2>",
            f"<div style='background-color:#f8f9fa; padding:16px; border-radius:8px; margin-bottom:24px;'>",
            f"<p style='margin:0; line-height:1.6;'>{safe_recommendation}</p>",
            "</div>"
        ])
    except Exception as e:
        logger.exception("Failed to generate recommendation for HTML: %s", e)
    
    html_lines.extend([
        "<h2>Vendor Comparison Table</h2>"
    ])

    # Create table header
    html_lines.extend([
        "<table>",
        "<thead>",
        "<tr>",
        "<th>Vendor</th>",
        "<th>Score</th>",
        "<th>Pricing</th>",
        "<th>Key Features</th>",
        "<th>Extracted Text</th>",
        "<th>Evidence Count</th>",
        "</tr>",
        "</thead>",
        "<tbody>"
    ])

    # Sort vendors by score (descending)
    sorted_vs = sorted(vendors, key=lambda x: x.get("score", 0), reverse=True)
    
    for v in sorted_vs:
        name = html.escape(v.get("name", "Unknown"))
        score = v.get("score", 0)
        pricing = html.escape(str(v.get("pricing", "")))
        key_features = html.escape(', '.join(v.get("key_features", [])))
        extracted_text = v.get("extracted_text", "") or ""
        # Truncate for display
        if len(extracted_text) > 150:
            extracted_text = extracted_text[:150] + "..."
        extracted_text = html.escape(extracted_text)
        evidence_count = len(v.get("evidence") or [])
        
        html_lines.extend([
            "<tr>",
            f"<td><strong>{name}</strong></td>",
            f"<td class='score'>{score:.1f}/100</td>",
            f"<td>{pricing}</td>",
            f"<td>{key_features}</td>",
            f"<td>{extracted_text}</td>",
            f"<td class='evidence-count'>{evidence_count}</td>",
            "</tr>"
        ])

    html_lines.extend(["</tbody>", "</table>"])

    # Add detailed evidence section below the table
    html_lines.append("<h2>Detailed Evidence</h2>")
    for v in sorted_vs:
        name = html.escape(v.get("name", "Unknown"))
        html_lines.append(f"<h3>{name}</h3>")
        if v.get("evidence"):
            html_lines.append("<ul>")
            for e in v.get("evidence")[:5]:
                title = html.escape(e.get("title") or e.get("url") or "")
                url = html.escape(e.get("url") or "")
                snippet = html.escape((e.get("text") or "")[:400])
                html_lines.append(f"<li><a href='{url}' target='_blank'>{title or url}</a>")
                if snippet:
                    html_lines.append(f"<blockquote>{snippet}</blockquote>")
                html_lines.append("</li>")
            html_lines.append("</ul>")

    html_lines.append("</body></html>")
    return "\n".join(html_lines)


def write_html_report(topic: str, vendors: List[Dict[str, Any]], path: str) -> None:
    html_content = _build_html_report(topic, vendors)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)


def write_pdf_from_html(html_path: str, pdf_path: str) -> bool:
    """
    Optional export to PDF using weasyprint. Returns True on success.
    If weasyprint not installed or fails, return False.
    """
    try:
        from weasyprint import HTML
    except Exception:
        logger.info("weasyprint not installed; skipping PDF generation")
        return False
    try:
        HTML(html_path).write_pdf(pdf_path)
        return True
    except Exception as e:
        logger.exception("PDF generation failed: %s", e)
        return False


def write_reports(run_dir: str,
                  topic: str,
                  vendors: List[Dict[str, Any]],
                  sources: Optional[List[Dict[str, Any]]] = None,
                  do_pdf: bool = False) -> Dict[str, str]:
    """
    High-level writer called by pipeline. Writes artifacts into run_dir and returns mapping.
    """
    os.makedirs(run_dir, exist_ok=True)
    artifacts: Dict[str, str] = {}

    matrix_path = os.path.join(run_dir, "matrix.csv")
    write_matrix_csv(vendors, matrix_path)
    artifacts["matrix"] = matrix_path

    vendors_path = os.path.join(run_dir, "vendors.json")
    write_vendors_json(vendors, vendors_path)
    artifacts["vendors_json"] = vendors_path

    brief_md_path = os.path.join(run_dir, "brief.md")
    write_brief_md(topic, vendors, brief_md_path)
    artifacts["brief_md"] = brief_md_path

    html_path = os.path.join(run_dir, "report.html")
    write_html_report(topic, vendors, html_path)
    artifacts["report_html"] = html_path

    if do_pdf:
        pdf_path = os.path.join(run_dir, "report.pdf")
        ok = write_pdf_from_html(html_path, pdf_path)
        if ok:
            artifacts["report_pdf"] = pdf_path

    # Optionally dump sources snapshot for auditing
    if sources is not None:
        sources_path = os.path.join(run_dir, "sources.json")
        try:
            with open(sources_path, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2, ensure_ascii=False)
            artifacts["sources_json"] = sources_path
        except Exception:
            logger.exception("Failed to write sources.json")

    logger.info("Wrote %d artifacts to %s", len(artifacts), run_dir)
    return artifacts
