"""
Streamlit UI to run the LangGraph pipeline (full pipeline only).

- Runs: backend.pipeline_graph.run_graph_sync(...)
- Displays progress, warnings, and artifact previews (matrix.csv, brief.md, vendors.json, report.html)
- Allows downloading artifacts (CSV/JSON/MD/HTML)

Usage:
    streamlit run webui/streamlit_langraph.py
"""
from __future__ import annotations
import os
import time
import json
import logging
from typing import Optional, Dict, Any

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Agentic Market Scan", layout="wide")
logger = logging.getLogger("streamlit_langraph")
logger.setLevel(logging.INFO)

st.title("Agentic Market Scan")
st.write("This UI runs the full LangGraph pipeline (decompose → scout → ingest → extract → score → write). So please be patient and wait for the pipeline to complete. Approximate time taken to complete the pipeline is 2-5 minutes.")
st.write("Ensure you have set your api-keys in the .env file for the backend services (Tavily, LLM keys) to be available.")

# Import backend pipeline
import sys
import os
# Add the repo root to Python path if not already there
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from backend.pipeline_graph import run_graph_sync
# Sidebar config
with st.sidebar:
    st.header("Run configuration")
    topic = st.text_area("Topic", value="enterprise agent orchestration frameworks 2025", height=80)
    constraints = st.text_input("Constraints (optional)", value="")
    user_id = st.text_input("User ID (optional)", value="demo_user")
    user_plan = st.selectbox("User plan", options=["free", "pro", "premium"], index=0)
    max_urls = st.number_input("Max URLs per subquery (scout)", min_value=1, max_value=10, value=6, step=1)
    run_btn = st.button("Run Full Pipeline")

# Area for logging and status
status_box = st.empty()
progress_box = st.empty()
left_col, right_col = st.columns([2, 1])

def _display_artifacts(artifacts: Dict[str, str]):
    """Show artifact list and previews / download buttons"""
    if not artifacts:
        st.info("No artifacts produced.")
        return

    st.subheader("Artifacts")
    for name, path in artifacts.items():
        st.markdown(f"- **{name}**: `{path}`")
    st.markdown("---")

    # preview matrix.csv if present
    matrix_path = artifacts.get("matrix") or artifacts.get("matrix.csv")
    if matrix_path and os.path.exists(matrix_path):
        st.subheader("Matrix (preview)")
        try:
            df = pd.read_csv(matrix_path)
            st.dataframe(df.head(50))
            # download button
            with open(matrix_path, "rb") as f:
                st.download_button("Download matrix.csv", f, file_name=os.path.basename(matrix_path), mime="text/csv")
        except Exception as e:
            st.error(f"Failed to read matrix.csv: {e}")

    # preview brief.md
    brief_path = artifacts.get("brief_md") or artifacts.get("brief")
    if brief_path and os.path.exists(brief_path):
        st.subheader("Brief (markdown preview)")
        try:
            md = open(brief_path, "r", encoding="utf-8").read()
            st.markdown(md)
            with open(brief_path, "rb") as f:
                st.download_button("Download brief.md", f, file_name=os.path.basename(brief_path), mime="text/markdown")
        except Exception as e:
            st.error(f"Failed to open brief.md: {e}")

    # preview vendors.json
    vendors_path = artifacts.get("vendors_json") or artifacts.get("vendors")
    if vendors_path and os.path.exists(vendors_path):
        st.subheader("Vendors JSON")
        try:
            with open(vendors_path, "r", encoding="utf-8") as f:
                vendors = json.load(f)
            st.json(vendors)
            with open(vendors_path, "rb") as f:
                st.download_button("Download vendors.json", f, file_name=os.path.basename(vendors_path), mime="application/json")
        except Exception as e:
            st.error(f"Failed to open vendors.json: {e}")

    # preview HTML report
    html_path = artifacts.get("report_html") or artifacts.get("report")
    if html_path and os.path.exists(html_path):
        st.subheader("HTML Report preview")
        try:
            html_content = open(html_path, "r", encoding="utf-8").read()
            # show as iframe
            st.components.v1.html(html_content, height=600, scrolling=True)
            with open(html_path, "rb") as f:
                st.download_button("Download report.html", f, file_name=os.path.basename(html_path), mime="text/html")
        except Exception as e:
            st.error(f"Failed to open report.html: {e}")

# Main run logic (blocking)
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic to scan.")
    else:
        job_id = f"job-{int(time.time())}"
        status_box.info(f"Starting LangGraph pipeline run: job_id={job_id}, plan={user_plan}")
        progress_placeholder = progress_box.empty()
        log_area = left_col.empty()
        right_col.header("Run summary")
        right_col.write(f"- **Job ID:** {job_id}")
        right_col.write(f"- **Plan:** {user_plan}")
        right_col.write(f"- **User ID:** {user_id or 'anonymous'}")
        right_col.write(f"- **Topic:** {topic[:120]}")

        try:
            # show spinner while running
            with st.spinner("Running full LangGraph pipeline — this may take a while..."):
                # The pipeline is synchronous: it returns final state
                start_t = time.time()
                final_state = run_graph_sync(job_id=job_id, topic=topic, constraints=constraints or None, user_id=(user_id or None), user_plan=user_plan)
                duration = time.time() - start_t

            status_box.success(f"Pipeline completed in {duration:.1f}s")
            # show warnings if any
            warnings = final_state.get("warnings") or []
            if warnings:
                st.warning("Pipeline emitted warnings. See log below.")
                for w in warnings:
                    st.write(f"- {w}")

            artifacts = final_state.get("artifacts") or {}
            _display_artifacts(artifacts)

            # quick summary table of top vendors if present
            vendors = final_state.get("vendors") or []
            if vendors:
                st.subheader("Top vendors (summary)")
                try:
                    df_v = pd.DataFrame(vendors)
                    # pick relevant columns if available
                    cols = [c for c in ["name", "score", "pricing", "key_features", "target_users"] if c in df_v.columns]
                    st.dataframe(df_v[cols].head(20))
                except Exception:
                    st.write(vendors)
            else:
                st.info("No vendors returned by pipeline.")

            # show raw final state (collapsible)
            with st.expander("Show raw pipeline state (debug)"):
                st.json(final_state)

        except Exception as e:
            status_box.error(f"Pipeline failed: {e}")
            logger.exception("Pipeline run failed")
            with st.expander("Error details"):
                st.text(str(e))
