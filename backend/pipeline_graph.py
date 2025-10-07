"""
LangGraph pipeline for Agentic Market Scan.

Wires together tested agents:
- decompose
- scout
- ingest (with dedupe)
- extract
- score
- write (reports)

Run via:
    from backend.pipeline_graph import pipeline
    out = pipeline.invoke({"topic": "enterprise agent orchestration frameworks 2025"})
"""

from __future__ import annotations
import os
import time
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END


from backend.agents.decompose import decompose_topic
from backend.agents.scout import scout
from backend.agents.ingest import ingest_sources
from backend.agents.extract import extract_vendors
from backend.agents.score import score_vendors
from backend.agents.write import write_reports

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class PipelineState(TypedDict, total=False):
    job_id: str
    topic: str
    constraints: Optional[str]
    user_id: Optional[str]
    user_plan: str
    fields: List[str]
    subqueries: List[str]
    sources: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    vendors: List[Dict[str, Any]]
    artifacts: Dict[str, str]
    warnings: List[str]
    start_time: float
    duration: float



def node_decompose(state: PipelineState) -> PipelineState:
    logger.info("[decompose] topic=%s", state["topic"])
    dec = decompose_topic(state["topic"], state.get("constraints"), use_llm=True)
    state["fields"] = dec.get("fields", [])
    state["subqueries"] = dec.get("subqueries", [])
    return state


def node_scout(state: PipelineState) -> PipelineState:
    logger.info("[scout] subqueries=%d", len(state.get("subqueries", [])))
    fetch_full = (state.get("user_plan") == "premium")
    try:
        sources = scout(
            state["subqueries"],
            max_urls_per_query=6,
            fetch_full=fetch_full,
            user_id=state.get("user_id"),
            user_plan=state.get("user_plan", "free"),
        )
    except Exception as e:
        logger.exception("Scout failed: %s", e)
        state.setdefault("warnings", []).append(f"scout_failed:{str(e)}")
        sources = []
    state["sources"] = sources
    return state


def node_ingest(state: PipelineState) -> PipelineState:
    logger.info("[ingest] sources=%d", len(state.get("sources", [])))
    try:
        chunks, dupes = ingest_sources(
            state["sources"],
            job_id=state.get("job_id", f"job_{int(time.time())}"),
            use_openai_embeddings=(state.get("user_plan") in ("pro", "premium")),
            collection_name="snippets",
        )
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        state.setdefault("warnings", []).append(f"ingest_failed:{str(e)}")
        chunks, dupes = [], {}
    state["chunks"] = chunks
    return state


def node_extract(state: PipelineState) -> PipelineState:
    logger.info("[extract] fields=%d", len(state.get("fields", [])))
    
    # Field-specific retrieval: query Chroma for each field
    snippets_per_field = {}
    fields = state.get("fields", [])
    
    try:
        from backend.services.chroma_client import query_by_embedding, get_or_create_collection
        from backend.agents.ingest import _init_local_embedder
        
        collection = get_or_create_collection("snippets")
        embedder = _init_local_embedder()
        
        for field in fields:
            # Create field-specific query
            field_query = f"{state['topic']} {field}"
            
            # Generate embedding using our sentence-transformers model
            query_embedding = embedder.encode([field_query])[0].tolist()
            
            
            results = query_by_embedding(
                collection_name="snippets",
                embedding=query_embedding,
                top_k=6  
            )
            
            # Convert Chroma results to the format expected by extract_vendors
            field_snippets = []
            if results and "documents" in results:
                documents = results.get("documents", [[]])[0]  
                metadatas = results.get("metadatas", [[]])[0] 
                ids = results.get("ids", [[]])[0]  
                
                for i, doc in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    doc_id = ids[i] if i < len(ids) else f"doc_{i}"
                    
                    snippet = {
                        "id": doc_id,
                        "url": metadata.get("url", ""),
                        "title": metadata.get("title", ""),
                        "text": doc,
                        "published": metadata.get("published")
                    }
                    field_snippets.append(snippet)
            
            snippets_per_field[field] = field_snippets
            logger.info("[extract] field=%s found %d snippets", field, len(field_snippets))
            
    except Exception as e:
        logger.exception("Field-specific retrieval failed: %s", e)
        # Fallback to simple approach
        snippets_per_field = {f: state.get("chunks", [])[:3] for f in fields}
        state.setdefault("warnings", []).append(f"field_retrieval_failed:{str(e)}")
    
    try:
        out = extract_vendors(
            state["topic"],
            fields,
            snippets_per_field,
            llm_provider="gemini" if state.get("user_plan") == "free" else "openai"
        )
        vendors, warnings = out.get("vendors", []), out.get("warnings", [])
    except Exception as e:
        logger.exception("Extract failed: %s", e)
        vendors, warnings = [], [f"extract_failed:{str(e)}"]
    state["vendors"] = vendors
    state.setdefault("warnings", []).extend(warnings)
    return state


def node_score(state: PipelineState) -> PipelineState:
    logger.info("[score] vendors=%d", len(state.get("vendors", [])))
    try:
        state["vendors"] = score_vendors(state.get("vendors", []), state.get("sources", []))
    except Exception as e:
        logger.exception("Score failed: %s", e)
        state.setdefault("warnings", []).append(f"score_failed:{str(e)}")
    return state


def node_write(state: PipelineState) -> PipelineState:
    logger.info("[write] vendors=%d", len(state.get("vendors", [])))
    run_dir = os.path.join("runs", state.get("job_id", f"job_{int(time.time())}"))
    try:
        artifacts = write_reports(run_dir, state["topic"], state.get("vendors", []), sources=state.get("sources", []))
        state["artifacts"] = artifacts
    except Exception as e:
        logger.exception("Write failed: %s", e)
        state.setdefault("warnings", []).append(f"write_failed:{str(e)}")
        state["artifacts"] = {}
    return state



graph = StateGraph(PipelineState)

graph.add_node("decompose", node_decompose)
graph.add_node("scout", node_scout)
graph.add_node("ingest", node_ingest)
graph.add_node("extract", node_extract)
graph.add_node("score", node_score)
graph.add_node("write", node_write)


graph.set_entry_point("decompose")


graph.add_edge("decompose", "scout")
graph.add_edge("scout", "ingest")
graph.add_edge("ingest", "extract")
graph.add_edge("extract", "score")
graph.add_edge("score", "write")
graph.add_edge("write", END)

pipeline = graph.compile()



def run_graph_sync(job_id: str, topic: str, constraints: Optional[str] = None, user_id: Optional[str] = None, user_plan: str = "free") -> PipelineState:
    init_state: PipelineState = {
        "job_id": job_id,
        "topic": topic,
        "constraints": constraints,
        "user_id": user_id,
        "user_plan": user_plan,
        "start_time": time.time(),
    }
    final_state = pipeline.invoke(init_state)
    final_state["duration"] = time.time() - init_state["start_time"]
    return final_state
