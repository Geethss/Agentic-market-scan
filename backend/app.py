from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import time

from backend.pipeline_graph import run_graph_sync

api = FastAPI()

class ScanRequest(BaseModel):
    topic: str
    constraints: Optional[str] = None
    user_id: Optional[str] = None
    user_plan: str = "free"

@api.post("/scan")
def scan(req: ScanRequest):
    """
    Run a market scan using the full pipeline.
    Returns job_id and status.
    """
    job_id = f"job-{int(time.time())}"
    
    try:
        # Run the full pipeline synchronously
        result = run_graph_sync(
            job_id=job_id,
            topic=req.topic,
            constraints=req.constraints,
            user_id=req.user_id,
            user_plan=req.user_plan
        )
        
        return {
            "job_id": job_id,
            "status": "DONE",
            "vendors_found": len(result.get("vendors", [])),
            "artifacts": result.get("artifacts", {}),
            "warnings": result.get("warnings", []),
            "duration": result.get("duration", 0)
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "FAILED",
            "error": str(e)
        }