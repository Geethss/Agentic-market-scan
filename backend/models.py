from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Source(BaseModel):
    url: str
    title: Optional[str] = None
    published: Optional[str] = None
    domain: str

class Evidence(BaseModel):
    source_url: str
    snippet: str
    selector: Optional[str] = None
    trust: float
    recency: float

class Vendor(BaseModel):
    name: str
    pricing: Optional[str] = None
    key_features: List[str] = Field(default_factory=list)
    target_users: Optional[str] = None
    extracted_text: Optional[str] = None
    support_and_sla: Optional[str] = None
    security_compliance: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None
    confidence: Optional[float] = None
    score: Optional[float] = None

class MatrixCell(BaseModel):
    vendor: str
    field: str
    value: str
    citations: List[Evidence] = Field(default_factory=list)
