from typing import Annotated, List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_file: str = Field(..., description="Relative path of the source file")
    page: Optional[int] = Field(None, description="Page number if available, else null")
    chunk_id: str = Field(..., description="Stable chunk identifier")
    quote: str = Field(..., description="Short supporting quote from the chunk")


class QAResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    insufficient_evidence: bool = False