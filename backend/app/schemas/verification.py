from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PipelineStageResult(BaseModel):
    stage: str
    status: str
    detail: Optional[str] = None
    score: Optional[float] = None


class EvidenceItem(BaseModel):
    source: str
    title: str
    publishedDate: str
    retrievedDate: str
    relation: str = Field(description="Relation: supports | contradicts | insufficient | irrelevant")
    relevance: float
    excerpt: str
    url: str


class VerificationResult(BaseModel):
    id: str
    prediction: str = Field(
    ...,
    description="Verification prediction label: Genuine | Misleading | Pending",
)
    confidence_score: float = Field(
        0.0,
        description="Model confidence score (0.0 indicates stub/uncalculated)",
    )
    # Frontend compatibility alias
    confidenceScore: Optional[float] = None

    # Real CLIP cross-modal alignment score (raw logit output)
    clip_score: Optional[float] = Field(
        None,
        description="Real CLIP image-caption alignment score (raw logit output)",
    )
    clipScore: Optional[float] = Field(
        None,
        description="Frontend camelCase alias for real CLIP alignment score",
    )
    alignment_score: Optional[float] = Field(
        None,
        description="Cross-modal alignment score alias",
    )
    alignmentScore: Optional[float] = Field(
        None,
        description="Frontend camelCase alias for alignment score",
    )

    reason: str = Field(
        ...,
        description="Reason statement for the prediction",
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Retrieved external evidence items",
    )
    explanation: str = Field(
        ...,
        description="Detailed natural language reasoning explanation",
    )
    inconsistency_type: str = Field(
        ...,
        description="Type of inconsistency detected, or stub notice",
    )
    # Frontend compatibility alias
    inconsistencyType: Optional[str] = None

    dataset: str = Field(
        "Custom Pair",
        description="Dataset name or source classification",
    )
    ground_truth_label: Optional[str] = Field(
        None,
        description="Ground truth label if available from benchmark dataset",
    )
    # Frontend compatibility alias
    groundTruthLabel: Optional[str] = None

    pipeline_stages: List[PipelineStageResult] = Field(
        default_factory=list,
        description="List of pipeline stages executed with completion status",
    )

    is_stub: bool = Field(
        True,
        description="Explicit flag indicating this response is from a backend test stub, not an AI prediction",
    )
