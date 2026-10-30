
from typing import Literal
from pydantic import BaseModel, Field


class ReviewScores(BaseModel):
    factual_accuracy: int = Field(ge=0, le=25)
    grammar: int = Field(ge=0, le=15)
    readability: int = Field(ge=0, le=15)
    structure: int = Field(ge=0, le=15)
    seo_compliance: int = Field(ge=0, le=15)
    source_quality: int = Field(ge=0, le=15)


class ReviewIssue(BaseModel):
    category: Literal[
        "factual_accuracy",
        "grammar",
        "readability",
        "structure",
        "seo",
        "sources"
    ]

    severity: Literal["critical", "major", "minor"]

    description: str
    suggested_correction: str


class ContentReview(BaseModel):
    decision: Literal[
        "APPROVED",
        "REVISION_REQUIRED"
    ]

    scores: ReviewScores

    summary: str

    strengths: list[str]

    issues: list[ReviewIssue]

    revision_instructions: list[str]

    factual_verification_limitations: list[str]
