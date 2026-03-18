"""
Pydantic schemas for structured outputs in the Construction Market Forecasting PoC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent 1: Market Assumption
# ---------------------------------------------------------------------------

class MarketAssumption(BaseModel):
    """A single structured market assumption for one indicator."""

    metric: str = Field(
        description="Name of the metric, e.g. 'Housing starts', 'Real interest rate'"
    )
    current_value: str = Field(
        description="Latest known value with source reference"
    )
    forecast_2026: str = Field(
        description="Year 1 (2026) forecast with short reasoning"
    )
    forecast_2027: str = Field(
        description="Year 2 (2027) forecast with short reasoning"
    )
    forecast_2028: str = Field(
        description="Year 3 (2028) forecast with short reasoning"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level for this forecast"
    )
    key_drivers: list[str] = Field(
        description="List of key factors driving this assumption"
    )
    source_documents: list[str] = Field(
        description="Filenames of source documents used to inform this assumption"
    )
    country: Literal["DK", "NO", "SE"] = Field(
        description="Country code this assumption applies to"
    )


class AssumptionSet(BaseModel):
    """A timestamped collection of market assumptions from one generation run."""

    id: str = Field(description="Unique identifier for this assumption set")
    timestamp: str = Field(description="ISO-format creation timestamp")
    country: Literal["DK", "NO", "SE"]
    query: str = Field(description="The user query that triggered this generation")
    assumptions: list[MarketAssumption]
    rag_chunks_used: int = Field(
        default=0, description="Number of RAG chunks retrieved"
    )

    @classmethod
    def create(
        cls,
        country: Literal["DK", "NO", "SE"],
        query: str,
        assumptions: list[MarketAssumption],
        rag_chunks_used: int = 0,
    ) -> "AssumptionSet":
        import uuid

        return cls(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            country=country,
            query=query,
            assumptions=assumptions,
            rag_chunks_used=rag_chunks_used,
        )


# ---------------------------------------------------------------------------
# Agent 2: Narrative
# ---------------------------------------------------------------------------

class NarrativeResult(BaseModel):
    """Result from the Table Narrative Writer agent."""

    id: str = Field(description="Unique identifier for this narrative")
    timestamp: str = Field(description="ISO-format creation timestamp")
    country: Literal["DK", "NO", "SE"]
    title: str
    narrative_text: str = Field(description="Full editorial narrative in plain text")
    referenced_assumption_ids: list[str] = Field(
        description="IDs of AssumptionSets referenced in this narrative"
    )

    @classmethod
    def create(
        cls,
        country: Literal["DK", "NO", "SE"],
        narrative_text: str,
        referenced_assumption_ids: list[str],
    ) -> "NarrativeResult":
        import uuid

        country_names = {"DK": "Denmark", "NO": "Norway", "SE": "Sweden"}
        return cls(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            country=country,
            title=f"Market Commentary — {country_names[country]} Construction Market",
            narrative_text=narrative_text,
            referenced_assumption_ids=referenced_assumption_ids,
        )


# ---------------------------------------------------------------------------
# Internal: Claude structured-output response wrapper
# ---------------------------------------------------------------------------

class AssumptionsResponse(BaseModel):
    """Wrapper that Claude returns when generating assumptions."""

    assumptions: list[MarketAssumption]
