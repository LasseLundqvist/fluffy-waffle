"""
Agent 1: Research & Assumption Generator

Uses RAG (ChromaDB + sentence-transformers) to retrieve relevant context,
then calls Claude (claude-sonnet-4-6-20250514) with tool_use / structured output
to generate a list of MarketAssumption objects.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from schemas import AssumptionSet, AssumptionsResponse, MarketAssumption
from vector_store import query_chunks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6-20250514"
MAX_TOKENS = 8096
N_RAG_CHUNKS = 10

SYSTEM_PROMPT = (
    "You are a senior construction market economist at a Nordic forecasting firm. "
    "You generate structured 3-year forward-looking assumptions for key market drivers. "
    "Be specific with numbers. Reference the source documents provided. "
    "Write in the language matching the country "
    "(Danish for DK, Norwegian for NO, Swedish for SE). "
    "Be concise and data-driven."
)

# ---------------------------------------------------------------------------
# Tool schema for structured output via Claude tool_use
# ---------------------------------------------------------------------------

ASSUMPTION_TOOL: dict[str, Any] = {
    "name": "submit_market_assumptions",
    "description": (
        "Submit a list of structured market assumptions for the specified country "
        "and time horizon. Each assumption covers one indicator with current value, "
        "three-year forecasts, confidence level, key drivers, and source documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "assumptions": {
                "type": "array",
                "description": "List of market assumptions",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "description": "Name of the metric / indicator",
                        },
                        "current_value": {
                            "type": "string",
                            "description": "Latest known value with source",
                        },
                        "forecast_2026": {
                            "type": "string",
                            "description": "2026 forecast with reasoning",
                        },
                        "forecast_2027": {
                            "type": "string",
                            "description": "2027 forecast with reasoning",
                        },
                        "forecast_2028": {
                            "type": "string",
                            "description": "2028 forecast with reasoning",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Confidence level",
                        },
                        "key_drivers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key factors driving this assumption",
                        },
                        "source_documents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source document filenames used",
                        },
                        "country": {
                            "type": "string",
                            "enum": ["DK", "NO", "SE"],
                            "description": "Country code",
                        },
                    },
                    "required": [
                        "metric",
                        "current_value",
                        "forecast_2026",
                        "forecast_2027",
                        "forecast_2028",
                        "confidence",
                        "key_drivers",
                        "source_documents",
                        "country",
                    ],
                },
            }
        },
        "required": ["assumptions"],
    },
}


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_assumptions(
    query: str,
    country: str,
    api_key: str,
) -> AssumptionSet:
    """
    1. Retrieve relevant RAG chunks for the query + country.
    2. Call Claude with structured tool_use output.
    3. Parse and validate the response into an AssumptionSet.
    """
    # --- RAG retrieval ---
    chunks = query_chunks(query, n_results=N_RAG_CHUNKS, country_filter=country)
    # Also try without country filter if not enough chunks
    if len(chunks) < 3:
        chunks = query_chunks(query, n_results=N_RAG_CHUNKS)

    rag_context = _format_rag_context(chunks)

    # --- Build prompt ---
    user_message = _build_user_message(query, country, rag_context)

    # --- Call Claude ---
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[ASSUMPTION_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_message}],
    )

    # --- Parse tool_use response ---
    assumptions = _parse_tool_response(response, country)

    return AssumptionSet.create(
        country=country,  # type: ignore[arg-type]
        query=query,
        assumptions=assumptions,
        rag_chunks_used=len(chunks),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_rag_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(No documents have been ingested yet. Generate assumptions based on general knowledge.)"

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        fname = meta.get("source_filename", "unknown")
        score = chunk.get("score", 0.0)
        parts.append(
            f"[Chunk {i} | Source: {fname} | Relevance: {score:.2f}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_user_message(query: str, country: str, rag_context: str) -> str:
    country_names = {"DK": "Denmark", "NO": "Norway", "SE": "Sweden"}
    country_name = country_names.get(country, country)

    return f"""## Task
Generate structured 3-year market assumptions (2026–2028) for **{country_name} ({country})**.

## User Request
{query}

## Retrieved Source Documents
{rag_context}

## Instructions
- Use the `submit_market_assumptions` tool to return your response.
- Generate one assumption object per market indicator requested.
- Be specific: include concrete numbers, percentages, or index values where possible.
- Reference actual source document filenames from the context above.
- Set confidence based on data availability and uncertainty.
- Write forecasts and reasoning in the language of {country_name}.
"""


def _parse_tool_response(
    response: anthropic.types.Message,
    country: str,
) -> list[MarketAssumption]:
    """Extract and validate assumptions from Claude's tool_use response."""
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_market_assumptions":
            raw = block.input
            validated = AssumptionsResponse.model_validate(raw)
            # Ensure country is set correctly on each assumption
            for assumption in validated.assumptions:
                if not assumption.country:
                    assumption.country = country  # type: ignore[assignment]
            return validated.assumptions

    # Fallback: if no tool call found, try to parse text content
    raise ValueError(
        "Claude did not return a structured tool_use response. "
        "Check the API key and model availability."
    )
