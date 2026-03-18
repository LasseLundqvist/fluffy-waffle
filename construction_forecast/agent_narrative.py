"""
Agent 2: Table Narrative Writer

Receives a forecast data table (as a pandas DataFrame) and selected AssumptionSets,
then calls Claude (claude-sonnet-4-6-20250514) to generate editorial-quality commentary.
"""

from __future__ import annotations

import anthropic

from schemas import AssumptionSet, NarrativeResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6-20250514"
MAX_TOKENS = 4096

SYSTEM_PROMPT = (
    "You are an editorial economist writing for a professional industry publication. "
    "Describe the trends in the data table provided, referencing the market assumptions. "
    "Write in a style similar to The Economist — authoritative, concise, with clear "
    "cause-and-effect reasoning. Use the language matching the country. "
    "Structure the text with a brief overview paragraph, then one paragraph per major trend. "
    "Never use bullet points."
)

COUNTRY_LANG = {"DK": "Danish", "NO": "Norwegian", "SE": "Swedish"}


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_narrative(
    table_markdown: str,
    assumption_sets: list[AssumptionSet],
    country: str,
    api_key: str,
) -> NarrativeResult:
    """
    Generate editorial narrative for the provided table using Claude.

    Args:
        table_markdown: The forecast table converted to markdown format.
        assumption_sets: Previously generated AssumptionSets to reference.
        country: Country code ("DK", "NO", "SE").
        api_key: Anthropic API key.

    Returns:
        NarrativeResult with the generated text.
    """
    user_message = _build_user_message(table_markdown, assumption_sets, country)

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    narrative_text = _extract_text(response)

    return NarrativeResult.create(
        country=country,  # type: ignore[arg-type]
        narrative_text=narrative_text,
        referenced_assumption_ids=[aset.id for aset in assumption_sets],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_assumptions(assumption_sets: list[AssumptionSet]) -> str:
    if not assumption_sets:
        return "(No assumption sets selected.)"

    parts: list[str] = []
    for aset in assumption_sets:
        parts.append(
            f"### Assumption Set {aset.id} — {aset.country} "
            f"({aset.timestamp[:10]})\nQuery: {aset.query}\n"
        )
        for assumption in aset.assumptions:
            parts.append(
                f"**{assumption.metric}** (confidence: {assumption.confidence})\n"
                f"- Current: {assumption.current_value}\n"
                f"- 2026: {assumption.forecast_2026}\n"
                f"- 2027: {assumption.forecast_2027}\n"
                f"- 2028: {assumption.forecast_2028}\n"
                f"- Key drivers: {'; '.join(assumption.key_drivers)}\n"
            )
    return "\n".join(parts)


def _build_user_message(
    table_markdown: str,
    assumption_sets: list[AssumptionSet],
    country: str,
) -> str:
    lang = COUNTRY_LANG.get(country, "English")
    country_names = {"DK": "Denmark", "NO": "Norway", "SE": "Sweden"}
    country_name = country_names.get(country, country)

    assumptions_text = _format_assumptions(assumption_sets)

    return f"""## Task
Write an editorial narrative for the **{country_name} construction market** based on the forecast table below.

## Language
Write entirely in **{lang}**.

## Forecast Data Table
{table_markdown}

## Market Driver Assumptions
{assumptions_text}

## Output Requirements
- Start with a concise overview paragraph (2–3 sentences) summarising the overall picture.
- Follow with one paragraph per major trend visible in the table, linking each trend to the relevant assumptions.
- Write in The Economist style: authoritative, precise, no bullet points.
- Do not include headers or section titles — only flowing prose paragraphs.
- Conclude with a brief forward-looking paragraph.
"""


def _extract_text(response: anthropic.types.Message) -> str:
    """Extract plain text from Claude's response."""
    parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n\n".join(parts).strip()
