"""
Word document (.docx) generator for narrative reports.

Produces professionally formatted documents using python-docx:
  - Calibri font, 11pt body, 14pt title
  - Title + subtitle
  - Narrative body paragraphs
  - "Referenced Assumptions" appendix section
"""

from __future__ import annotations

import io
from datetime import datetime

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.util import Inches

from schemas import AssumptionSet, NarrativeResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FONT_BODY = "Calibri"
FONT_TITLE = "Calibri"
SIZE_TITLE = Pt(18)
SIZE_SUBTITLE = Pt(12)
SIZE_HEADING = Pt(13)
SIZE_BODY = Pt(11)
COLOR_TITLE = RGBColor(0x1E, 0x5F, 0xA6)  # Prognosecenteret blue
COLOR_BODY = RGBColor(0x1A, 0x1A, 0x2E)


COUNTRY_NAMES = {"DK": "Denmark", "NO": "Norway", "SE": "Sweden"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_narrative_docx(
    narrative: NarrativeResult,
    assumption_sets: list[AssumptionSet],
) -> bytes:
    """
    Generate a Word document from a NarrativeResult.
    Returns raw bytes suitable for st.download_button.
    """
    doc = Document()

    _set_default_styles(doc)

    # --- Title ---
    _add_title(doc, narrative.title)

    # --- Subtitle ---
    date_str = datetime.fromisoformat(narrative.timestamp).strftime("%d %B %Y")
    _add_subtitle(doc, f"Generated {date_str} | {COUNTRY_NAMES.get(narrative.country, narrative.country)}")

    _add_horizontal_rule(doc)

    # --- Narrative body ---
    paragraphs = [p.strip() for p in narrative.narrative_text.split("\n\n") if p.strip()]
    for para_text in paragraphs:
        _add_body_paragraph(doc, para_text)

    # --- Referenced Assumptions section ---
    if assumption_sets:
        doc.add_paragraph()  # spacer
        _add_section_heading(doc, "Referenced Assumptions")

        for aset in assumption_sets:
            _add_assumption_set_summary(doc, aset)

    # --- Serialize to bytes ---
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _set_default_styles(doc: Document) -> None:
    """Apply global font defaults to Normal style."""
    style = doc.styles["Normal"]
    font = style.font
    font.name = FONT_BODY
    font.size = SIZE_BODY
    font.color.rgb = COLOR_BODY

    # Set document margins (2.5cm all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.2)


def _add_title(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = para.add_run(text)
    run.font.name = FONT_TITLE
    run.font.size = SIZE_TITLE
    run.font.bold = True
    run.font.color.rgb = COLOR_TITLE


def _add_subtitle(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = para.add_run(text)
    run.font.name = FONT_TITLE
    run.font.size = SIZE_SUBTITLE
    run.font.bold = False
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x77)
    para.space_after = Pt(6)


def _add_horizontal_rule(doc: Document) -> None:
    """Add a simple paragraph with bottom border as a horizontal rule."""
    para = doc.add_paragraph()
    para.space_after = Pt(12)
    # Add border via XML
    from docx.oxml import OxmlElement
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "1E5FA6")
    pBdr.append(bottom)
    pPr.append(pBdr)


def _add_body_paragraph(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_after = Pt(8)
    run = para.add_run(text)
    run.font.name = FONT_BODY
    run.font.size = SIZE_BODY
    run.font.color.rgb = COLOR_BODY


def _add_section_heading(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = FONT_TITLE
    run.font.size = SIZE_HEADING
    run.font.bold = True
    run.font.color.rgb = COLOR_TITLE
    para.space_before = Pt(12)
    para.space_after = Pt(6)


def _add_assumption_set_summary(doc: Document, aset: AssumptionSet) -> None:
    """Render a compact summary of one AssumptionSet."""
    date_str = aset.timestamp[:10]
    header = doc.add_paragraph()
    run = header.add_run(
        f"Assumption Set {aset.id} — {COUNTRY_NAMES.get(aset.country, aset.country)} ({date_str})"
    )
    run.font.name = FONT_BODY
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x55)
    header.space_after = Pt(2)

    for assumption in aset.assumptions:
        row = doc.add_paragraph()
        row.paragraph_format.left_indent = Inches(0.25)
        row.space_after = Pt(4)

        metric_run = row.add_run(f"{assumption.metric}:  ")
        metric_run.font.bold = True
        metric_run.font.size = Pt(10)
        metric_run.font.name = FONT_BODY

        detail_run = row.add_run(
            f"2026 → {assumption.forecast_2026[:80]}...  |  "
            f"Confidence: {assumption.confidence}"
        )
        detail_run.font.size = Pt(10)
        detail_run.font.name = FONT_BODY
        detail_run.font.color.rgb = RGBColor(0x44, 0x44, 0x66)
