# Construction Market Forecasting PoC

**Prognosecenteret — Nordic Construction Market Analysis**

An AI-powered Streamlit application for generating structured construction market assumptions and editorial narratives for Denmark, Norway, and Sweden.

---

## Features

| Capability | Details |
|---|---|
| **Document Ingestion** | Upload PDFs → PyMuPDF parse → chunk → embed → ChromaDB |
| **Agent 1: Research & Assumptions** | RAG retrieval + Claude claude-sonnet-4-6-20250514 → structured 3-year market assumptions |
| **Agent 2: Table Narrative Writer** | Forecast table + assumptions → Economist-style editorial commentary |
| **Word Export** | Download narratives as professionally formatted `.docx` |
| **Report Archive** | All outputs saved as JSON in `./output/` |

---

## Project Structure

```
construction_forecast/
├── app.py                 # Main Streamlit application
├── schemas.py             # Pydantic data models
├── ingestion.py           # PDF parsing & chunking
├── vector_store.py        # ChromaDB + sentence-transformers
├── agent_research.py      # Agent 1: Assumption Generator
├── agent_narrative.py     # Agent 2: Narrative Writer
├── docx_generator.py      # Word document generator
├── requirements.txt       # Pinned dependencies
├── .env.example           # Environment variable template
├── .streamlit/
│   └── config.toml        # Streamlit theme config
├── chroma_db/             # Local ChromaDB storage (auto-created)
└── output/                # Generated reports (auto-created)
```

---

## Setup

### 1. Prerequisites

- Python 3.10 or 3.11
- pip

### 2. Install dependencies

```bash
cd construction_forecast
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `paraphrase-multilingual-MiniLM-L12-v2` model (~120 MB) on first run. An internet connection is required.

### 3. Configure API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# Edit .env and set your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

Alternatively, enter the key directly in the app sidebar.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage Guide

### Tab 1 — Document Ingestion
1. Select the target country in the sidebar.
2. Upload one or more PDF files (market reports, policy documents, statistics).
3. Click **Ingest Documents** — the progress bar shows per-file status.
4. The sidebar updates with total chunk count.

### Tab 2 — Research & Assumptions
1. Describe the market drivers to analyse in the text area (in any language).
2. Click **Generate Assumptions** — RAG retrieves relevant chunks, Claude generates structured assumptions.
3. Expand individual assumption cards to review forecasts and drivers.
4. Download as JSON or review session history.

### Tab 3 — Table Narratives
1. Upload a CSV or Excel forecast table.
2. Select assumption sets from the session to reference.
3. Click **Generate Narrative** — Claude writes Economist-style commentary.
4. Download as Word (`.docx`).

### Tab 4 — Report Archive
- All generated outputs are saved to `./output/` as JSON.
- Download archived reports or Word exports from this tab.

---

## Technical Details

| Component | Technology |
|---|---|
| LLM | `anthropic` SDK · `claude-sonnet-4-6-20250514` |
| Structured output | Claude `tool_use` with Pydantic validation |
| Embeddings | `sentence-transformers` · `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | `chromadb` · `PersistentClient` · cosine similarity |
| PDF parsing | `pymupdf` (fitz) · text + table extraction |
| Text chunking | Pure-Python recursive character splitter (~800 words/chunk, 200 overlap) |
| Word export | `python-docx` · Calibri 11pt body / 18pt title |
| Data handling | `pandas` + `openpyxl` |

---

## Multilingual Support

The embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) natively supports Danish, Norwegian, and Swedish. Claude is instructed to respond in the language matching the selected country:

- 🇩🇰 DK → Danish
- 🇳🇴 NO → Norwegian
- 🇸🇪 SE → Swedish

---

## Data Privacy

All processing is local except for:
- Claude API calls (text sent to Anthropic)
- Embedding model download on first run

ChromaDB data is stored locally in `./chroma_db/`. No data is sent to third-party services except the Anthropic API.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| API key error | Check `.env` file or sidebar input |
| Empty RAG context | Ingest documents in Tab 1 first |
| ChromaDB error on cold start | Delete `./chroma_db/` and re-ingest |
| Slow first run | sentence-transformers downloads model (~120 MB) once |
