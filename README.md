# Automated Insight Engine (H-001)

**Description:** Ingest CSV/DB data and automatically produce executive-ready PowerPoint reports with charts and a short AI-generated executive summary.

---

## The Problem (Real World Scenario)
Marketing and analytics teams spend hours manually pulling data, creating charts, and writing executive summaries for stakeholders. This process is slow, error-prone, and does not scale for terabytes of data or frequent reporting cycles.

---

## Expected End Result
Given a structured CSV (or DB export), the system will:
- Generate a downloadable PPTX report containing:
  - Title slide
  - Executive summary (AI-generated if key available, otherwise templated)
  - 1 chart for the primary numeric metric
  - Top-level KPIs slide
- The output file will be placed in `output/` and is ready for sharing with stakeholders.

---

## Technical Approach
1. **Ingest** CSV and compute descriptive statistics using `pandas`.
2. **Analyze**: extract top KPIs and sample rows.
3. **Summarize**: use OpenAI (optional) to generate a short executive summary. If no key is set, a templated summary is used.
4. **Render**: produce a PowerPoint (`python-pptx`) with slides and a matplotlib chart.
5. **Expose**: FastAPI endpoint `/generate_report` to upload CSV and download the report.

---

## Tech Stack
- Python 3.10+
- FastAPI (API)
- pandas (data processing)
- matplotlib (plots)
- python-pptx (PPTX generation)
- Docker + docker-compose (containerization)
- OpenAI (optional, for nicer executive summaries)

---

## Challenges & Learnings
- **Chart layout in PPTX**: python-pptx does not directly embed matplotlib objects; images must be saved and attached.
- **Clean summary:** LLMs can hallucinate; we only send high-level KPIs to LLM for a concise summary to reduce hallucination risk.
- **Extensibility:** this pipeline is intentionally modular so you can add more KPIs, multiple charts, and DB ingestion later.

---

## Visual Proof
- `output/example_report.pptx` — generated report (run the pipeline to create it).
- `screenshots/` — include 1–3 PNG images showing the slides (take screenshots of the PPTX slides and put here).

---

## How to Run (Local)
### Pre-requisites
- Python 3.10+
- (Optional) OpenAI API key if you want the AI-generated summary: `export OPENAI_API_KEY="sk-..."`

### Quick local (no Docker)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate example output via CLI:
python run_report.py data/sample_input.csv output/example_report.pptx

# Or run the API:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000 and use the upload form or use /docs
