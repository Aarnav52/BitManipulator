# Talent Intelligence Platform
### Multi-Agent AI Resume Parsing, Skill Normalization & Job Matching
**DA-IICT Hackathon — Problem Statement 9 | Prama Innovations**

---

## Project Structure

```
talent_intel/
├── main.py                      ← FastAPI entry point (serves UI + API)
├── agents/
│   ├── parsing_agent.py         ← Agent 1: PDF/DOCX/TXT → structured data via LLM
│   ├── taxonomy_agent.py        ← Agent 2: raw skills → canonical taxonomy + inference
│   ├── matching_agent.py        ← Agent 3: semantic job-candidate scoring
│   └── orchestrator.py          ← Pipeline: parse → normalize → register
├── api/
│   ├── dependencies.py          ← API key auth, file validation
│   └── routes/
│       ├── parse.py             ← POST /api/v1/parse  (single + batch)
│       ├── match.py             ← POST /api/v1/match  + candidate store
│       ├── candidates.py        ← GET  /api/v1/candidates/*
│       └── taxonomy.py          ← GET  /api/v1/skills/*
├── models/
│   └── resume.py                ← All Pydantic models (single source of truth)
├── core/
│   └── config.py                ← Settings from .env
├── frontend/
│   └── index.html               ← Full single-page web UI
├── tests/
│   ├── test_parsing_agent.py
│   ├── test_taxonomy_agent.py
│   └── test_matching_agent.py
├── data/sample_resumes/
│   └── john_doe.txt             ← Sample resume for testing
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Option A — Docker (recommended)
```bash
cp .env.example .env
# Edit .env → set ANTHROPIC_API_KEY=your_key_here

docker-compose up --build
```
Open http://localhost:8000 for the web UI.

### Option B — Local dev
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set ANTHROPIC_API_KEY in .env (leave blank to use regex fallback)

# Start Postgres + Redis via Docker
docker-compose up postgres redis -d

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Reference

**Auth:** pass `X-API-Key: demo-key-hackathon` in all requests.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/parse` | Parse single resume (file upload) |
| POST | `/api/v1/parse/text` | Parse plain-text resume |
| POST | `/api/v1/parse/batch` | Batch parse up to 50 resumes (async) |
| GET  | `/api/v1/parse/batch/{job_id}` | Poll batch job status |
| GET  | `/api/v1/candidates` | List all parsed candidates |
| GET  | `/api/v1/candidates/{id}` | Get full candidate profile |
| GET  | `/api/v1/candidates/{id}/skills` | Get normalized skill profile |
| POST | `/api/v1/match` | Semantic job-candidate matching |
| GET  | `/api/v1/skills/taxonomy` | Browse skill taxonomy |
| GET  | `/api/v1/skills/search?q=python` | Search skills |
| GET  | `/api/v1/skills/{name}` | Lookup single skill |
| GET  | `/health` | Health check |
| GET  | `/docs` | Swagger UI |

### Parse example
```bash
curl -X POST http://localhost:8000/api/v1/parse \
  -H "X-API-Key: demo-key-hackathon" \
  -F "file=@data/sample_resumes/john_doe.txt"
```

### Match example
```bash
curl -X POST http://localhost:8000/api/v1/match \
  -H "X-API-Key: demo-key-hackathon" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "<id-from-parse>",
    "job_title": "Senior Backend Engineer",
    "job_description": "We need a Python expert with FastAPI and AWS experience.",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "preferred_skills": ["Kubernetes", "AWS"],
    "min_experience_years": 4
  }'
```

---

## Run Tests
```bash
pytest tests/ -v
```

---

## Evaluation Criteria Coverage

| Criterion | Implementation |
|-----------|----------------|
| Resume parsing F1 | LLM extraction + pdfplumber + PyMuPDF fallback + regex fallback |
| Skill normalization precision | Exact alias dict + fuzzy SequenceMatcher (82% threshold) |
| Matching quality (NDCG) | sentence-transformers cosine similarity + proficiency weighting |
| API completeness | 11 endpoints, OpenAPI/Swagger auto-docs |
| Multi-agent orchestration | Pipeline: parse → normalize → register, graceful degradation |
| End-to-end latency <10s | Synchronous pipeline, pre-cached transformer model |
| Multi-format support | PDF (pdfplumber + PyMuPDF), DOCX, TXT |
