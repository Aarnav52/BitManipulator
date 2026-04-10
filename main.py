"""
Talent Intelligence API
=======================
Multi-Agent Resume Parsing, Skill Normalization & Job Matching
DA-IICT Hackathon — Problem Statement 9 | Prama Innovations

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000        → Web UI
    http://localhost:8000/docs   → Swagger API docs
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import get_settings
from api.routes.parse import router as parse_router
from api.routes.candidates import router as candidates_router
from api.routes.taxonomy import router as taxonomy_router
from api.routes.match import router as match_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Talent Intelligence API starting up")
    logger.info(f"  API key:   demo-key-hackathon")
    logger.info(f"  Docs:      http://localhost:8000/docs")
    logger.info(f"  UI:        http://localhost:8000")
    logger.info("=" * 60)
    yield
    logger.info("Talent Intelligence API shutting down")


app = FastAPI(
    title="Talent Intelligence API",
    description="""
## Multi-Agent AI Resume Intelligence Platform
**DA-IICT Hackathon — Problem Statement 9 | Prama Innovations**

### Agents
| Agent | Role |
|-------|------|
| **Parsing Agent** | Extracts structured data from PDF/DOCX/TXT using pdfplumber + Claude LLM |
| **Taxonomy Agent** | Normalizes skills to canonical names, infers implied skills |
| **Matching Agent** | Semantic embedding-based job-candidate scoring with gap analysis |

### Authentication
Pass `X-API-Key: demo-key-hackathon` in all requests (or set `MASTER_API_KEY` in `.env`).

### Quick Start
1. `POST /api/v1/parse` — upload a resume → get structured profile + candidate_id
2. `POST /api/v1/match` — supply candidate_id + job requirements → get match score
3. `GET /api/v1/skills/taxonomy` — browse the 80+ canonical skill taxonomy
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Register all routers ──────────────────────────────────────────────────────
app.include_router(parse_router)
app.include_router(candidates_router)
app.include_router(taxonomy_router)
app.include_router(match_router)


# ── Health endpoints ──────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/api/v1/status", tags=["System"])
async def api_status():
    from api.routes.match import _candidate_store
    return {
        "status":    "running",
        "version":   "1.0.0",
        "agents":    ["parsing", "taxonomy", "matching"],
        "candidates_in_memory": len(_candidate_store),
    }


# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND = Path(__file__).parent / "frontend"

if FRONTEND.exists():
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        ico = FRONTEND / "favicon.ico"
        return FileResponse(str(ico)) if ico.exists() else JSONResponse({})
