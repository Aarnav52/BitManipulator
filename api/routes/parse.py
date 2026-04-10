"""Parse routes — POST /api/v1/parse, /api/v1/parse/text, /api/v1/parse/batch"""
import uuid
import time
from typing import List
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Depends, BackgroundTasks, HTTPException
from api.dependencies import verify_api_key, validate_resume_file
from models.resume import ParsedResume, ParseRequest, BatchParseResponse, BatchJobStatus

router = APIRouter(prefix="/api/v1/parse", tags=["Resume Parsing"])

# Shared in-memory store for batch jobs
_batch_jobs: dict = {}


@router.post("", response_model=ParsedResume, summary="Parse a single resume file")
async def parse_resume(
    file: UploadFile = File(..., description="PDF, DOCX, or TXT resume"),
    _: str = Depends(verify_api_key),
):
    """Upload a resume and receive a fully structured, skill-normalized candidate profile."""
    from agents.orchestrator import orchestrator
    content = await validate_resume_file(file)
    # Orchestrator handles registration in the candidate store
    return orchestrator.process_resume(content, file.filename or "resume")


@router.post("/text", response_model=ParsedResume, summary="Parse plain-text resume")
async def parse_text_resume(body: ParseRequest, _: str = Depends(verify_api_key)):
    """Submit raw resume text (no file upload)."""
    from agents.orchestrator import orchestrator
    return orchestrator.process_resume(body.text.encode("utf-8"), body.filename)


@router.post("/batch", response_model=BatchParseResponse, status_code=202, summary="Batch parse (async)")
async def batch_parse(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    _: str = Depends(verify_api_key),
):
    """Upload up to 50 resumes. Returns a job_id — poll GET /api/v1/parse/batch/{job_id}."""
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch")
    
    job_id = str(uuid.uuid4())
    file_data = []
    
    # Read files into memory before background task starts
    for f in files:
        content = await f.read()
        file_data.append((content, f.filename or "resume"))

    _batch_jobs[job_id] = {
        "status": "queued", 
        "total": len(file_data),
        "completed": 0, 
        "failed_count": 0, 
        "results": [],
        "created_at": datetime.utcnow(), # FIX: Use datetime objects
        "completed_at": None,
    }
    
    background_tasks.add_task(_process_batch, job_id, file_data)
    
    return BatchParseResponse(
        job_id=job_id, 
        status="queued", 
        total_files=len(file_data),
        message=f"Batch queued ({len(file_data)} files). Poll GET /api/v1/parse/batch/{job_id}.",
    )


@router.get("/batch/{job_id}", response_model=BatchJobStatus, summary="Poll batch job status")
async def get_batch_status(job_id: str, _: str = Depends(verify_api_key)):
    if job_id not in _batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    j = _batch_jobs[job_id]
    
    # FIX: No need to convert from timestamp if we store as datetime
    return BatchJobStatus(
        job_id=job_id, 
        status=j["status"], 
        total=j["total"],
        completed=j["completed"], 
        failed_count=j["failed_count"], 
        results=j["results"],
        created_at=j["created_at"],
        completed_at=j["completed_at"],
    )


async def _process_batch(job_id: str, file_data: list):
    from agents.orchestrator import orchestrator
    j = _batch_jobs[job_id]
    j["status"] = "processing"
    
    for content, filename in file_data:
        try:
            # FIX: Ensure result is a valid ParsedResume object
            result = orchestrator.process_resume(content, filename)
            j["results"].append(result)
        except Exception as e:
            print(f"❌ Batch Error for {filename}: {str(e)}")
            j["failed_count"] += 1
        finally:
            j["completed"] += 1
    
    j["status"] = "done"
    j["completed_at"] = datetime.utcnow() # FIX: Use datetime objects