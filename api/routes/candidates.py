"""Candidates route — GET skills and full profile."""
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import verify_api_key

router = APIRouter(prefix="/api/v1", tags=["Candidates"])


@router.get("/candidates/{candidate_id}/skills", summary="Get candidate's normalized skill profile")
async def get_candidate_skills(candidate_id: str, _: str = Depends(verify_api_key)):
    from api.routes.match import get_candidate
    
    # This calls the get_candidate function in match.py
    c = get_candidate(candidate_id)
    
    by_cat: dict = {}
    for s in c.skills:
        # FIX 1: Ensure category is never null
        cat = s.category or "Uncategorized"
        
        # FIX 2: Fallback to raw_name if canonical_name is missing.
        # This prevents the "0 skills" error when normalization fails.
        skill_display = s.canonical_name or s.raw_name or "Unknown Skill"
        
        by_cat.setdefault(cat, []).append(skill_display)
        
    return {
        "candidate_id":          c.id,
        # FIX 3: Global fallback for name
        "full_name":             c.full_name or "Unknown Candidate",
        "total_skills":          len(c.skills),
        # FIX 4: Ensure experience is at least 0.0 for the UI
        "total_years_experience": c.total_years_experience or 0.0,
        "seniority_level":       c.seniority_level,
        "skills_by_category":    by_cat,
        "skills":                [s.model_dump() for s in c.skills],
    }


@router.get("/candidates/{candidate_id}", response_model=None, summary="Get full candidate profile")
async def get_candidate_full(candidate_id: str, _: str = Depends(verify_api_key)):
    from api.routes.match import get_candidate
    c = get_candidate(candidate_id)
    
    # We return the object, but Pydantic handles the serialization
    return c