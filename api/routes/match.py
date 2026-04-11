"""
Match route — POST /api/v1/match
Candidate store — shared dict populated by parse routes
"""
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import verify_api_key
from models.resume import MatchRequest, MatchResponse, ParsedResume

router = APIRouter(prefix="/api/v1", tags=["Matching"])

# Shared in-memory store: candidate_id (str) -> ParsedResume
_candidate_store: dict[str, ParsedResume] = {}


def register_candidate(candidate_id: str, resume: ParsedResume):
    """Called by parse routes + orchestrator to register a parsed resume."""
    _candidate_store[candidate_id] = resume


def get_candidate(candidate_id: str) -> ParsedResume:
    if candidate_id not in _candidate_store:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate '{candidate_id}' not found. Parse a resume first via POST /api/v1/parse.",
        )
    return _candidate_store[candidate_id]


@router.post("/match", response_model=MatchResponse, summary="Match candidate to a job description")
async def match_candidate(request: MatchRequest, _: str = Depends(verify_api_key)):
    """
    Semantic skill-fit scoring between a parsed candidate and a job description.
    Returns overall score, matched/missing skills, and upskilling suggestions.
    """
    from agents.matching_agent import matching_agent
    resume = get_candidate(request.candidate_id)
    return matching_agent.match(resume, request)


@router.get("/candidates", summary="List all parsed candidates")
async def list_candidates(_: str = Depends(verify_api_key)):
    return {
        "total": len(_candidate_store),
        "candidates": [
            {
                "id":                  c.id,
                "name":                c.full_name or "Unknown Candidate",
                "email":               c.email or "N/A",
                "location":            c.location or "N/A",
                "total_skills":        len(c.skills) if c.skills else 0,
                "experience_years":    c.total_years_experience or 0.0,
                "seniority":           c.seniority_level or "Junior",
                
                "parse_confidence":    c.parsing_confidence or 0.0,
                
                "parsed_at":           c.parsed_at.isoformat() if c.parsed_at else None,
            }
            for c in _candidate_store.values()
        ],
    }