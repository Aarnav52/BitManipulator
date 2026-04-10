"""Taxonomy route — GET /api/v1/skills/*"""
from typing import Optional
from fastapi import APIRouter, Query
from agents.taxonomy_agent import SKILL_TAXONOMY, taxonomy_agent

router = APIRouter(prefix="/api/v1/skills", tags=["Skill Taxonomy"])


@router.get("/taxonomy", summary="Browse full skill taxonomy grouped by category")
async def get_taxonomy():
    return {
        "taxonomy":    taxonomy_agent.get_all_categories(),
        "total_skills": len(SKILL_TAXONOMY),
    }


@router.get("/search", summary="Search skills by name or alias")
async def search_skills(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    results = taxonomy_agent.search_taxonomy(q, limit=limit)
    return {"query": q, "total": len(results), "results": results}


@router.get("/{skill_name}", summary="Lookup a single skill")
async def lookup_skill(skill_name: str):
    return taxonomy_agent.lookup_skill(skill_name)
