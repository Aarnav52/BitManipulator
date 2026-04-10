"""
Unified data models for the Talent Intelligence platform.
Single source of truth used by all agents and API routes.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class FileFormat(str, Enum):
    PDF  = "pdf"
    DOCX = "docx"
    TXT  = "txt"
    UNKNOWN = "unknown"

class ProficiencyLevel(str, Enum):
    BEGINNER     = "beginner"
    FAMILIAR     = "familiar"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"
    EXPERT       = "expert"


# ── Sub-models (used by parsing agent) ───────────────────────────────────────

class WorkExperience(BaseModel):
    company:          str = ""
    role:             str = ""
    start_date:       Optional[str] = None
    end_date:         Optional[str] = None
    duration_months:  Optional[int] = None
    responsibilities: List[str] = Field(default_factory=list)
    technologies:     List[str] = Field(default_factory=list)

class Education(BaseModel):
    institution: str = ""
    degree:      Optional[str] = None
    field:       Optional[str] = None
    year:        Optional[int] = None
    gpa:         Optional[float] = None

class ExtractedSkill(BaseModel):
    raw_name:         str = ""
    canonical_name:   Optional[str] = None
    category:         Optional[str] = None
    proficiency:      Optional[ProficiencyLevel] = None
    years_experience: Optional[float] = 0.0 # Default to 0.0
    context_snippet:  Optional[str] = None

class Certification(BaseModel):
    name:        str = ""
    issuer:      Optional[str] = None
    year:        Optional[int] = None
    expiry_year: Optional[int] = None

class Project(BaseModel):
    name:         str = ""
    description:  Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    url:          Optional[str] = None


# ── ParsedResume (output of Parsing Agent) ────────────────────────────────────

class ParsedResume(BaseModel):
    id:                   str = Field(default_factory=lambda: str(uuid4()))
    full_name:            Optional[str] = "Unknown Candidate" # Default for UI
    email:                Optional[str] = None
    phone:                Optional[str] = None
    location:             Optional[str] = None
    linkedin_url:         Optional[str] = None
    github_url:           Optional[str] = None
    portfolio_url:        Optional[str] = None
    summary:              str = ""
    
    # FIX: Default to 0.0 so the UI doesn't show "null"
    total_years_experience: Optional[float] = 0.0 
    
    work_experience:      List[WorkExperience] = Field(default_factory=list)
    education:            List[Education] = Field(default_factory=list)
    skills:               List[ExtractedSkill] = Field(default_factory=list)
    certifications:       List[Certification] = Field(default_factory=list)
    projects:             List[Project] = Field(default_factory=list)
    publications:         List[str] = Field(default_factory=list)
    languages:            List[str] = Field(default_factory=list)
    source_format:        FileFormat = FileFormat.UNKNOWN
    source_filename:      str = ""
    raw_text_length:      int = 0
    
    # FIX: Ensure this name matches what is used in match.py and parsing_agent.py
    parsing_confidence:   float = 0.0 
    
    parse_warnings:       List[str] = Field(default_factory=list)
    parsed_at:            datetime = Field(default_factory=datetime.utcnow)

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def seniority_level(self) -> str:
        yrs = self.total_years_experience or 0
        if yrs < 1:   return "Intern / Fresher"
        if yrs < 3:   return "Junior"
        if yrs < 6:   return "Mid-level"
        if yrs < 10:  return "Senior"
        return "Lead / Principal"

    @property
    def normalized_skills(self) -> List[ExtractedSkill]:
        """Skills that have been normalized (have a canonical_name)."""
        return [s for s in self.skills if s.canonical_name]


# ── API Request / Response models ─────────────────────────────────────────────

class ParseRequest(BaseModel):
    text:     str   = Field(..., description="Plain-text resume content")
    filename: str   = Field(default="resume.txt")

class BatchParseResponse(BaseModel):
    job_id:      str
    status:      str
    total_files: int
    message:     str

class BatchJobStatus(BaseModel):
    job_id:       str
    status:       str
    total:        int
    completed:    int
    failed_count: int
    results:      List[ParsedResume]
    created_at:   datetime
    completed_at: Optional[datetime] = None

class MatchRequest(BaseModel):
    candidate_id:         str   = Field(..., description="ID from a previous /parse call")
    job_title:            str
    job_description:      str   = ""
    required_skills:      List[str] = Field(default_factory=list)
    preferred_skills:     List[str] = Field(default_factory=list)
    min_experience_years: Optional[float] = None
    company:              Optional[str] = None

class MatchResponse(BaseModel):
    candidate_id:              str
    job_title:                 str
    overall_score:             float
    skill_match_score:         float
    experience_score:          float
    matched_skills:            List[str] = Field(default_factory=list)
    missing_required_skills:   List[str] = Field(default_factory=list)
    missing_preferred_skills:  List[str] = Field(default_factory=list)
    upskilling_suggestions:    List[str] = Field(default_factory=list)
    explanation:               str = ""
    matched_at:                datetime = Field(default_factory=datetime.utcnow)