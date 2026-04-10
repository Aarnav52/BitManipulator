import io
import json
import re
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime
import pdfplumber
import fitz # PyMuPDF
import google.generativeai as genai
from docx import Document as DocxDocument

from core.config import get_settings
from models.resume import (
    ParsedResume, WorkExperience, Education, ExtractedSkill,
    Certification, Project, FileFormat, ProficiencyLevel,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# --- SYSTEM PROMPT (Strict JSON) ---
EXTRACTION_SYSTEM_PROMPT = """You are an expert resume parser. Extract structured information into JSON.
Return ONLY valid JSON. No markdown backticks.

Structure:
{
  "full_name": "",
  "email": "",
  "phone": "",
  "location": "",
  "linkedin_url": "",
  "github_url": "",
  "portfolio_url": "",
  "summary": "",
  "total_years_experience": 0,
  "confidence_score": 0.95,
  "work_experience": [
    {
      "company": "",
      "role": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "responsibilities": [],
      "technologies": []
    }
  ],
  "education": [],
  "skills": [
    {
      "raw_name": "",
      "proficiency": null,
      "years_experience": 0,
      "context_snippet": ""
    }
  ],
  "certifications": [],
  "projects": []
}"""

def _clean_llm_json(raw: str) -> str:
    raw = raw.strip()
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        return raw[start:end+1]
    return raw

def structure_with_llm(raw_text: str) -> Tuple[dict, List[str]]:
    warnings: List[str] = []

    api_key = getattr(settings, 'google_api_key', None)
    
    if not api_key or "your_" in api_key:
        print("❌ CRITICAL: No Google API Key found in settings!")
        return _regex_fallback(raw_text), ["Missing API Key"]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        full_prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nTEXT TO PARSE:\n{raw_text[:12000]}"
        
        print("🚀 Sending to Gemini...")
        response = model.generate_content(full_prompt)
        
        if not response.text:
            print("⚠️ Gemini returned an empty response.")
            return _regex_fallback(raw_text), ["Empty AI response"]

        cleaned = _clean_llm_json(response.text)
        data = json.loads(cleaned)
        print(f"✅ AI Parsed Successfully. Found {len(data.get('skills', []))} skills.")
        return data, warnings

    except Exception as e:
        print(f"❌ Gemini Error: {str(e)}")
        return _regex_fallback(raw_text), [str(e)]

def _regex_fallback(text: str) -> dict:
    print("⚠️ Using Regex Fallback...")
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.I)
    return {
        "full_name": "AI Parsing Failed",
        "email": email_match.group(0) if email_match else "",
        "confidence_score": 0.1,
        "work_experience": [],
        "education": [],
        "skills": []
    }

def build_parsed_resume(structured, source_format, source_filename, raw_text, warnings, parse_start) -> ParsedResume:
    # 1. FIX: Map confidence_score to the correct float
    conf = structured.get("confidence_score", 0.0)
    
    # Standardize Skills
    skills_list = []
    for s in structured.get("skills", []):
        if isinstance(s, dict) and s.get("raw_name"):
            skills_list.append(ExtractedSkill(
                raw_name=str(s["raw_name"]),
                years_experience=s.get("years_experience") or 0.0,
                context_snippet=str(s.get("context_snippet", ""))[:120]
            ))
        elif isinstance(s, str):
            skills_list.append(ExtractedSkill(raw_name=s))

    # Standardize Work Experience
    work_list = []
    for w in structured.get("work_experience", []):
        if isinstance(w, dict):
            work_list.append(WorkExperience(
                company=w.get("company") or "Unknown",
                role=w.get("role") or "Unknown",
                start_date=w.get("start_date") or "",
                end_date=w.get("end_date") or "",
                responsibilities=w.get("responsibilities") or []
            ))

    # 2. FIX: Align field names with resume.py (parsing_confidence)
    return ParsedResume(
        full_name=structured.get("full_name") or "Unknown Candidate",
        email=structured.get("email", ""),
        phone=structured.get("phone", ""),
        location=structured.get("location", ""),
        linkedin_url=structured.get("linkedin_url", ""),
        github_url=structured.get("github_url", ""),
        portfolio_url=structured.get("portfolio_url", ""),
        summary=structured.get("summary", ""),
        total_years_experience=float(structured.get("total_years_experience") or 0.0),
        work_experience=work_list,
        education=structured.get("education", []),
        skills=skills_list,
        certifications=structured.get("certifications", []),
        projects=structured.get("projects", []),
        source_format=source_format,
        source_filename=source_filename,
        raw_text_length=len(raw_text),
        parsing_confidence=float(conf), # CHANGED from parse_confidence to parsing_confidence
        parse_warnings=warnings,
        parsed_at=datetime.utcnow()
    )

class ParsingAgent:
    def parse_file(self, file_bytes: bytes, filename: str) -> ParsedResume:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf": fmt = FileFormat.PDF
        elif ext in [".docx", ".doc"]: fmt = FileFormat.DOCX
        else: fmt = FileFormat.TXT

        raw_text = ""
        try:
            if fmt == FileFormat.PDF:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    raw_text = "\n".join([p.extract_text() or "" for p in pdf.pages])
            elif fmt == FileFormat.DOCX:
                doc = DocxDocument(io.BytesIO(file_bytes))
                raw_text = "\n".join([p.text for p in doc.paragraphs])
            else:
                raw_text = file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            raw_text = f"Extraction error: {str(e)}"

        if len(raw_text.strip()) < 20:
            return build_parsed_resume(_regex_fallback(raw_text), fmt, filename, raw_text, ["Text too short"], 0)

        structured, warnings = structure_with_llm(raw_text)
        return build_parsed_resume(structured, fmt, filename, raw_text, warnings, time.perf_counter())

parsing_agent = ParsingAgent()