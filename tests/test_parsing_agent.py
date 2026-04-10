"""Tests for the Parsing Agent — pytest tests/test_parsing_agent.py -v"""
import pytest
from unittest.mock import patch, MagicMock
from agents.parsing_agent import (
    detect_format, extract_text_from_txt, _regex_fallback, ParsingAgent
)
from models.resume import FileFormat, ParsedResume

SAMPLE_TEXT = """
Jane Smith
jane.smith@email.com | +1-555-0101 | New York, NY
linkedin.com/in/janesmith | github.com/janesmith

EXPERIENCE
Senior Engineer — TechCorp (Jan 2020 – Present)
- Built APIs with Python and FastAPI
- Deployed services on AWS with Docker and Kubernetes

Engineer — StartupABC (Jun 2017 – Dec 2019)
- React + Node.js full-stack development

EDUCATION
B.S. Computer Science — MIT, 2017

SKILLS
Python, FastAPI, React, Docker, Kubernetes, AWS, PostgreSQL

CERTIFICATIONS
AWS Solutions Architect — Amazon, 2021
"""


class TestFormatDetection:
    def test_pdf_extension(self):
        assert detect_format("resume.pdf", b"%PDF") == FileFormat.PDF

    def test_pdf_magic_bytes(self):
        assert detect_format("file", b"%PDF-1.4 content") == FileFormat.PDF

    def test_docx_extension(self):
        assert detect_format("resume.docx", b"PK") == FileFormat.DOCX

    def test_docx_magic_bytes(self):
        assert detect_format("file.xyz", b"PK\x03\x04") == FileFormat.DOCX

    def test_txt_extension(self):
        assert detect_format("resume.txt", b"hello") == FileFormat.TXT

    def test_unknown_defaults_to_txt(self):
        # non-PDF/DOCX magic bytes fall back to TXT
        assert detect_format("file.xyz", b"random data") == FileFormat.TXT


class TestTextExtraction:
    def test_utf8(self):
        text, warnings = extract_text_from_txt("Hello World".encode("utf-8"))
        assert "Hello World" in text
        assert warnings == []

    def test_latin1(self):
        text, _ = extract_text_from_txt("Héllo Wörld".encode("latin-1"))
        assert len(text) > 0

    def test_empty(self):
        text, _ = extract_text_from_txt(b"")
        assert text == ""


class TestRegexFallback:
    def test_extracts_email(self):
        result = _regex_fallback(SAMPLE_TEXT)
        assert result["email"] == "jane.smith@email.com"

    def test_extracts_linkedin(self):
        result = _regex_fallback(SAMPLE_TEXT)
        assert "janesmith" in result.get("linkedin_url", "")

    def test_extracts_github(self):
        result = _regex_fallback(SAMPLE_TEXT)
        assert "janesmith" in result.get("github_url", "")

    def test_extracts_skills(self):
        result = _regex_fallback(SAMPLE_TEXT)
        skill_names = [s["raw_name"] for s in result.get("skills", [])]
        assert len(skill_names) > 0

    def test_returns_dict(self):
        result = _regex_fallback(SAMPLE_TEXT)
        required_keys = ["full_name","email","work_experience","education","skills","certifications"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestParsingAgent:
    def test_parse_txt_no_api_key(self):
        """Without API key, should fall back to regex and still return ParsedResume."""
        with patch("agents.parsing_agent.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_max_tokens = 4096
            agent = ParsingAgent()
            result = agent.parse_file(SAMPLE_TEXT.encode("utf-8"), "test.txt")
        assert isinstance(result, ParsedResume)
        assert result.source_format == FileFormat.TXT
        assert result.email == "jane.smith@email.com"
        assert len(result.skills) > 0

    def test_parse_text_method(self):
        with patch("agents.parsing_agent.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_max_tokens = 4096
            agent = ParsingAgent()
            result = agent.parse_text(SAMPLE_TEXT)
        assert isinstance(result, ParsedResume)
        assert result.id  # UUID generated

    def test_empty_file_returns_safe_result(self):
        with patch("agents.parsing_agent.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_max_tokens = 4096
            agent = ParsingAgent()
            result = agent.parse_file(b"", "empty.txt")
        assert isinstance(result, ParsedResume)
        # Should not raise — empty content is handled gracefully

    def test_confidence_increases_with_more_fields(self):
        with patch("agents.parsing_agent.settings") as ms:
            ms.anthropic_api_key = ""
            ms.llm_model = "claude-sonnet-4-20250514"
            ms.llm_max_tokens = 4096
            agent = ParsingAgent()
            rich   = agent.parse_file(SAMPLE_TEXT.encode(), "rich.txt")
            sparse = agent.parse_file(b"No information here", "sparse.txt")
        assert rich.parse_confidence >= sparse.parse_confidence
