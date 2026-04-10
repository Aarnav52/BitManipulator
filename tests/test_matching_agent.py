"""Tests for the Matching Agent — pytest tests/test_matching_agent.py -v"""
import pytest
from agents.matching_agent import MatchingAgent
from agents.taxonomy_agent import TaxonomyAgent
from models.resume import ParsedResume, ExtractedSkill, MatchRequest, FileFormat, ProficiencyLevel


def make_resume(skill_names: list[str], exp_years: float = 4.0) -> ParsedResume:
    """Create a ParsedResume with normalized skills for testing."""
    agent = TaxonomyAgent()
    resume = ParsedResume(
        source_format=FileFormat.TXT,
        full_name="Test Candidate",
        email="test@example.com",
        total_years_experience=exp_years,
        skills=[ExtractedSkill(raw_name=s, proficiency=ProficiencyLevel.ADVANCED) for s in skill_names],
    )
    return agent.normalize_resume(resume)


@pytest.fixture(scope="module")
def matcher():
    return MatchingAgent()


class TestMatchingAgent:
    def test_perfect_match_high_score(self, matcher):
        resume = make_resume(["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS"])
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Senior Backend Engineer",
            job_description="Python FastAPI PostgreSQL Docker",
            required_skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
            preferred_skills=["Kubernetes", "AWS"],
            min_experience_years=3.0,
        )
        result = matcher.match(resume, req)
        assert result.overall_score >= 0.75
        assert len(result.matched_skills) >= 4
        assert result.missing_required_skills == []

    def test_no_skills_low_score(self, matcher):
        resume = make_resume(["Excel"])
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Python Developer",
            job_description="Python Django PostgreSQL",
            required_skills=["Python", "Django", "PostgreSQL"],
            min_experience_years=2.0,
        )
        result = matcher.match(resume, req)
        assert result.overall_score < 0.5
        assert len(result.missing_required_skills) > 0

    def test_partial_match(self, matcher):
        resume = make_resume(["Python", "Django"])
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Backend Engineer",
            job_description="Python Django AWS Kubernetes",
            required_skills=["Python", "Django", "AWS", "Kubernetes"],
        )
        result = matcher.match(resume, req)
        assert 0.0 < result.overall_score < 1.0
        assert "Python" in result.matched_skills or "Django" in result.matched_skills
        assert len(result.missing_required_skills) > 0

    def test_experience_score_sufficient(self, matcher):
        resume = make_resume(["Python"], exp_years=5.0)
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Senior Dev",
            job_description="Python developer",
            required_skills=["Python"],
            min_experience_years=3.0,
        )
        result = matcher.match(resume, req)
        assert result.experience_score == 1.0

    def test_experience_score_insufficient(self, matcher):
        resume = make_resume(["Python"], exp_years=1.0)
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Senior Dev",
            job_description="Python developer",
            required_skills=["Python"],
            min_experience_years=5.0,
        )
        result = matcher.match(resume, req)
        assert result.experience_score < 1.0

    def test_no_required_skills_full_exp_score(self, matcher):
        resume = make_resume(["Python"])
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Any Dev",
            job_description="Open role",
        )
        result = matcher.match(resume, req)
        assert result.experience_score == 1.0

    def test_upskilling_suggestions_for_missing(self, matcher):
        resume = make_resume(["Python"])
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Cloud Engineer",
            job_description="AWS Kubernetes Docker",
            required_skills=["AWS", "Kubernetes", "Docker"],
        )
        result = matcher.match(resume, req)
        assert len(result.upskilling_suggestions) > 0
        # Each suggestion should reference the missing skill
        combined = " ".join(result.upskilling_suggestions)
        assert "AWS" in combined or "Kubernetes" in combined or "Docker" in combined

    def test_scores_in_valid_range(self, matcher):
        resume = make_resume(["Python", "React", "Docker"])
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Full Stack Engineer",
            job_description="Python React Docker AWS",
            required_skills=["Python", "React", "Docker", "AWS"],
        )
        result = matcher.match(resume, req)
        for score in [result.overall_score, result.skill_match_score, result.experience_score]:
            assert 0.0 <= score <= 1.0

    def test_semantic_matching_alias(self, matcher):
        """'ReactJS' in job desc should still match 'React' in candidate skills."""
        resume = make_resume(["React", "Python"])
        req = MatchRequest(
            candidate_id=resume.id,
            job_title="Frontend Dev",
            job_description="ReactJS and Python developer",
            required_skills=["ReactJS", "Python"],
        )
        result = matcher.match(resume, req)
        # Semantic similarity should catch ReactJS ≈ React
        assert result.skill_match_score > 0.5
