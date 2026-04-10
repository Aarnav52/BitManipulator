"""Tests for the Taxonomy Agent — pytest tests/test_taxonomy_agent.py -v"""
import pytest
from agents.taxonomy_agent import normalize_skill, infer_implied_skills, TaxonomyAgent, SKILL_TAXONOMY
from models.resume import ParsedResume, ExtractedSkill, FileFormat


class TestNormalizeSkill:
    def test_exact_canonical(self):
        canonical, cat, parent = normalize_skill("Python")
        assert canonical == "Python"
        assert cat == "Programming Languages"

    def test_alias_k8s(self):
        canonical, _, _ = normalize_skill("k8s")
        assert canonical == "Kubernetes"

    def test_alias_js(self):
        canonical, _, _ = normalize_skill("js")
        assert canonical == "JavaScript"

    def test_alias_sklearn(self):
        canonical, _, _ = normalize_skill("sklearn")
        assert canonical == "scikit-learn"

    def test_alias_tf(self):
        canonical, _, _ = normalize_skill("tf")
        assert canonical == "TensorFlow"

    def test_case_insensitive(self):
        canonical, _, _ = normalize_skill("PYTHON")
        assert canonical == "Python"

    def test_unknown_returns_none(self):
        canonical, cat, parent = normalize_skill("CobolMagic9000XYZ")
        assert canonical is None
        assert cat is None

    def test_fuzzy_pytorch(self):
        # "Pytorch" (capital P, different case) should fuzzy-match
        canonical, _, _ = normalize_skill("Pytorch")
        assert canonical == "PyTorch"


class TestInferImplied:
    def test_pytorch_implies_deep_learning(self):
        implied = infer_implied_skills("PyTorch")
        assert "Deep Learning" in implied

    def test_pytorch_implies_python(self):
        implied = infer_implied_skills("PyTorch")
        assert "Python" in implied

    def test_typescript_implies_javascript(self):
        implied = infer_implied_skills("TypeScript")
        assert "JavaScript" in implied

    def test_kubernetes_implies_docker(self):
        implied = infer_implied_skills("Kubernetes")
        assert "Docker" in implied

    def test_unknown_returns_empty(self):
        assert infer_implied_skills("CobolMagic") == []


class TestTaxonomyAgent:
    def setup_method(self):
        self.agent = TaxonomyAgent()

    def test_normalize_resume_canonical_names(self):
        resume = ParsedResume(
            source_format=FileFormat.TXT,
            skills=[
                ExtractedSkill(raw_name="pytorch"),
                ExtractedSkill(raw_name="k8s"),
                ExtractedSkill(raw_name="reactjs"),
            ]
        )
        result = self.agent.normalize_resume(resume)
        canonical_names = [s.canonical_name for s in result.skills if s.canonical_name]
        assert "PyTorch" in canonical_names
        assert "Kubernetes" in canonical_names
        assert "React" in canonical_names

    def test_normalize_adds_implied_skills(self):
        resume = ParsedResume(
            source_format=FileFormat.TXT,
            skills=[ExtractedSkill(raw_name="PyTorch")]
        )
        result = self.agent.normalize_resume(resume)
        canonical_names = [s.canonical_name for s in result.skills if s.canonical_name]
        assert "Deep Learning" in canonical_names
        assert "Python" in canonical_names

    def test_no_duplicate_canonicals(self):
        resume = ParsedResume(
            source_format=FileFormat.TXT,
            skills=[
                ExtractedSkill(raw_name="Python"),
                ExtractedSkill(raw_name="python"),   # duplicate
                ExtractedSkill(raw_name="py"),        # alias
            ]
        )
        result = self.agent.normalize_resume(resume)
        python_skills = [s for s in result.skills if s.canonical_name == "Python"]
        assert len(python_skills) == 1

    def test_search_taxonomy(self):
        results = self.agent.search_taxonomy("python")
        assert len(results) > 0
        assert any(r["canonical_name"] == "Python" for r in results)

    def test_get_all_categories(self):
        cats = self.agent.get_all_categories()
        assert "Programming Languages" in cats
        assert "AI/ML" in cats
        assert "DevOps" in cats
        assert len(cats) > 5

    def test_lookup_known_skill(self):
        result = self.agent.lookup_skill("Docker")
        assert result["found"] is True
        assert result["canonical_name"] == "Docker"

    def test_lookup_unknown_skill(self):
        result = self.agent.lookup_skill("CobolMagic9000")
        assert result["found"] is False

    def test_taxonomy_has_minimum_skills(self):
        assert len(SKILL_TAXONOMY) >= 50
