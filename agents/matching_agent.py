"""
Matching Agent
==============
Semantic skill-to-job matching using sentence-transformers.
Works directly with ParsedResume (normalized skills attached).
"""
from __future__ import annotations
import logging
import numpy as np
from sentence_transformers import SentenceTransformer, util
from models.resume import MatchRequest, MatchResponse, ParsedResume, ProficiencyLevel

logger = logging.getLogger(__name__)

PROFICIENCY_WEIGHTS = {
    ProficiencyLevel.BEGINNER:     0.3,
    ProficiencyLevel.FAMILIAR:     0.5,
    ProficiencyLevel.INTERMEDIATE: 0.7,
    ProficiencyLevel.ADVANCED:     0.9,
    ProficiencyLevel.EXPERT:       1.0,
}

UPSKILLING_MAP = {
    "Kubernetes":          "Study CKA cert; practice locally with minikube or kind",
    "AWS":                 "AWS Solutions Architect Associate — best entry-point cert",
    "Azure":               "AZ-900 fundamentals → AZ-104 Administrator path",
    "GCP":                 "Google Cloud Digital Leader → Associate Cloud Engineer",
    "React":               "Official React docs + Scrimba React course",
    "TypeScript":          "Add TypeScript incrementally to an existing JS project",
    "Deep Learning":       "fast.ai Practical Deep Learning → Hugging Face tutorials",
    "Machine Learning":    "Andrew Ng's ML Specialization on Coursera",
    "LLM":                 "Karpathy Neural Networks series + LangChain hands-on",
    "Docker":              "Play with Docker classroom; containerize a side project",
    "PostgreSQL":          "pgexercises.com + official PostgreSQL docs",
    "CI/CD":               "Set up GitHub Actions on a personal repo first",
    "Python":              "Python.org tutorial → build a real project",
    "Terraform":           "HashiCorp Learn platform — free official tutorials",
    "Apache Kafka":        "Confluent's free Kafka tutorials + local docker setup",
    "Elasticsearch":       "Elastic's free training at learn.elastic.co",
}


class MatchingAgent:
    def __init__(self, threshold: float = 0.52):
        self.threshold = threshold
        logger.info("Loading sentence-transformer model for matching…")
        self._enc = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("MatchingAgent ready.")

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384))
        return self._enc.encode(texts, normalize_embeddings=True)

    def _best_match(self, job_skill: str, cand_embs: np.ndarray, cand_skills: list):
        if cand_embs.shape[0] == 0:
            return None, 0.0
        job_emb = self._enc.encode([job_skill], normalize_embeddings=True)
        sims    = util.cos_sim(job_emb, cand_embs)[0].numpy()
        idx     = int(np.argmax(sims))
        sim     = float(sims[idx])
        return (cand_skills[idx], sim) if sim >= self.threshold else (None, 0.0)

    def _proficiency_weight(self, skill) -> float:
        base = PROFICIENCY_WEIGHTS.get(skill.proficiency, 0.6)
        yrs  = skill.years_experience or 0
        if yrs >= 5:   base = min(1.0, base + 0.10)
        elif yrs >= 2: base = min(1.0, base + 0.05)
        return base

    def _experience_score(self, resume: ParsedResume, req_years: float | None) -> float:
        if not req_years:
            return 1.0
        actual = resume.total_years_experience or 0
        if actual >= req_years:           return 1.0
        if actual >= req_years * 0.75:   return 0.8
        if actual >= req_years * 0.5:    return 0.6
        return max(0.2, actual / req_years)

    def match(self, resume: ParsedResume, req: MatchRequest) -> MatchResponse:
        # Use normalized skills (those with canonical_name) — fall back to all skills
        skills = resume.normalized_skills or resume.skills
        names  = [s.canonical_name or s.raw_name for s in skills]
        embs   = self._encode(names)

        matched, miss_req, miss_pref = [], [], []
        req_scores, pref_scores = [], []

        for sk in req.required_skills:
            m, sim = self._best_match(sk, embs, skills)
            if m:
                req_scores.append(sim * self._proficiency_weight(m))
                canonical = m.canonical_name or m.raw_name
                if canonical not in matched:
                    matched.append(canonical)
            else:
                req_scores.append(0.0)
                miss_req.append(sk)

        for sk in req.preferred_skills:
            m, sim = self._best_match(sk, embs, skills)
            if m:
                pref_scores.append(sim * self._proficiency_weight(m))
                canonical = m.canonical_name or m.raw_name
                if canonical not in matched:
                    matched.append(canonical)
            else:
                pref_scores.append(0.0)
                miss_pref.append(sk)

        req_avg  = float(np.mean(req_scores))  if req_scores  else 1.0
        pref_avg = float(np.mean(pref_scores)) if pref_scores else 1.0
        skill_score = req_avg * 0.70 + pref_avg * 0.30
        exp_score   = self._experience_score(resume, req.min_experience_years)
        overall     = round(min(1.0, skill_score * 0.75 + exp_score * 0.25), 4)

        suggestions = []
        for s in miss_req[:5]:
            hint = UPSKILLING_MAP.get(s, f"Search '{s} tutorial for beginners' on Coursera or YouTube")
            suggestions.append(f"{s}: {hint}")

        total_job_skills = len(req.required_skills) + len(req.preferred_skills)
        explanation = (
            f"Matched {len(matched)}/{total_job_skills} required+preferred skills. "
            f"Skill fit: {skill_score:.0%} · Experience fit: {exp_score:.0%}."
        )
        if miss_req:
            explanation += f" Missing critical: {', '.join(miss_req[:3])}."

        return MatchResponse(
            candidate_id=resume.id,
            job_title=req.job_title,
            overall_score=overall,
            skill_match_score=round(skill_score, 4),
            experience_score=round(exp_score, 4),
            matched_skills=matched,
            missing_required_skills=miss_req,
            missing_preferred_skills=miss_pref,
            upskilling_suggestions=suggestions,
            explanation=explanation,
        )


matching_agent = MatchingAgent()
