"""
Taxonomy & Normalization Agent
==============================
Maps raw extracted skill names to canonical entries in a hierarchical skill taxonomy.
Handles synonyms, abbreviations, and skill hierarchy inference.
"""

import json
import logging
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from models.resume import ParsedResume, ExtractedSkill, ProficiencyLevel

logger = logging.getLogger(__name__)

# ── Built-in skill taxonomy ───────────────────────────────────────────────────
# Format: canonical_name -> {"category": str, "parent": str, "aliases": [str], "implies": [str]}

SKILL_TAXONOMY: Dict[str, dict] = {
    # --- Programming Languages ---
    "Python":       {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["py", "python3", "python 3"], "implies": []},
    "JavaScript":   {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["js", "javascript", "ecmascript", "es6", "es2015"], "implies": []},
    "TypeScript":   {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["ts", "typescript"], "implies": ["JavaScript"]},
    "Java":         {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["java"], "implies": []},
    "C++":          {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["cpp", "c plus plus", "c/c++"], "implies": []},
    "C#":           {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["csharp", "c sharp", "dotnet c#"], "implies": []},
    "Go":           {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["golang", "go lang"], "implies": []},
    "Rust":         {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["rust lang", "rust-lang"], "implies": []},
    "Ruby":         {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["ruby", "ruby on rails language"], "implies": []},
    "PHP":          {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["php", "php8", "php7"], "implies": []},
    "Swift":        {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["swift", "swiftui language"], "implies": []},
    "Kotlin":       {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["kotlin"], "implies": []},
    "Scala":        {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["scala"], "implies": []},
    "R":            {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["r language", "r programming"], "implies": []},
    "SQL":          {"category": "Programming Languages", "parent": "Technical Skills", "aliases": ["structured query language", "sql queries"], "implies": []},

    # --- Web Frameworks ---
    "React":        {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["reactjs", "react.js", "react js"], "implies": ["JavaScript"]},
    "Angular":      {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["angularjs", "angular.js", "angular js"], "implies": ["TypeScript"]},
    "Vue.js":       {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["vue", "vuejs", "vue.js"], "implies": ["JavaScript"]},
    "Next.js":      {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["nextjs", "next js"], "implies": ["React"]},
    "Django":       {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["django"], "implies": ["Python"]},
    "FastAPI":      {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["fast api"], "implies": ["Python"]},
    "Flask":        {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["flask"], "implies": ["Python"]},
    "Spring Boot":  {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["springboot", "spring-boot", "spring"], "implies": ["Java"]},
    "Ruby on Rails":{"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["rails", "ror"], "implies": ["Ruby"]},
    "Express.js":   {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["express", "expressjs"], "implies": ["JavaScript"]},
    "Node.js":      {"category": "Web Frameworks", "parent": "Technical Skills", "aliases": ["node", "nodejs", "node js"], "implies": ["JavaScript"]},

    # --- Machine Learning & AI ---
    "Machine Learning":     {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["ml", "machine-learning"], "implies": []},
    "Deep Learning":        {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["dl", "deep-learning", "neural networks"], "implies": ["Machine Learning"]},
    "TensorFlow":           {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["tensorflow", "tf"], "implies": ["Deep Learning", "Python"]},
    "PyTorch":              {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["pytorch", "torch"], "implies": ["Deep Learning", "Python"]},
    "scikit-learn":         {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["sklearn", "scikit learn", "scikitlearn"], "implies": ["Machine Learning", "Python"]},
    "Hugging Face":         {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["huggingface", "transformers library"], "implies": ["Deep Learning"]},
    "LangChain":            {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["langchain"], "implies": ["Python"]},
    "Natural Language Processing": {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["nlp", "text mining", "computational linguistics"], "implies": ["Machine Learning"]},
    "Computer Vision":      {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["cv", "image processing", "opencv"], "implies": ["Machine Learning"]},
    "LLM":                  {"category": "AI/ML", "parent": "Technical Skills", "aliases": ["large language model", "llms", "gpt", "generative ai"], "implies": ["Deep Learning"]},

    # --- Data & Databases ---
    "PostgreSQL":   {"category": "Databases", "parent": "Technical Skills", "aliases": ["postgres", "psql", "postgresql"], "implies": ["SQL"]},
    "MySQL":        {"category": "Databases", "parent": "Technical Skills", "aliases": ["mysql", "my sql"], "implies": ["SQL"]},
    "MongoDB":      {"category": "Databases", "parent": "Technical Skills", "aliases": ["mongo", "mongodb"], "implies": []},
    "Redis":        {"category": "Databases", "parent": "Technical Skills", "aliases": ["redis"], "implies": []},
    "Elasticsearch":{"category": "Databases", "parent": "Technical Skills", "aliases": ["elastic", "elk", "opensearch"], "implies": []},
    "Apache Kafka": {"category": "Data Engineering", "parent": "Technical Skills", "aliases": ["kafka"], "implies": []},
    "Apache Spark": {"category": "Data Engineering", "parent": "Technical Skills", "aliases": ["spark", "pyspark"], "implies": []},
    "Pandas":       {"category": "Data Engineering", "parent": "Technical Skills", "aliases": ["pandas"], "implies": ["Python"]},
    "NumPy":        {"category": "Data Engineering", "parent": "Technical Skills", "aliases": ["numpy"], "implies": ["Python"]},

    # --- Cloud & DevOps ---
    "AWS":          {"category": "Cloud Platforms", "parent": "Technical Skills", "aliases": ["amazon web services", "amazon aws"], "implies": []},
    "Azure":        {"category": "Cloud Platforms", "parent": "Technical Skills", "aliases": ["microsoft azure", "azure cloud"], "implies": []},
    "GCP":          {"category": "Cloud Platforms", "parent": "Technical Skills", "aliases": ["google cloud", "google cloud platform"], "implies": []},
    "Docker":       {"category": "DevOps", "parent": "Technical Skills", "aliases": ["docker", "docker container"], "implies": []},
    "Kubernetes":   {"category": "DevOps", "parent": "Technical Skills", "aliases": ["k8s", "kube", "k8"], "implies": ["Docker"]},
    "Terraform":    {"category": "DevOps", "parent": "Technical Skills", "aliases": ["terraform", "tf iac"], "implies": []},
    "CI/CD":        {"category": "DevOps", "parent": "Technical Skills", "aliases": ["cicd", "continuous integration", "continuous deployment", "github actions", "jenkins", "gitlab ci"], "implies": []},
    "Git":          {"category": "DevOps", "parent": "Technical Skills", "aliases": ["git", "version control", "github", "gitlab", "bitbucket"], "implies": []},

    # --- Soft Skills ---
    "Leadership":         {"category": "Soft Skills", "parent": "Soft Skills", "aliases": ["team lead", "team leadership", "people management"], "implies": []},
    "Communication":      {"category": "Soft Skills", "parent": "Soft Skills", "aliases": ["written communication", "verbal communication", "presentation skills"], "implies": []},
    "Problem Solving":    {"category": "Soft Skills", "parent": "Soft Skills", "aliases": ["analytical thinking", "critical thinking", "troubleshooting"], "implies": []},
    "Agile":              {"category": "Methodologies", "parent": "Soft Skills", "aliases": ["scrum", "kanban", "agile methodology", "sprint"], "implies": []},
    "Project Management": {"category": "Methodologies", "parent": "Soft Skills", "aliases": ["pm", "project manager", "program management"], "implies": []},
}

# Build reverse alias lookup: alias_lower -> canonical_name
_ALIAS_INDEX: Dict[str, str] = {}
for canonical, meta in SKILL_TAXONOMY.items():
    _ALIAS_INDEX[canonical.lower()] = canonical
    for alias in meta.get("aliases", []):
        _ALIAS_INDEX[alias.lower()] = canonical


def _fuzzy_match(raw: str, threshold: float = 0.82) -> Optional[str]:
    """Find best fuzzy match in alias index."""
    raw_l = raw.lower()
    best_score = 0.0
    best_match = None
    for alias in _ALIAS_INDEX:
        score = SequenceMatcher(None, raw_l, alias).ratio()
        if score > best_score:
            best_score = score
            best_match = alias
    if best_score >= threshold and best_match:
        return _ALIAS_INDEX[best_match]
    return None


def normalize_skill(raw_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (canonical_name, category, parent) or (None, None, None) if no match.
    """
    raw_l = raw_name.strip().lower()
    if not raw_l:
        return None, None, None

    # 1. Exact alias match
    if raw_l in _ALIAS_INDEX:
        canonical = _ALIAS_INDEX[raw_l]
        meta = SKILL_TAXONOMY[canonical]
        return canonical, meta["category"], meta["parent"]

    # 2. Fuzzy match
    canonical = _fuzzy_match(raw_l)
    if canonical:
        meta = SKILL_TAXONOMY[canonical]
        return canonical, meta["category"], meta["parent"]

    return None, None, None


def infer_implied_skills(canonical_name: str) -> List[str]:
    """Return skills implied by knowing this skill (e.g., TypeScript implies JavaScript)."""
    if canonical_name in SKILL_TAXONOMY:
        return SKILL_TAXONOMY[canonical_name].get("implies", [])
    return []


class TaxonomyAgent:
    """
    Normalizes all skills on a ParsedResume and adds implied skills.
    """

    def normalize_resume(self, resume: ParsedResume) -> ParsedResume:
        """
        In-place normalize all skills and inject implied ones.
        Returns the same resume object (mutated).
        """
        normalized: List[ExtractedSkill] = []
        seen_canonicals: set = set()
        unknown_skills: List[str] = []

        for skill in resume.skills:
            canonical, category, parent = normalize_skill(skill.raw_name)
            if canonical:
                skill.canonical_name = canonical
                skill.category = category
                seen_canonicals.add(canonical)
            else:
                unknown_skills.append(skill.raw_name)
            normalized.append(skill)

        # Inject implied skills (not already present)
        implied_added: List[ExtractedSkill] = []
        for canonical in list(seen_canonicals):
            for implied in infer_implied_skills(canonical):
                if implied not in seen_canonicals:
                    meta = SKILL_TAXONOMY.get(implied, {})
                    implied_added.append(ExtractedSkill(
                        raw_name=f"{implied} (inferred from {canonical})",
                        canonical_name=implied,
                        category=meta.get("category"),
                        proficiency=ProficiencyLevel.INTERMEDIATE,
                        context_snippet=f"Inferred from {canonical} experience",
                    ))
                    seen_canonicals.add(implied)

        resume.skills = normalized + implied_added

        if unknown_skills:
            resume.parse_warnings.append(
                f"Unrecognized skills (not in taxonomy): {', '.join(unknown_skills[:10])}"
            )

        logger.info("Skills normalized", extra={
            "candidate_id": resume.id,
            "total": len(resume.skills),
            "normalized": len(normalized),
            "implied": len(implied_added),
            "unknown": len(unknown_skills),
        })
        return resume

    def lookup_skill(self, raw_name: str) -> dict:
        """Public lookup for taxonomy browsing."""
        canonical, category, parent = normalize_skill(raw_name)
        if canonical:
            meta = SKILL_TAXONOMY[canonical]
            return {
                "raw_input": raw_name,
                "canonical_name": canonical,
                "category": category,
                "parent": parent,
                "aliases": meta.get("aliases", []),
                "implies": meta.get("implies", []),
                "found": True,
            }
        return {"raw_input": raw_name, "found": False}

    def search_taxonomy(self, query: str, limit: int = 20) -> List[dict]:
        """Search taxonomy by partial name match."""
        query_l = query.lower()
        results = []
        for canonical, meta in SKILL_TAXONOMY.items():
            if query_l in canonical.lower() or any(query_l in a for a in meta.get("aliases", [])):
                results.append({
                    "canonical_name": canonical,
                    "category": meta["category"],
                    "parent": meta["parent"],
                    "aliases": meta.get("aliases", []),
                })
        return results[:limit]

    def get_all_categories(self) -> Dict[str, List[str]]:
        """Return skill taxonomy grouped by category."""
        categories: Dict[str, List[str]] = {}
        for canonical, meta in SKILL_TAXONOMY.items():
            cat = meta["category"]
            categories.setdefault(cat, []).append(canonical)
        return dict(sorted(categories.items()))


taxonomy_agent = TaxonomyAgent()
