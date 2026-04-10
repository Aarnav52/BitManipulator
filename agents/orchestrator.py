"""
Agent Orchestrator
==================
Wires ParsingAgent → TaxonomyAgent together.
Returns ParsedResume with normalized skills attached.
Registers each candidate in the match-route store for /match calls.
"""
from __future__ import annotations
import asyncio
import concurrent.futures
import logging

from agents.parsing_agent import parsing_agent
from agents.taxonomy_agent import taxonomy_agent
from models.resume import ParsedResume

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def _run_async(self, coro):
        """Run an async coroutine safely whether or not an event loop is running."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result(timeout=90)
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def process_resume(self, file_bytes: bytes, filename: str) -> ParsedResume:
        """
        Full pipeline: parse → normalize skills → return enriched ParsedResume.
        Also registers the candidate in the match-route store.
        """
        # Step 1: parse
        resume = parsing_agent.parse_file(file_bytes, filename)

        # Step 2: normalize skills via taxonomy agent (mutates resume.skills in-place)
        resume = taxonomy_agent.normalize_resume(resume)

        # Step 3: register in candidate store so /match can find it
        try:
            from api.routes.match import register_candidate
            register_candidate(resume.id, resume)
        except Exception as e:
            logger.warning(f"Could not register candidate in match store: {e}")

        logger.info(
            f"Pipeline complete: id={resume.id} name='{resume.full_name}' "
            f"skills={len(resume.skills)} confidence={resume.parsing_confidence:.2f}"
        )
        return resume


orchestrator = AgentOrchestrator()
