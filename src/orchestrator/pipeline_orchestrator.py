"""Multi-agent pipeline orchestrator — coordinates the full ETL generation workflow.

State machine: PARSING → GENERATING → REVIEWING → VALIDATING → DEPLOYING → COMPLETED
                                                                                ↘ FAILED
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.agents.base_agent import AgentResult
from src.agents.code_gen_agent import CodeGenAgent
from src.agents.code_review_agent import CodeReviewAgent
from src.agents.data_validator import DataValidatorAgent
from src.config import get_settings
from src.parser.requirement_parser import ParsedRequirement, RequirementParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class PipelinePhase(str, Enum):
    """Phases of the pipeline state machine."""

    PARSING = "PARSING"
    GENERATING = "GENERATING"
    REVIEWING = "REVIEWING"
    VALIDATING = "VALIDATING"
    DEPLOYING = "DEPLOYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PipelineState(BaseModel):
    """Snapshot of the pipeline state at any point in time."""

    phase: PipelinePhase = PipelinePhase.PARSING
    current_step: str = ""
    attempt: int = 0
    max_attempts: int = 3
    errors: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)


# Type alias for progress callbacks
ProgressCallback = Callable[[PipelinePhase, str, float], None]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Coordinates parse → generate → review → validate → deploy.

    Features:
    - Parallel agent execution where possible.
    - Auto-fix loop: if the code reviewer rejects the output, feedback is
      sent back to the code generator (max 3 retries).
    - State machine with phase transitions and progress callbacks.
    """

    def __init__(
        self,
        max_retry_attempts: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        settings = get_settings()
        self._max_attempts = max_retry_attempts or settings.max_retry_attempts
        self._output_dir = settings.output_dir
        self._progress_callback = progress_callback
        self.state = PipelineState(max_attempts=self._max_attempts)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def run(self, requirement_text: str) -> PipelineState:
        """Execute the full pipeline for a natural-language requirement.

        Args:
            requirement_text: Free-form ETL requirement.

        Returns:
            Final PipelineState with all artifacts.
        """
        try:
            # Phase 1 — Parse
            parsed = self._parse(requirement_text)
            self.state.artifacts["parsed_requirement"] = parsed.model_dump()

            # Phase 2 — Generate (with review loop)
            code_result = self._generate_with_review_loop(parsed)
            self.state.artifacts["generated_code"] = code_result.payload.get(
                "generated_code", {}
            )

            # Phase 3 — Validate (runs in parallel concept with generation done)
            self._validate(parsed, code_result)

            # Phase 4 — Deploy
            self._deploy(parsed, code_result)

            self._transition(PipelinePhase.COMPLETED, "Pipeline completed successfully")
            return self.state

        except Exception as exc:
            logger.exception("Pipeline failed")
            self.state.errors.append(str(exc))
            self._transition(PipelinePhase.FAILED, f"Pipeline failed: {exc}")
            return self.state

    # -------------------------------------------------------------------
    # Phase implementations
    # -------------------------------------------------------------------

    def _parse(self, text: str) -> ParsedRequirement:
        """Phase 1: Parse the natural language requirement."""
        self._transition(PipelinePhase.PARSING, "Parsing natural language requirement", 0.05)
        parser = RequirementParser()
        parsed = parser.parse(text)
        self._transition(PipelinePhase.PARSING, "Requirement parsed", 0.15)
        return parsed

    def _generate_with_review_loop(self, parsed: ParsedRequirement) -> AgentResult:
        """Phase 2–3: Generate code and review in a retry loop.

        The code review agent examines the generated code. If it finds
        CRITICAL or HIGH issues, the feedback is fed back to the code
        generator for another attempt (up to max_attempts).
        """
        gen_agent = CodeGenAgent(
            progress_callback=lambda msg, pct: self._emit(PipelinePhase.GENERATING, msg, pct)
        )
        review_agent = CodeReviewAgent(
            progress_callback=lambda msg, pct: self._emit(PipelinePhase.REVIEWING, msg, pct)
        )

        last_review_payload: dict[str, Any] = {}
        code_result: AgentResult | None = None

        for attempt in range(1, self._max_attempts + 1):
            self.state.attempt = attempt
            self._transition(
                PipelinePhase.GENERATING,
                f"Code generation attempt {attempt}/{self._max_attempts}",
                0.2 + 0.1 * (attempt - 1),
            )

            code_result = gen_agent.execute(requirement=parsed)
            if not code_result.success:
                raise RuntimeError(f"Code generation failed: {code_result.errors}")

            generated = code_result.payload["generated_code"]
            self._transition(
                PipelinePhase.REVIEWING,
                f"Reviewing generated code (attempt {attempt})",
                0.5 + 0.1 * (attempt - 1),
            )

            review_result = review_agent.execute(
                sql_code=generated["sql_code"],
                python_code=generated["python_code"],
            )
            if not review_result.success:
                raise RuntimeError(f"Code review failed: {review_result.errors}")

            review_data = review_result.payload["review"]
            last_review_payload = review_data
            self.state.artifacts["code_review"] = review_data

            if review_data.get("approved", False):
                logger.info("Code approved on attempt %d", attempt)
                return code_result

            # Feed review feedback back into next generation attempt
            issues_summary = json.dumps(review_data.get("issues", []), indent=2)
            logger.warning(
                "Code not approved (attempt %d). Issues:\n%s",
                attempt,
                issues_summary[:500],
            )
            # Append reviewer feedback so the next generation round can fix it
            parsed = self._augment_requirement_with_feedback(parsed, issues_summary)

        logger.warning("Max retries reached; using last generated code")
        return code_result  # type: ignore[return-value]

    def _validate(
        self, parsed: ParsedRequirement, code_result: AgentResult
    ) -> None:
        """Phase 4: Validate the generated code's output logic."""
        self._transition(PipelinePhase.VALIDATING, "Validating data output", 0.7)

        validator = DataValidatorAgent(
            progress_callback=lambda msg, pct: self._emit(PipelinePhase.VALIDATING, msg, pct)
        )

        generated = code_result.payload["generated_code"]
        validation_result = validator.execute(
            requirement_summary=parsed.summary,
            sql_code=generated["sql_code"],
            python_code=generated["python_code"],
        )

        if not validation_result.success:
            logger.warning("Validation agent failed: %s", validation_result.errors)
            self.state.artifacts["validation"] = {"error": validation_result.errors}
        else:
            self.state.artifacts["validation"] = validation_result.payload.get("validation", {})

        self._transition(PipelinePhase.VALIDATING, "Validation complete", 0.85)

    def _deploy(self, parsed: ParsedRequirement, code_result: AgentResult) -> None:
        """Phase 5: Write generated artifacts to disk."""
        self._transition(PipelinePhase.DEPLOYING, "Deploying artifacts", 0.9)

        settings = get_settings()
        output_dir = self._output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        generated = code_result.payload["generated_code"]

        # Write SQL file
        sql_path = output_dir / "generated_query.sql"
        sql_path.write_text(generated["sql_code"])
        logger.info("Wrote SQL to %s", sql_path)

        # Write Python file
        py_path = output_dir / "etl_script.py"
        py_path.write_text(generated["python_code"])
        logger.info("Wrote Python script to %s", py_path)

        # Write metadata
        meta_path = output_dir / "pipeline_metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "summary": parsed.summary,
                    "data_sources": [ds.model_dump() for ds in parsed.data_sources],
                    "review": self.state.artifacts.get("code_review"),
                    "validation": self.state.artifacts.get("validation"),
                },
                indent=2,
                default=str,
            )
        )
        logger.info("Wrote metadata to %s", meta_path)

        self.state.artifacts["output_files"] = {
            "sql": str(sql_path),
            "python": str(py_path),
            "metadata": str(meta_path),
        }
        self._transition(PipelinePhase.DEPLOYING, "Artifacts deployed", 0.95)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _augment_requirement_with_feedback(
        parsed: ParsedRequirement, review_feedback: str
    ) -> ParsedRequirement:
        """Append reviewer feedback to the requirement's glossary so the next
        generation round can address the issues."""
        from src.parser.requirement_parser import GlossaryTerm

        new_terms = parsed.glossary_terms + [
            GlossaryTerm(
                term="CODE_REVIEW_FEEDBACK",
                definition="Issues found in the previous code generation attempt that MUST be fixed.",
                sql_mapping=review_feedback[:2000],
            )
        ]
        return parsed.model_copy(update={"glossary_terms": new_terms})

    def _transition(self, phase: PipelinePhase, step: str, progress: float = 0.0) -> None:
        """Update the pipeline state and emit a progress callback."""
        self.state.phase = phase
        self.state.current_step = step
        logger.info("[%s] %s", phase.value, step)
        if self._progress_callback:
            self._progress_callback(phase, step, progress)

    def _emit(self, phase: PipelinePhase, message: str, fraction: float) -> None:
        """Progress helper used by agent callbacks."""
        if self._progress_callback:
            self._progress_callback(phase, message, fraction)
