"""Data validation agent — validates ETL outputs against source data and business rules."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.agents.base_agent import AgentResult, BaseAgent
from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    """Validation check outcome."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ValidationCheck(BaseModel):
    """Result of a single validation check."""

    name: str = Field(description="Name of the check performed")
    status: ValidationStatus
    metric: float = Field(description="Numeric result of the check")
    threshold: float = Field(description="Threshold that was applied")
    details: str = Field(description="Human-readable explanation")


class ValidationResult(BaseModel):
    """Complete validation result for an ETL output."""

    overall_status: ValidationStatus
    checks: list[ValidationCheck] = Field(default_factory=list)
    recommendation: str = Field(description="What to do next based on the results")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DataValidatorAgent(BaseAgent):
    """Validates ETL output data quality.

    Checks:
    - Row count comparison (source vs. target).
    - Null rate detection per column.
    - Distribution shift detection.
    - Business rule conformance.

    In production this would query real databases; here it uses the LLM to
    analyze expected vs. actual schemas and produce a structured validation
    report.
    """

    name = "data_validator"

    def _run(
        self,
        *,
        requirement_summary: str,
        sql_code: str,
        python_code: str,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_progress("Building validation plan", 0.1)

        settings = get_settings()

        user_message = json.dumps(
            {
                "requirement_summary": requirement_summary,
                "sql_code": sql_code,
                "python_code": python_code,
                "validation_thresholds": {
                    "max_null_rate": settings.max_null_rate,
                    "max_distribution_shift": settings.max_distribution_shift,
                    "row_count_tolerance": settings.row_count_tolerance,
                },
            },
            indent=2,
        )

        self._emit_progress("Running LLM-based validation analysis", 0.3)
        raw = self._call_llm(
            messages=[
                {"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        self._emit_progress("Parsing validation results", 0.8)
        result = self._parse_response(raw)

        self._emit_progress(
            f"Validation complete — overall={result.overall_status.value}", 1.0
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            payload={"validation": result.model_dump()},
        )

    @staticmethod
    def _parse_response(raw: str) -> ValidationResult:
        json_start = raw.rfind("{")
        json_end = raw.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON block found in validation response")
        data: dict[str, Any] = json.loads(raw[json_start:json_end])
        return ValidationResult(**data)


_VALIDATION_SYSTEM_PROMPT = """\
You are a data quality engineer. Given an ETL requirement, the SQL query, and
the Python script, produce a validation plan with concrete checks and expected
results.

Analyze the code and produce checks for:
1. Row count — source vs. target (estimate based on filters/aggregation).
2. Null rate — for every output column, estimate expected null rate and compare
   against the provided threshold.
3. Distribution shift — identify columns where aggregation could skew
   distributions.
4. Business rule conformance — verify the output satisfies the original
   requirement.

Return ONLY a JSON object:
{
  "overall_status": "pass|warn|fail",
  "checks": [
    {
      "name": "...",
      "status": "pass|warn|fail",
      "metric": 0.0,
      "threshold": 0.0,
      "details": "..."
    }
  ],
  "recommendation": "..."
}

Mark overall_status as "fail" if any check fails, "warn" if all pass but some
warn, and "pass" if everything is clean.
"""
