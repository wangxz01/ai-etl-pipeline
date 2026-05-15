"""Code review agent — reviews generated code for security, performance, and correctness."""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.agents.base_agent import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewIssue(BaseModel):
    """A single issue found during code review."""

    severity: Severity
    category: str = Field(
        description="One of: sql_injection, index_usage, pandas_efficiency, error_handling, style"
    )
    location: str = Field(description="File, line range, or SQL clause affected")
    description: str = Field(description="What the issue is")
    suggestion: str = Field(description="How to fix it")


class CodeReview(BaseModel):
    """Structured output of the code review agent."""

    approved: bool = Field(description="Whether the code passes review (no CRITICAL/HIGH issues)")
    issues: list[ReviewIssue] = Field(default_factory=list)
    overall_assessment: str = Field(description="Summary paragraph of the review")
    security_score: float = Field(ge=0.0, le=10.0, description="Security score 0–10")
    performance_score: float = Field(ge=0.0, le=10.0, description="Performance score 0–10")
    maintainability_score: float = Field(ge=0.0, le=10.0, description="Maintainability score 0–10")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_REVIEW_SYSTEM_PROMPT = """\
You are a senior data engineer performing a rigorous code review. Analyze the
provided SQL and Python code for the following categories:

1. **SQL Injection**: Detect string interpolation, unsanitized inputs, or
   dynamic SQL construction that could allow injection.
2. **Index Usage / Query Performance**: Check that WHERE/JOIN columns are
   indexed, that CTEs are not materializing unnecessarily, and that window
   functions have proper PARTITION BY clauses.
3. **Pandas Chain Efficiency**: Flag .iterrows(), repeated df.copy(),
   in-place operations on slices, and suggest vectorized alternatives.
4. **Error Handling Completeness**: Verify try/except blocks, transaction
   rollback, checkpoint cleanup, and graceful failure modes.
5. **General Style**: Readability, naming, comments where needed.

For each issue, provide severity (critical/high/medium/low/info), category,
location, description, and a concrete fix suggestion.

Return ONLY a JSON object:
{
  "approved": true/false,
  "issues": [
    {
      "severity": "critical|high|medium|low|info",
      "category": "...",
      "location": "...",
      "description": "...",
      "suggestion": "..."
    }
  ],
  "overall_assessment": "...",
  "security_score": 0-10,
  "performance_score": 0-10,
  "maintainability_score": 0-10
}

Mark approved=false if any CRITICAL or HIGH issues exist.
"""


class CodeReviewAgent(BaseAgent):
    """Reviews generated ETL code for security, performance, and correctness."""

    name = "code_review_agent"

    def _run(
        self,
        *,
        sql_code: str,
        python_code: str,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_progress("Preparing code for review", 0.1)

        user_message = json.dumps(
            {"sql_code": sql_code, "python_code": python_code},
            indent=2,
        )

        self._emit_progress("Running LLM-based code review", 0.3)
        raw = self._call_llm(
            messages=[
                {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        self._emit_progress("Parsing review results", 0.8)
        review = self._parse_response(raw)

        self._emit_progress(
            f"Review complete — approved={review.approved}, "
            f"{len(review.issues)} issues found",
            1.0,
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            payload={"review": review.model_dump()},
        )

    @staticmethod
    def _parse_response(raw: str) -> CodeReview:
        json_start = raw.rfind("{")
        json_end = raw.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON block found in code-review response")
        data: dict[str, Any] = json.loads(raw[json_start:json_end])
        return CodeReview(**data)
