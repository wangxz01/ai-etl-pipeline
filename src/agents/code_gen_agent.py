"""Code generation agent — produces SQL and Python ETL scripts from parsed requirements."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.agents.base_agent import AgentResult, BaseAgent
from src.parser.requirement_parser import ParsedRequirement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class TestCase(BaseModel):
    """A test case for the generated code."""

    name: str = Field(description="Descriptive test name")
    input_description: str = Field(description="What the test input looks like")
    expected_behavior: str = Field(description="What the test asserts")


class GeneratedCode(BaseModel):
    """Structured output of the code generation agent."""

    sql_code: str = Field(description="Full SQL query with CTEs and window functions")
    python_code: str = Field(description="Python / pandas ETL script")
    explanation: str = Field(description="Plain-English explanation of the approach")
    test_cases: list[TestCase] = Field(default_factory=list, description="Suggested test cases")
    dependencies: list[str] = Field(
        default_factory=list, description="Python packages required beyond stdlib"
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CODE_GEN_SYSTEM_PROMPT = """\
You are a senior data engineer writing production-grade ETL code. Given a
structured ETL requirement, generate:

1. A SQL query using CTEs, window functions, and proper error handling.
   - Use COALESCE for null safety.
   - Include query hints for large datasets.
   - Add comments for complex logic.

2. A Python script using pandas that:
   - Reads from the source data.
   - Applies all cleaning rules.
   - Performs aggregations.
   - Writes to the target in the specified format.
   - Includes progress bars (tqdm), checkpointing to disk, and proper logging.

3. A plain-English explanation of the approach.

4. At least 3 test case descriptions.

Return ONLY a JSON object with keys:
{
  "sql_code": "...",
  "python_code": "...",
  "explanation": "...",
  "test_cases": [{"name": "...", "input_description": "...", "expected_behavior": "..."}],
  "dependencies": ["..."]
}
"""


class CodeGenAgent(BaseAgent):
    """Generates SQL queries and Python ETL scripts from a parsed requirement."""

    name = "code_gen_agent"

    def _run(self, *, requirement: ParsedRequirement, **kwargs: Any) -> AgentResult:
        self._emit_progress("Building prompt from parsed requirement", 0.1)

        user_message = json.dumps(
            {
                "summary": requirement.summary,
                "data_sources": [ds.model_dump() for ds in requirement.data_sources],
                "cleaning_rules": [cr.model_dump() for cr in requirement.cleaning_rules],
                "aggregation_rules": [ar.model_dump() for ar in requirement.aggregation_rules],
                "output_format": requirement.output_format.model_dump(),
                "glossary_terms": [gt.model_dump() for gt in requirement.glossary_terms],
            },
            indent=2,
        )

        self._emit_progress("Calling LLM for code generation", 0.3)
        raw = self._call_llm(
            messages=[
                {"role": "system", "content": _CODE_GEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

        self._emit_progress("Parsing generated code", 0.8)
        generated = self._parse_response(raw)

        self._emit_progress("Code generation complete", 1.0)
        return AgentResult(
            agent_name=self.name,
            success=True,
            payload={"generated_code": generated.model_dump()},
        )

    @staticmethod
    def _parse_response(raw: str) -> GeneratedCode:
        """Extract the JSON payload from the LLM response."""
        json_start = raw.rfind("{")
        json_end = raw.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON block found in code-gen response")
        data: dict[str, Any] = json.loads(raw[json_start:json_end])
        return GeneratedCode(**data)
