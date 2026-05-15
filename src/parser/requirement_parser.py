"""Natural-language ETL requirement parser using chain-of-thought reasoning.

Sends the user requirement through an 8–12 step reasoning chain to extract
structured metadata: data sources, cleaning rules, aggregation rules, output
format, and business glossary terms.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------

class DataSource(BaseModel):
    """Identified data source (table, API, file, etc.)."""

    name: str = Field(description="Logical name for this data source")
    source_type: str = Field(description="One of: table, api, file, view")
    identifier: str = Field(description="Table name, URL, or file path")
    key_columns: list[str] = Field(default_factory=list, description="Primary / join key columns")
    filters: list[str] = Field(default_factory=list, description="Pre-extraction filters to apply")


class CleaningRule(BaseModel):
    """A data-cleaning transformation."""

    column: str = Field(description="Target column name")
    rule_type: str = Field(
        description="One of: null_handling, dedup, type_cast, trim, regex_replace, outlier_removal"
    )
    description: str = Field(description="What the rule does in plain English")
    params: dict[str, Any] = Field(default_factory=dict, description="Rule parameters")


class AggregationRule(BaseModel):
    """An aggregation / transformation step."""

    source_columns: list[str] = Field(description="Input columns")
    function: str = Field(description="Aggregation function (sum, count, avg, etc.)")
    output_column: str = Field(description="Name of the resulting column")
    group_by: list[str] = Field(default_factory=list, description="GROUP BY columns")
    window: str | None = Field(default=None, description="Window specification if applicable")


class OutputFormat(BaseModel):
    """Desired output configuration."""

    format_type: str = Field(description="One of: parquet, csv, table, view")
    destination: str = Field(description="Target table name or file path")
    partition_columns: list[str] = Field(default_factory=list, description="Partition columns")
    sort_columns: list[str] = Field(default_factory=list, description="Sort/order-by columns")


class GlossaryTerm(BaseModel):
    """A business glossary term extracted from the requirement."""

    term: str = Field(description="The term or phrase as it appears in the requirement")
    definition: str = Field(description="Formal definition for the ETL context")
    sql_mapping: str = Field(description="SQL expression that implements this term")


class ParsedRequirement(BaseModel):
    """Fully structured ETL requirement produced by the parser."""

    original_text: str = Field(description="Original natural-language input")
    summary: str = Field(description="One-sentence summary of the ETL task")
    data_sources: list[DataSource] = Field(description="Identified data sources")
    cleaning_rules: list[CleaningRule] = Field(description="Required cleaning transformations")
    aggregation_rules: list[AggregationRule] = Field(description="Aggregation / transformation steps")
    output_format: OutputFormat = Field(description="Target output specification")
    glossary_terms: list[GlossaryTerm] = Field(description="Business glossary terms")
    reasoning_trace: list[str] = Field(
        default_factory=list, description="Step-by-step reasoning chain"
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert data engineer analyzing ETL requirements. You MUST perform
exactly 8 to 12 reasoning steps before producing the final structured output.

For each reasoning step, write a <step N> block that explains your analysis.
Cover AT LEAST the following topics (one per step):

1. Identify the core analytical question or business metric requested.
2. Enumerate every data source mentioned (tables, APIs, files) and infer
   likely schemas where not explicit.
3. Identify join conditions and relationships between sources.
4. Determine required filters and their conditions.
5. Specify data cleaning operations (null handling, deduplication, type casts).
6. Define aggregation logic — functions, groupings, window specifications.
7. Identify the desired output format and partitioning strategy.
8. Extract business glossary terms and map each to a concrete SQL expression.

Additional steps (9–12) should resolve ambiguities, note edge cases, or
document assumptions.

After ALL reasoning steps, output a JSON object matching this schema:
{
  "summary": "...",
  "data_sources": [{"name": "...", "source_type": "...", "identifier": "...",
                     "key_columns": [...], "filters": [...]}],
  "cleaning_rules": [{"column": "...", "rule_type": "...", "description": "...",
                       "params": {}}],
  "aggregation_rules": [{"source_columns": [...], "function": "...",
                          "output_column": "...", "group_by": [...],
                          "window": "..."}],
  "output_format": {"format_type": "...", "destination": "...",
                     "partition_columns": [...], "sort_columns": [...]},
  "glossary_terms": [{"term": "...", "definition": "...", "sql_mapping": "..."}]
}

Output ONLY the reasoning steps followed by the JSON — no other text.
"""


class RequirementParser:
    """Parse natural-language ETL requirements via chain-of-thought LLM calls."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = settings.deepseek_model
        self._max_tokens = settings.deepseek_max_tokens
        self._temperature = settings.deepseek_temperature

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def parse(self, text: str) -> ParsedRequirement:
        """Parse a natural-language requirement into a structured object.

        Args:
            text: The free-form ETL requirement description.

        Returns:
            A ParsedRequirement with extracted metadata and reasoning trace.

        Raises:
            ValueError: If the LLM output cannot be parsed.
        """
        logger.info("Parsing requirement: %s", text[:120])

        response = self._call_llm(text)
        reasoning_trace, payload = self._extract_structured_output(response)

        requirement = ParsedRequirement(
            original_text=text,
            summary=payload.get("summary", ""),
            data_sources=[DataSource(**ds) for ds in payload.get("data_sources", [])],
            cleaning_rules=[CleaningRule(**cr) for cr in payload.get("cleaning_rules", [])],
            aggregation_rules=[AggregationRule(**ar) for ar in payload.get("aggregation_rules", [])],
            output_format=OutputFormat(**payload.get("output_format", {})),
            glossary_terms=[GlossaryTerm(**gt) for gt in payload.get("glossary_terms", [])],
            reasoning_trace=reasoning_trace,
        )

        logger.info(
            "Parsed requirement: %d sources, %d cleaning rules, %d aggregations, %d glossary terms",
            len(requirement.data_sources),
            len(requirement.cleaning_rules),
            len(requirement.aggregation_rules),
            len(requirement.glossary_terms),
        )
        return requirement

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _call_llm(self, user_text: str) -> str:
        """Send the requirement to the LLM and return the raw response text."""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned an empty response")
        return content

    @staticmethod
    def _extract_structured_output(raw: str) -> tuple[list[str], dict[str, Any]]:
        """Split the LLM response into reasoning steps and the final JSON payload.

        Returns:
            A tuple of (reasoning_steps, parsed_json_dict).

        Raises:
            ValueError: If no valid JSON block is found.
        """
        # Collect <step N>...</step> blocks as the reasoning trace.
        import re

        steps = re.findall(r"<step\s+\d+>(.*?)</step\s+\d+>", raw, re.DOTALL)
        reasoning_trace = [s.strip() for s in steps]

        # Extract the JSON block — try from the last occurrence of '{' to end.
        json_start = raw.rfind("{")
        json_end = raw.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in LLM response")

        json_str = raw[json_start:json_end]
        try:
            payload: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in LLM response: {exc}") from exc

        # If no structured <step> blocks were found, fall back to splitting by
        # numbered lines before the JSON block.
        if not reasoning_trace:
            pre_json = raw[:json_start]
            lines = [ln.strip() for ln in pre_json.splitlines() if ln.strip()]
            reasoning_trace = lines

        return reasoning_trace, payload
