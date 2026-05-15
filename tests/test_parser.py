"""Unit tests for the requirement parser with mocked LLM API calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.parser.requirement_parser import (
    ParsedRequirement,
    RequirementParser,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE_WITH_STEPS = """\
<step 1>The user wants daily active users by region for the last 30 days.</step 1>
<step 2>Two data sources identified: user_events table and user_profiles table.</step 2>
<step 3>Join on user_id between events and profiles.</step 3>
<step 4>Filter: event_date >= CURRENT_DATE - 30, exclude is_test_account = true.</step 4>
<step 5>Cleaning: handle null regions with COALESCE, deduplicate events by user+date.</step 5>
<step 6>Aggregation: COUNT(DISTINCT user_id) grouped by region and date.</step 6>
<step 7>Output: partitioned Parquet files by region.</step 7>
<step 8>Glossary: "daily active user" = user with at least one event on a given date.</step 8>
{
  "summary": "Count daily active users by region for the last 30 days, excluding test accounts, output as partitioned Parquet",
  "data_sources": [
    {
      "name": "user_events",
      "source_type": "table",
      "identifier": "public.user_events",
      "key_columns": ["user_id", "event_date"],
      "filters": ["event_date >= CURRENT_DATE - INTERVAL '30 days'"]
    },
    {
      "name": "user_profiles",
      "source_type": "table",
      "identifier": "public.user_profiles",
      "key_columns": ["user_id"],
      "filters": ["is_test_account = false"]
    }
  ],
  "cleaning_rules": [
    {
      "column": "region",
      "rule_type": "null_handling",
      "description": "Default null regions to 'Unknown'",
      "params": {"default": "Unknown"}
    },
    {
      "column": "user_id",
      "rule_type": "dedup",
      "description": "Deduplicate events by user_id and event_date",
      "params": {"subset": ["user_id", "event_date"]}
    }
  ],
  "aggregation_rules": [
    {
      "source_columns": ["user_id"],
      "function": "count_distinct",
      "output_column": "daily_active_users",
      "group_by": ["region", "event_date"],
      "window": null
    }
  ],
  "output_format": {
    "format_type": "parquet",
    "destination": "s3://warehouse/daily_active_users/",
    "partition_columns": ["region"],
    "sort_columns": ["event_date"]
  },
  "glossary_terms": [
    {
      "term": "daily active user",
      "definition": "A user who generated at least one event on a given calendar date",
      "sql_mapping": "COUNT(DISTINCT CASE WHEN event_date = CURRENT_DATE THEN user_id END)"
    }
  ]
}
"""


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client that returns a predictable response."""
    mock_client = MagicMock()

    # Simulate the .chat.completions.create call chain
    mock_choice = MagicMock()
    mock_choice.message.content = MOCK_LLM_RESPONSE_WITH_STEPS

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 500
    mock_usage.completion_tokens = 1200

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRequirementParser:
    """Tests for RequirementParser.parse()."""

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_returns_parsed_requirement(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        parser = RequirementParser()
        result = parser.parse("Get daily active users by region for the last 30 days")

        assert isinstance(result, ParsedRequirement)
        assert "daily active users" in result.summary.lower()
        assert len(result.data_sources) == 2
        assert result.data_sources[0].name == "user_events"
        assert result.data_sources[1].name == "user_profiles"

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_extracts_cleaning_rules(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        parser = RequirementParser()
        result = parser.parse("Get daily active users by region for the last 30 days")

        assert len(result.cleaning_rules) >= 2
        rule_types = {cr.rule_type for cr in result.cleaning_rules}
        assert "null_handling" in rule_types
        assert "dedup" in rule_types

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_extracts_aggregation_rules(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        parser = RequirementParser()
        result = parser.parse("Get daily active users by region for the last 30 days")

        assert len(result.aggregation_rules) >= 1
        agg = result.aggregation_rules[0]
        assert agg.function == "count_distinct"
        assert agg.output_column == "daily_active_users"
        assert "region" in agg.group_by

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_extracts_output_format(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        parser = RequirementParser()
        result = parser.parse("Get daily active users by region for the last 30 days")

        assert result.output_format.format_type == "parquet"
        assert "region" in result.output_format.partition_columns

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_extracts_glossary_terms(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        parser = RequirementParser()
        result = parser.parse("Get daily active users by region for the last 30 days")

        assert len(result.glossary_terms) >= 1
        assert result.glossary_terms[0].term == "daily active user"

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_extracts_reasoning_trace(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        parser = RequirementParser()
        result = parser.parse("Get daily active users by region for the last 30 days")

        assert len(result.reasoning_trace) == 8
        assert "daily active users" in result.reasoning_trace[0].lower()

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_preserves_original_text(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        text = "Get daily active users by region for the last 30 days"
        parser = RequirementParser()
        result = parser.parse(text)

        assert result.original_text == text

    def test_parse_raises_on_empty_response(self) -> None:
        with patch("src.parser.requirement_parser.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = None
            mock_client.chat.completions.create.return_value.choices = [mock_choice]
            mock_client.chat.completions.create.return_value.usage = MagicMock(
                prompt_tokens=0, completion_tokens=0
            )
            mock_cls.return_value = mock_client

            parser = RequirementParser()
            with pytest.raises(ValueError, match="empty response"):
                parser.parse("test")

    def test_parse_raises_on_missing_json(self) -> None:
        with patch("src.parser.requirement_parser.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "No JSON here, just text."
            mock_client.chat.completions.create.return_value.choices = [mock_choice]
            mock_client.chat.completions.create.return_value.usage = MagicMock(
                prompt_tokens=10, completion_tokens=10
            )
            mock_cls.return_value = mock_client

            parser = RequirementParser()
            with pytest.raises(ValueError, match="No JSON"):
                parser.parse("test")

    @patch("src.parser.requirement_parser.OpenAI")
    def test_parse_sends_correct_model(self, mock_openai_cls: MagicMock, mock_openai_client: MagicMock) -> None:
        mock_openai_cls.return_value = mock_openai_client

        parser = RequirementParser()
        parser.parse("test requirement")

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == parser._model
        messages = call_kwargs.kwargs.get("messages")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
