"""Unit tests for ETL agents with mocked LLM responses."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base_agent import AgentResult, BaseAgent, TokenUsage
from src.agents.code_gen_agent import CodeGenAgent, GeneratedCode
from src.agents.code_review_agent import CodeReview, CodeReviewAgent, ReviewIssue, Severity
from src.agents.data_validator import DataValidatorAgent, ValidationResult, ValidationStatus


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 500) -> MagicMock:
    """Build a mock OpenAI completion object."""
    mock_choice = MagicMock()
    mock_choice.message.content = content

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage = mock_usage
    return mock_completion


def _make_parsed_requirement() -> MagicMock:
    """Create a mock ParsedRequirement for code generation tests."""
    mock = MagicMock()
    mock.summary = "Count daily active users by region"
    mock.data_sources = []
    mock.cleaning_rules = []
    mock.aggregation_rules = []
    mock.output_format = MagicMock(model_dump=lambda: {"format_type": "parquet", "destination": "test"})
    mock.glossary_terms = []
    mock.model_dump.return_value = {"summary": mock.summary}
    return mock


# ---------------------------------------------------------------------------
# Test: BaseAgent
# ---------------------------------------------------------------------------

class TestBaseAgent:
    """Tests for BaseAgent retry and token tracking."""

    @patch("src.agents.base_agent.OpenAI")
    def test_token_usage_tracks_cumulative(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        usage = TokenUsage()
        usage.add(prompt=100, completion=200)
        usage.add(prompt=50, completion=150)
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 350
        assert usage.total_tokens == 500

    @patch("src.agents.base_agent.OpenAI")
    def test_execute_returns_agent_result(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()

        class TestAgent(BaseAgent):
            name = "test_agent"
            def _run(self, **kwargs):
                return AgentResult(agent_name=self.name, success=True, payload={"key": "value"})

        agent = TestAgent()
        result = agent.execute()
        assert result.success is True
        assert result.agent_name == "test_agent"
        assert result.payload["key"] == "value"
        assert result.elapsed_seconds >= 0

    @patch("src.agents.base_agent.OpenAI")
    def test_execute_captures_exceptions(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()

        class FailAgent(BaseAgent):
            name = "fail_agent"
            def _run(self, **kwargs):
                raise RuntimeError("boom")

        agent = FailAgent()
        result = agent.execute()
        assert result.success is False
        assert "boom" in result.errors[0]

    @patch("src.agents.base_agent.OpenAI")
    def test_call_llm_retries_on_failure(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Fail twice then succeed
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            _make_mock_completion("success"),
        ]

        class RetryAgent(BaseAgent):
            name = "retry_agent"
            def _run(self, **kwargs):
                return AgentResult(agent_name=self.name, success=True)

        agent = RetryAgent(max_retries=3)
        with patch("src.agents.base_agent.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 0.1, 0.2, 0.3, 0.4]
            mock_time.sleep = MagicMock()
            content = agent._call_llm([{"role": "user", "content": "test"}])

        assert content == "success"
        assert mock_client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# Test: CodeGenAgent
# ---------------------------------------------------------------------------

MOCK_CODE_GEN_RESPONSE = json.dumps({
    "sql_code": "SELECT COUNT(DISTINCT user_id) AS dau FROM events WHERE date >= CURRENT_DATE - 30 GROUP BY region;",
    "python_code": "import pandas as pd\ndf = pd.read_sql(sql, conn)\ndf.to_parquet('output.parquet')",
    "explanation": "Count distinct users per region for the last 30 days.",
    "test_cases": [
        {"name": "test_row_count", "input_description": "30 days of events", "expected_behavior": "At least 30 rows (one per day)"},
        {"name": "test_null_region", "input_description": "Events with null region", "expected_behavior": "Null regions defaulted to 'Unknown'"},
        {"name": "test_test_accounts_excluded", "input_description": "Mix of test and real accounts", "expected_behavior": "Only real accounts counted"},
    ],
    "dependencies": ["pandas", "sqlalchemy"],
})


class TestCodeGenAgent:

    @patch("src.agents.base_agent.OpenAI")
    def test_generate_produces_sql_and_python(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            MOCK_CODE_GEN_RESPONSE
        )

        agent = CodeGenAgent()
        result = agent.execute(requirement=_make_parsed_requirement())

        assert result.success is True
        code = result.payload["generated_code"]
        assert "SELECT" in code["sql_code"]
        assert "import pandas" in code["python_code"]
        assert len(code["test_cases"]) == 3
        assert "pandas" in code["dependencies"]

    @patch("src.agents.base_agent.OpenAI")
    def test_generate_tracks_tokens(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            MOCK_CODE_GEN_RESPONSE, prompt_tokens=200, completion_tokens=800
        )

        agent = CodeGenAgent()
        agent.execute(requirement=_make_parsed_requirement())

        assert agent.token_usage.prompt_tokens == 200
        assert agent.token_usage.completion_tokens == 800


# ---------------------------------------------------------------------------
# Test: CodeReviewAgent
# ---------------------------------------------------------------------------

MOCK_REVIEW_RESPONSE = json.dumps({
    "approved": True,
    "issues": [
        {
            "severity": "info",
            "category": "style",
            "location": "etl_script.py:12",
            "description": "Consider adding a docstring to the main function",
            "suggestion": "Add a docstring explaining the ETL pipeline steps",
        }
    ],
    "overall_assessment": "Code is clean and well-structured. Minor style suggestion.",
    "security_score": 9.5,
    "performance_score": 8.0,
    "maintainability_score": 9.0,
})


class TestCodeReviewAgent:

    @patch("src.agents.base_agent.OpenAI")
    def test_review_returns_structured_result(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            MOCK_REVIEW_RESPONSE
        )

        agent = CodeReviewAgent()
        result = agent.execute(
            sql_code="SELECT 1",
            python_code="print('hello')",
        )

        assert result.success is True
        review = result.payload["review"]
        assert review["approved"] is True
        assert len(review["issues"]) == 1
        assert review["security_score"] == 9.5

    @patch("src.agents.base_agent.OpenAI")
    def test_review_flags_critical_issues(self, mock_cls: MagicMock) -> None:
        critical_response = json.dumps({
            "approved": False,
            "issues": [
                {
                    "severity": "critical",
                    "category": "sql_injection",
                    "location": "etl_script.py:45",
                    "description": "User input interpolated directly into SQL query",
                    "suggestion": "Use parameterized queries instead of f-strings",
                }
            ],
            "overall_assessment": "Critical SQL injection vulnerability found.",
            "security_score": 2.0,
            "performance_score": 7.0,
            "maintainability_score": 6.0,
        })
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(critical_response)

        agent = CodeReviewAgent()
        result = agent.execute(sql_code="SELECT 1", python_code="print('hello')")

        review = result.payload["review"]
        assert review["approved"] is False
        assert review["issues"][0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# Test: DataValidatorAgent
# ---------------------------------------------------------------------------

MOCK_VALIDATION_RESPONSE = json.dumps({
    "overall_status": "pass",
    "checks": [
        {
            "name": "row_count_comparison",
            "status": "pass",
            "metric": 0.005,
            "threshold": 0.01,
            "details": "Row count difference within tolerance (0.5% vs 1% threshold)",
        },
        {
            "name": "null_rate_check",
            "status": "pass",
            "metric": 0.001,
            "threshold": 0.05,
            "details": "Max null rate across columns is 0.1% (threshold 5%)",
        },
        {
            "name": "distribution_shift",
            "status": "warn",
            "metric": 0.08,
            "threshold": 0.1,
            "details": "KS statistic for user_count column is 0.08 (close to threshold)",
        },
        {
            "name": "business_rule_conformance",
            "status": "pass",
            "metric": 1.0,
            "threshold": 1.0,
            "details": "All output rows satisfy the daily active user definition",
        },
    ],
    "recommendation": "Monitor the distribution shift in user_count over time. Consider increasing the sample window if drift continues.",
})


class TestDataValidatorAgent:

    @patch("src.agents.base_agent.OpenAI")
    def test_validate_returns_structured_result(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(
            MOCK_VALIDATION_RESPONSE
        )

        agent = DataValidatorAgent()
        result = agent.execute(
            requirement_summary="Count daily active users",
            sql_code="SELECT 1",
            python_code="pass",
        )

        assert result.success is True
        validation = result.payload["validation"]
        assert validation["overall_status"] == "pass"
        assert len(validation["checks"]) == 4

    @patch("src.agents.base_agent.OpenAI")
    def test_validate_detects_failures(self, mock_cls: MagicMock) -> None:
        fail_response = json.dumps({
            "overall_status": "fail",
            "checks": [
                {
                    "name": "row_count_comparison",
                    "status": "fail",
                    "metric": 0.5,
                    "threshold": 0.01,
                    "details": "50% row count mismatch — likely missing filter",
                }
            ],
            "recommendation": "Re-examine the WHERE clause in the SQL query.",
        })
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_completion(fail_response)

        agent = DataValidatorAgent()
        result = agent.execute(
            requirement_summary="test",
            sql_code="SELECT 1",
            python_code="pass",
        )

        validation = result.payload["validation"]
        assert validation["overall_status"] == "fail"
        assert validation["checks"][0]["status"] == "fail"
