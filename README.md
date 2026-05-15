# AI ETL Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

**Multi-agent ETL automation powered by large language models.**

Parse a natural language requirement → generate production SQL & Python ETL code → run an automated review loop → validate outputs against business rules → deploy as an Airflow DAG.

---

## Architecture

```
                          ┌─────────────────────────┐
                          │   Natural Language       │
                          │   Requirement (CLI/API)  │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   Requirement Parser     │
                          │   (DeepSeek Reasoner)    │
                          │   8–12 step chain-of-    │
                          │   thought extraction     │
                          └────────────┬────────────┘
                                       │
                          Parsed Requirement
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │  Code Gen Agent │ │ Code Review     │ │  Data Validator │
          │  (DeepSeek      │ │ Agent           │ │  Agent          │
          │   Coder)        │ │ (Security +     │ │  (Row counts,   │
          │                 │ │  Perf review)   │ │   null rates,   │
          │  • SQL with     │ │                 │ │   distribution) │
          │    CTEs, window │ │ • SQL injection │ │                 │
          │  • Python/pandas│ │ • Index usage   │ │                 │
          └────────┬────────┘ │ • Pandas perf   │ └────────┬────────┘
                   │          └────────┬────────┘          │
                   │                   │                   │
                   └───────────────────┼───────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  Pipeline Orchestrator   │
                          │  State machine:          │
                          │  PARSE → GENERATE →      │
                          │  REVIEW → VALIDATE →     │
                          │  DEPLOY                  │
                          │  (auto-fix loop ≤3×)     │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   Airflow DAG Factory    │
                          │   Dynamic DAG generation │
                          │   from parsed metadata   │
                          └─────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/ai-etl-pipeline.git
cd ai-etl-pipeline
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your DeepSeek API key and database URLs

# 3. Parse a natural language requirement
ai-etl parse --input "Get daily active users by region for the last 30 days"

# 4. Generate ETL code from a parsed requirement
ai-etl generate --input examples/sample_request.json

# 5. Run the full pipeline (parse → generate → review → validate → deploy)
ai-etl run-pipeline --input "Calculate monthly revenue per product category"

# 6. Validate existing output data
ai-etl validate --source-table raw_events --target-table agg_daily_users
```

## CLI Commands

| Command         | Description                                                       |
|-----------------|-------------------------------------------------------------------|
| `parse`         | Parse a natural language ETL requirement into structured metadata |
| `generate`      | Generate SQL + Python ETL code from a parsed requirement          |
| `run-pipeline`  | Execute the full multi-agent pipeline end-to-end                  |
| `validate`      | Validate ETL output data against source and business rules        |

## Project Structure

```
ai-etl-pipeline/
├── pyproject.toml                  # Project config, dependencies, tool settings
├── .env.example                    # Environment variable template
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                   # Pydantic-settings configuration
│   ├── main.py                     # Click CLI entry point
│   ├── parser/
│   │   ├── __init__.py
│   │   └── requirement_parser.py   # NL → structured requirement (chain-of-thought)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # Retry logic, token tracking, callbacks
│   │   ├── code_gen_agent.py       # SQL + Python code generation
│   │   ├── code_review_agent.py    # Security & performance review
│   │   └── data_validator.py       # Output data validation
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── pipeline_orchestrator.py # Multi-agent state machine
│   └── airflow/
│       ├── __init__.py
│       └── etl_dag_factory.py      # Dynamic Airflow DAG generation
├── tests/
│   ├── __init__.py
│   ├── test_parser.py              # Parser unit tests with mocked API
│   └── test_agents.py              # Agent unit tests with mocked responses
└── examples/
    ├── README.md                   # Examples guide
    └── sample_request.json         # Sample ETL request
```

## Configuration

All settings are loaded from environment variables (or a `.env` file) and validated by `pydantic-settings`. Key variables:

| Variable               | Default                          | Description                              |
|------------------------|----------------------------------|------------------------------------------|
| `DEEPSEEK_API_KEY`     | —                                | API key for the LLM provider             |
| `DEEPSEEK_BASE_URL`    | `https://api.deepseek.com/v1`   | OpenAI-compatible endpoint               |
| `DEEPSEEK_MODEL`       | `deepseek-reasoner`             | Model for reasoning tasks                |
| `DEEPSEEK_CODER_MODEL` | `deepseek-coder`                | Model for code generation/review         |
| `SOURCE_DB_URL`        | —                                | Source database connection string        |
| `TARGET_DB_URL`        | —                                | Target warehouse connection string       |
| `MAX_RETRY_ATTEMPTS`   | `3`                              | Max auto-fix retries in review loop      |
| `OUTPUT_DIR`           | `./output`                       | Directory for generated artifacts        |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## License

MIT
