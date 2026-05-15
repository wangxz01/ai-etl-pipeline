"""Airflow DAG factory — generates DAGs dynamically from parsed ETL requirements.

This module reads a parsed requirement's metadata and produces a fully-wired
Airflow DAG with tasks matching the pipeline flow:

    parse → generate → review → (conditional) regenerate → validate → deploy

Usage in an Airflow DAGs folder::

    from src.airflow.etl_dag_factory import build_dag_from_metadata

    dag = build_dag_from_metadata("path/to/pipeline_metadata.json")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Airflow imports are optional — the factory is usable even without a running
# Airflow environment (e.g., for unit testing). We guard the imports so that
# the rest of the package can be imported without Airflow installed.
try:
    from airflow.models.dag import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.empty import EmptyOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    logger.debug("Airflow not installed — DAG factory will produce JSON-only output")


# ---------------------------------------------------------------------------
# Default DAG kwargs
# ---------------------------------------------------------------------------

_DEFAULT_DAG_ARGS: dict[str, Any] = {
    "owner": "ai-etl-pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ---------------------------------------------------------------------------
# Task implementations (standalone callables)
# ---------------------------------------------------------------------------

def _task_parse(**context: Any) -> dict[str, Any]:
    """Airflow task: parse a natural language requirement."""
    from src.parser.requirement_parser import RequirementParser

    requirement_text = context["params"]["requirement_text"]
    parser = RequirementParser()
    parsed = parser.parse(requirement_text)
    return parsed.model_dump()


def _task_generate(**context: Any) -> dict[str, Any]:
    """Airflow task: generate ETL code from a parsed requirement."""
    from src.agents.code_gen_agent import CodeGenAgent
    from src.parser.requirement_parser import ParsedRequirement

    ti = context["ti"]
    parsed_dict = ti.xcom_pull(task_ids="parse_requirement")
    parsed = ParsedRequirement(**parsed_dict)
    agent = CodeGenAgent()
    result = agent.execute(requirement=parsed)
    return result.payload.get("generated_code", {})


def _task_review(**context: Any) -> dict[str, Any]:
    """Airflow task: review generated code."""
    from src.agents.code_review_agent import CodeReviewAgent

    ti = context["ti"]
    code = ti.xcom_pull(task_ids="generate_code")
    agent = CodeReviewAgent()
    result = agent.execute(sql_code=code["sql_code"], python_code=code["python_code"])
    return result.payload.get("review", {})


def _task_validate(**context: Any) -> dict[str, Any]:
    """Airflow task: validate data output."""
    from src.agents.data_validator import DataValidatorAgent

    ti = context["ti"]
    code = ti.xcom_pull(task_ids="generate_code")
    parsed_dict = ti.xcom_pull(task_ids="parse_requirement")
    agent = DataValidatorAgent()
    result = agent.execute(
        requirement_summary=parsed_dict.get("summary", ""),
        sql_code=code["sql_code"],
        python_code=code["python_code"],
    )
    return result.payload.get("validation", {})


def _task_deploy(**context: Any) -> dict[str, str]:
    """Airflow task: write artifacts to output directory."""
    ti = context["ti"]
    code = ti.xcom_pull(task_ids="generate_code")
    parsed_dict = ti.xcom_pull(task_ids="parse_requirement")

    output_dir = Path(context["params"].get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    sql_path = output_dir / "generated_query.sql"
    sql_path.write_text(code.get("sql_code", ""))

    py_path = output_dir / "etl_script.py"
    py_path.write_text(code.get("python_code", ""))

    meta_path = output_dir / "pipeline_metadata.json"
    meta_path.write_text(json.dumps({"parsed": parsed_dict, "code": code}, indent=2, default=str))

    return {"sql": str(sql_path), "python": str(py_path), "metadata": str(meta_path)}


# ---------------------------------------------------------------------------
# DAG factory
# ---------------------------------------------------------------------------

def build_dag(
    dag_id: str,
    requirement_text: str,
    schedule: str | None = "@daily",
    start_date: datetime | None = None,
    output_dir: str = "./output",
    default_args: dict[str, Any] | None = None,
) -> Any:
    """Construct an Airflow DAG from a natural language requirement.

    Args:
        dag_id: Unique DAG identifier.
        requirement_text: The ETL requirement in natural language.
        schedule: Cron expression or Airflow preset.
        start_date: DAG start date (defaults to today).
        output_dir: Where to write generated artifacts.
        default_args: Override default task arguments.

    Returns:
        An Airflow DAG instance (if Airflow is installed) or a dict describing
        the DAG structure (if Airflow is not installed).
    """
    args = {**_DEFAULT_DAG_ARGS, **(default_args or {})}
    if start_date is None:
        start_date = datetime(2024, 1, 1)

    params = {"requirement_text": requirement_text, "output_dir": output_dir}

    if not _AIRFLOW_AVAILABLE:
        logger.warning("Airflow not installed — returning DAG descriptor as dict")
        return {
            "dag_id": dag_id,
            "schedule": schedule,
            "start_date": str(start_date),
            "default_args": args,
            "params": params,
            "tasks": [
                "start",
                "parse_requirement",
                "generate_code",
                "review_code",
                "validate_data",
                "deploy",
                "end",
            ],
            "dependencies": [
                "start >> parse_requirement",
                "parse_requirement >> generate_code",
                "generate_code >> review_code",
                "review_code >> validate_data",
                "validate_data >> deploy",
                "deploy >> end",
            ],
        }

    with DAG(
        dag_id=dag_id,
        schedule=schedule,
        start_date=start_date,
        default_args=args,
        description=f"AI-generated ETL pipeline: {requirement_text[:80]}",
        params=params,
        catchup=False,
        tags=["ai-generated", "etl"],
    ) as dag:

        start = EmptyOperator(task_id="start")

        parse = PythonOperator(
            task_id="parse_requirement",
            python_callable=_task_parse,
            provide_context=True,
        )

        generate = PythonOperator(
            task_id="generate_code",
            python_callable=_task_generate,
            provide_context=True,
        )

        review = PythonOperator(
            task_id="review_code",
            python_callable=_task_review,
            provide_context=True,
        )

        validate = PythonOperator(
            task_id="validate_data",
            python_callable=_task_validate,
            provide_context=True,
        )

        deploy = PythonOperator(
            task_id="deploy",
            python_callable=_task_deploy,
            provide_context=True,
        )

        end = EmptyOperator(task_id="end")

        # Wire dependencies
        start >> parse >> generate >> review >> validate >> deploy >> end  # type: ignore[misc]

    return dag


def build_dag_from_metadata(
    metadata_path: str | Path,
    schedule: str | None = "@daily",
) -> Any:
    """Construct a DAG from a previously saved pipeline_metadata.json file.

    Args:
        metadata_path: Path to the JSON metadata file.
        schedule: Cron expression or Airflow preset.

    Returns:
        An Airflow DAG instance or a dict descriptor.
    """
    metadata = json.loads(Path(metadata_path).read_text())
    summary = metadata.get("summary", "AI-generated ETL pipeline")
    dag_id = f"etl_{summary[:40].lower().replace(' ', '_').replace('-', '_')}"
    return build_dag(dag_id=dag_id, requirement_text=summary, schedule=schedule)
