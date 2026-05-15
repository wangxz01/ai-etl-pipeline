"""CLI entry point — click commands for the AI ETL pipeline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from src.config import get_settings
from src.orchestrator.pipeline_orchestrator import PipelineOrchestrator, PipelinePhase
from src.parser.requirement_parser import RequirementParser

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = "DEBUG" if verbose else get_settings().log_level
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _read_input(input_text: str | None, input_file: str | None) -> str:
    """Resolve CLI input from a string, a file, or stdin."""
    if input_file:
        return Path(input_file).read_text().strip()
    if input_text:
        return input_text
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise click.UsageError("Provide --input TEXT, --input-file PATH, or pipe input via stdin.")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """AI ETL Pipeline — parse, generate, review, validate, and deploy ETL code."""
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_text", help="Natural language ETL requirement.")
@click.option("--input-file", "-f", type=click.Path(exists=True), help="Path to a JSON or text file.")
@click.option("--output", "-o", type=click.Path(), help="Write parsed output to this JSON file.")
def parse(input_text: str | None, input_file: str | None, output: str | None) -> None:
    """Parse a natural language ETL requirement into structured metadata."""
    text = _read_input(input_text, input_file)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Parsing requirement…", total=None)
        parser = RequirementParser()
        parsed = parser.parse(text)

    # Display reasoning trace
    if parsed.reasoning_trace:
        console.print(Panel("\n".join(f"[dim]Step {i+1}:[/] {s}" for i, s in enumerate(parsed.reasoning_trace)), title="Reasoning Trace", border_style="dim"))

    # Display summary table
    table = Table(title="Parsed Requirement", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Summary", parsed.summary)
    table.add_row("Data Sources", ", ".join(ds.name for ds in parsed.data_sources))
    table.add_row("Cleaning Rules", str(len(parsed.cleaning_rules)))
    table.add_row("Aggregations", str(len(parsed.aggregation_rules)))
    table.add_row("Output Format", parsed.output_format.format_type)
    table.add_row("Destination", parsed.output_format.destination)
    table.add_row("Glossary Terms", ", ".join(gt.term for gt in parsed.glossary_terms))
    console.print(table)

    if output:
        Path(output).write_text(parsed.model_dump_json(indent=2))
        console.print(f"\n[green]Output written to {output}[/green]")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_text", help="Natural language ETL requirement.")
@click.option("--input-file", "-f", type=click.Path(exists=True), help="Path to a JSON requirement file.")
@click.option("--output-dir", "-o", type=click.Path(), default="./output", help="Output directory.")
def generate(input_text: str | None, input_file: str | None, output_dir: str) -> None:
    """Generate SQL and Python ETL code from a parsed requirement."""
    from src.agents.code_gen_agent import CodeGenAgent

    text = _read_input(input_text, input_file)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task("Parsing and generating code…", total=None)

        parser = RequirementParser()
        parsed = parser.parse(text)

        agent = CodeGenAgent()
        result = agent.execute(requirement=parsed)

    if not result.success:
        console.print(f"[red]Generation failed:[/red] {result.errors}")
        raise SystemExit(1)

    code = result.payload["generated_code"]

    console.print(Panel(code["explanation"], title="Explanation", border_style="green"))
    console.print(Panel(Syntax(code["sql_code"], "sql", theme="monokai"), title="Generated SQL"))
    console.print(Panel(Syntax(code["python_code"], "python", theme="monokai"), title="Generated Python"))

    # Write to disk
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generated_query.sql").write_text(code["sql_code"])
    (out / "etl_script.py").write_text(code["python_code"])
    console.print(f"\n[green]Artifacts written to {out}/[/green]")


# ---------------------------------------------------------------------------
# run-pipeline
# ---------------------------------------------------------------------------

@cli.command(name="run-pipeline")
@click.option("--input", "-i", "input_text", help="Natural language ETL requirement.")
@click.option("--input-file", "-f", type=click.Path(exists=True), help="Path to a JSON requirement file.")
@click.option("--max-retries", type=int, default=None, help="Override max retry attempts.")
def run_pipeline(input_text: str | None, input_file: str | None, max_retries: int | None) -> None:
    """Execute the full multi-agent pipeline (parse → generate → review → validate → deploy)."""
    text = _read_input(input_text, input_file)

    def progress_cb(phase: PipelinePhase, step: str, fraction: float) -> None:
        console.print(f"  [{phase.value}] {step} ({fraction:.0%})")

    orchestrator = PipelineOrchestrator(
        max_retry_attempts=max_retries,
        progress_callback=progress_cb,
    )

    console.print(Panel(f"[bold]{text}[/bold]", title="Running Pipeline", border_style="cyan"))
    state = orchestrator.run(text)

    # Summary
    if state.phase.value == "COMPLETED":
        console.print("\n[bold green]Pipeline completed successfully![/bold green]")
        files = state.artifacts.get("output_files", {})
        if files:
            for label, path in files.items():
                console.print(f"  {label}: {path}")
    else:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {state.errors}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--source-table", "-s", required=True, help="Source table name.")
@click.option("--target-table", "-t", required=True, help="Target table name.")
@click.option("--sql-file", type=click.Path(exists=True), help="Path to the SQL file to validate against.")
@click.option("--python-file", type=click.Path(exists=True), help="Path to the Python ETL script.")
def validate(source_table: str, target_table: str, sql_file: str | None, python_file: str | None) -> None:
    """Validate ETL output data against source and business rules."""
    from src.agents.data_validator import DataValidatorAgent

    sql_code = Path(sql_file).read_text() if sql_file else "-- no SQL file provided"
    python_code = Path(python_file).read_text() if python_file else "# no Python file provided"

    agent = DataValidatorAgent()
    result = agent.execute(
        requirement_summary=f"Validate {source_table} → {target_table}",
        sql_code=sql_code,
        python_code=python_code,
    )

    if not result.success:
        console.print(f"[red]Validation failed:[/red] {result.errors}")
        raise SystemExit(1)

    validation = result.payload["validation"]
    status_style = {"pass": "green", "warn": "yellow", "fail": "red"}
    overall = validation["overall_status"]
    console.print(f"\nOverall Status: [{status_style.get(overall, 'white')}]{overall.upper()}[/{status_style.get(overall, 'white')}]")

    if validation.get("checks"):
        table = Table(title="Validation Checks", show_header=True, header_style="bold cyan")
        table.add_column("Check", style="bold")
        table.add_column("Status")
        table.add_column("Details")
        for check in validation["checks"]:
            style = status_style.get(check["status"], "white")
            table.add_row(check["name"], f"[{style}]{check['status']}[/{style}]", check["details"])
        console.print(table)

    if validation.get("recommendation"):
        console.print(Panel(validation["recommendation"], title="Recommendation"))


if __name__ == "__main__":
    cli()
