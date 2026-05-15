# Examples

This directory contains sample inputs for the AI ETL pipeline.

## sample_request.json

A representative ETL request that asks the pipeline to:

> Get daily active users by region for the last 30 days, joined with user profile table, filter out test accounts, output as partitioned Parquet files.

### What's in the file

| Section                    | Purpose                                                       |
|----------------------------|---------------------------------------------------------------|
| `requirement`              | Natural language description (fed to the parser)              |
| `context`                  | Business context — who requested it, priority, etc.           |
| `data_sources`             | Schema hints for the two source tables                        |
| `output_specification`     | Format, destination, partitioning, and naming convention      |
| `validation_expectations`  | Expected row counts, null rate thresholds, business rules     |
| `schedule`                 | Airflow schedule hint (daily at 06:00 UTC)                    |

### How to use it

```bash
# Parse the requirement from the JSON file
ai-etl parse --input-file examples/sample_request.json

# Generate code from it
ai-etl generate --input-file examples/sample_request.json

# Run the full pipeline
ai-etl run-pipeline --input-file examples/sample_request.json
```

### Creating your own request

Copy `sample_request.json` and modify:

1. Replace the `requirement` string with your ETL task.
2. Update `data_sources` with your actual table schemas.
3. Adjust `output_specification` for your target format and location.
4. Add business rules to `validation_expectations` so the validator knows what to check.

The minimum required field is `requirement` — the rest is optional context that helps the agents produce more accurate output.
