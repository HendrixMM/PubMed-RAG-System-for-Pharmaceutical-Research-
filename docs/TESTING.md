# Testing Strategy

Reference checklist for validating code, docs, and data pipelines.

## Automated Suites

| Command                                      | Description                                    |
| -------------------------------------------- | ---------------------------------------------- |
| `pytest`                                     | Core unit and integration tests under `tests/` |
| `pytest tests/monitoring`                    | Monitoring-only smoke tests                    |
| `python scripts/quality_check.py`            | Lint + formatting summary                      |
| `python scripts/nvidia_build_api_test.py`    | API connectivity + sample embedding call       |
| `python scripts/nvidia_build_rerank_test.py` | Reranking exercise via OpenAI wrapper          |

## Pre-Commit Checklist

- Run `pytest -m "not slow"` before every PR.
- Execute `mkdocs build --strict` to validate documentation links.
- For benchmark changes, run `python scripts/run_pharmaceutical_benchmarks.py --simulate`.

## Test Data

- Synthetic fixtures stored under `tests/fixtures/`.
- Benchmarks reference datasets under `Data/` (see [BENCHMARKS.md](BENCHMARKS.md)).

## Manual Validation

1. Launch the CLI (`python main.py --mode cli`) and ask two representative questions.
2. Review logs in `logs/latest.log` for warnings.
3. Capture results in the release checklist before tagging.
