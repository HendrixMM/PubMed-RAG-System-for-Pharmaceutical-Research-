# Configuration Checklist

Use this checklist to align environment variables, feature flags, and service endpoints with your deployment target.

## Core Settings

| Variable                  | Purpose                                   | Default                                                   |
| ------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| `NVIDIA_API_KEY`          | Authenticates embedding + rerank APIs     | _required_                                                |
| `NEMO_EMBEDDING_ENDPOINT` | Override NeMo embedding base URL          | `https://ai.api.nvidia.com/v1`                            |
| `NEMO_RERANKING_ENDPOINT` | Override NeMo rerank endpoint             | `https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking` |
| `VECTOR_DB_PATH`          | Persistent vector database directory      | `./vector_db`                                             |
| `ENABLE_GUARDRAILS`       | Toggle safety rails in `EnhancedRAGAgent` | `true`                                                    |

## Feature Flags

- `ENABLE_ADVANCED_CACHING=true` for query result caching.
- `ENABLE_RERANK_MODEL_MAPPING=true` to map NeMo models to NVIDIA Build-compatible variants.
- `ENABLE_CLOUD_FIRST_RERANK=true` to prefer NVIDIA Build before local NeMo endpoints.

## Secrets Management

- Store API keys in a secrets manager or Vault; avoid committing `.env` files.
- Rotate keys monthly and record updates in `docs/security/key-rotation-tracker.md`.

## Environment Files

Create `.env` (local) and `.env.production` (deployment) containing:

```
NVIDIA_API_KEY=nvapi-your-key
ENABLE_ADVANCED_CACHING=true
VECTOR_DB_PATH=./vector_db
```

## Validation

Run the automated validators after every change:

```bash
python scripts/config_validator.py
python scripts/validate_env.py
```

Continue with [Monitoring](MONITORING.md) to ensure metrics and quotas are wired up.
