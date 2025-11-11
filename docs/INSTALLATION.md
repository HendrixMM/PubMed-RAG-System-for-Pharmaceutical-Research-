# Installation Guide

Minimal steps to set up the Pharmaceutical RAG Template locally or in CI.

## Prerequisites

- Python 3.11+
- `pip` and `virtualenv` (or `uv`, `poetry`)
- NVIDIA API credentials with embedding and reranking access
- Git, Docker (optional for self-hosted services)

## Setup Steps

1. **Clone and enter the repository**
   ```bash
   git clone https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever.git
   cd RAG-Template-for-NVIDIA-nemoretriever
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # for docs + tooling
   ```
4. **Configure environment variables**
   ```bash
   cp .env.example .env  # if available
   export NVIDIA_API_KEY="nvapi-your-key"
   export NGC_API_KEY="ngc-your-key"  # optional, for NeMo services
   ```
5. **Validate installation**
   ```bash
   python scripts/validate_env.py
   ```

## Next Steps

- Continue with [Configuration](CONFIGURATION.md) for advanced tuning.
- Run `python main.py --mode cli` for a smoke test before ingesting documents.
