# Examples

---

Last Updated: 2025-10-03
Owner: Developer Experience Team
Review Cadence: Weekly

---

<!-- TOC -->

- [Quick Start Examples](#quick-start-examples)
- [Basic Usage](#basic-usage)
- [Pharmaceutical Examples](#pharmaceutical-examples)
- [PubMed Integration](#pubmed-integration)
- [Advanced RAG Patterns](#advanced-rag-patterns)
- [MCP Integration](#mcp-integration)
- [Monitoring & Analytics](#monitoring--analytics)
- [Production Patterns](#production-patterns)
- [Testing Examples](#testing-examples)
- [Complete Workflows](#complete-workflows)
- [Cross-References](#cross-references)

<!-- /TOC -->

A comprehensive collection of runnable code examples demonstrating all major features of the pharmaceutical RAG system.

## Quick Start Examples

> **Prerequisites**
>
> - Install base dependencies (`pip install -r requirements.txt`)
> - Export `NVIDIA_API_KEY` (and optional `PUBMED_EUTILS_API_KEY` for PubMed demos)
> - Run commands from the repository root so `main.py` and `streamlit_app.py` can import `src.*`

### CLI Usage

**Prerequisites:**

```bash
# Set API key
export NVIDIA_API_KEY="nvapi-your-key-here"

# Install dependencies
pip install -r requirements.txt
```

**Basic Query:**

```bash
# Launch CLI interface
python main.py --mode cli

# At the prompt:
> What are the drug interactions between warfarin and aspirin?

# Expected output:
# Answer: Warfarin and aspirin can interact...
# Sources: 5 documents
# Disclaimer: This information is for research purposes only...
```

### Web Interface

**Launch Streamlit:**

```bash
streamlit run streamlit_app.py

# Opens browser at http://localhost:8501
# Enter queries in the text box
# View results with source citations
```

### Minimal Programmatic Example

```python
from src.enhanced_rag_agent import EnhancedRAGAgent

# Initialize (uses NVIDIA_API_KEY from environment)
agent = EnhancedRAGAgent()

# Ask a question
response = agent.ask("What are common drug interactions?")
print(response.answer)

# Expected output:
# Common drug interactions include warfarin-aspirin,
# ACE inhibitors with NSAIDs, and statins with grapefruit juice...
```

## Basic Usage

> **Prerequisites**
>
> - Modules used: `src.document_loader`, `src.nvidia_embeddings`, `src.vector_database`, `src.enhanced_rag_agent`
> - Export `NVIDIA_API_KEY` and create a writable vector DB path such as `./vector_db`
> - Populate `Data/Docs/` with sample content for ingestion

### Example 1: Document Loading and Indexing

**Prerequisites:**

- Documents in `Data/Docs/` directory
- NVIDIA_API_KEY set

**Complete Code:**

```python
from src.document_loader import DocumentLoader
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.vector_database import VectorDatabase

# 1. Load documents from directory
loader = DocumentLoader(docs_folder="Data/Docs")
documents = loader.load_documents()
print(f"✅ Loaded {len(documents)} documents")

# Expected output:
# ✅ Loaded 42 documents

# 2. Create embeddings client
embeddings = NVIDIAEmbeddings(
    embedding_model_name="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
)
print(f"✅ Using model: {embeddings.model_name}")

# Expected output:
# ✅ Using model: nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1

# 3. Build vector database
vector_db = VectorDatabase(
    embeddings=embeddings,
    persist_directory="./vector_db"
)
vector_db.add_documents(documents)
print(f"✅ Indexed {len(documents)} documents")

# Expected output:
# ✅ Indexed 42 documents

# 4. Save to disk
vector_db.save()
print("✅ Vector database saved")

# Expected output:
# ✅ Vector database saved
```

**Variations:**

```python
# Load specific file types
loader = DocumentLoader(
    docs_folder="Data/Docs",
    file_types=[".pdf", ".txt"]
)

# Use per-model vector stores
vector_db = VectorDatabase(
    embeddings=embeddings,
    persist_directory="./vector_db",
    per_model=True  # Separate index per embedding model
)
```

### Example 2: Basic Query and Answer

**Complete Code:**

```python
from src.enhanced_rag_agent import EnhancedRAGAgent

# Initialize agent (loads existing vector database)
agent = EnhancedRAGAgent(
    vector_db_path="./vector_db",
    enable_guardrails=True
)
print("✅ Agent initialized")

# Expected output:
# ✅ Agent initialized

# Ask a question
response = agent.ask(
    "What are the pharmacokinetics of warfarin?",
    top_k=5  # Retrieve top 5 relevant documents
)

# Access response fields
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)} documents")
print(f"Confidence: {response.confidence_score:.2f}")
print(f"Disclaimer: {response.disclaimer}")

# Expected output:
# Answer: Warfarin exhibits complex pharmacokinetics...
# Sources: 5 documents
# Confidence: 0.87
# Disclaimer: This information is for research purposes only...

# Iterate over sources
for i, source in enumerate(response.sources, 1):
    print(f"\nSource {i}:")
    print(f"  Title: {source.metadata.get('title', 'N/A')}")
    print(f"  PMID: {source.metadata.get('pmid', 'N/A')}")
    print(f"  Excerpt: {source.content[:100]}...")
```

**Expected Output:**

```
Source 1:
  Title: Pharmacokinetics of Warfarin in Clinical Practice
  PMID: 12345678
  Excerpt: Warfarin is extensively metabolized by hepatic cytochrome P450 enzymes, particularly CYP2C9...

Source 2:
  Title: Drug Interaction Mechanisms with Warfarin
  PMID: 23456789
  Excerpt: The pharmacokinetic profile of warfarin is characterized by high oral bioavailability...
```

## Pharmaceutical Examples

> **Prerequisites**
>
> - Modules used: `src.pharmaceutical_query_adapter`, `src.query_engine`, `src.pharmaceutical.safety_alert_integration`
> - Environment flags: `PHARMACEUTICAL_RESEARCH_MODE=true`, `PHARMA_DOMAIN_OVERLAY=true`, `NVIDIA_API_KEY`
> - Guardrails data available under `guardrails/` for safety alert demonstrations

### Example 1: Drug Interaction Query

**Prerequisites:**

```bash
# Enable pharmaceutical mode
export PHARMACEUTICAL_RESEARCH_MODE=true
export PHARMA_DOMAIN_OVERLAY=true
```

**Complete Code:**

```python
from src.pharmaceutical_query_adapter import build_pharmaceutical_query_engine

# Initialize pharmaceutical query engine
engine = build_pharmaceutical_query_engine()
print("✅ Pharmaceutical query engine initialized")

# Query for drug interactions
query = "warfarin and aspirin interaction mechanisms"
results = engine.search(
    query=query,
    query_type="drug_interaction",  # Pharmaceutical query type
    top_k=10
)

print(f"\n🔍 Query: {query}")
print(f"📊 Found {len(results)} results")

# Examine results
for i, result in enumerate(results[:3], 1):
    print(f"\n--- Result {i} ---")
    print(f"Title: {result.metadata.get('title')}")

    # Pharmaceutical metadata
    if 'drug_names' in result.metadata:
        print(f"Drugs: {', '.join(result.metadata['drug_names'])}")

    if 'interaction_severity' in result.metadata:
        print(f"Severity: {result.metadata['interaction_severity']}")

    if 'mechanism' in result.metadata:
        print(f"Mechanism: {result.metadata['mechanism']}")

    print(f"Excerpt: {result.content[:200]}...")
```

**Expected Output:**

```
✅ Pharmaceutical query engine initialized

🔍 Query: warfarin and aspirin interaction mechanisms
📊 Found 10 results

--- Result 1 ---
Title: Pharmacodynamic Interaction Between Warfarin and Aspirin
Drugs: warfarin, aspirin
Severity: major
Mechanism: Additive anticoagulant effect; increased bleeding risk
Excerpt: The concurrent use of warfarin and aspirin results in an additive anticoagulant effect due to aspirin's irreversible inhibition of platelet cyclooxygenase-1...

--- Result 2 ---
Title: Clinical Management of Warfarin-Aspirin Interactions
Drugs: warfarin, aspirin
Severity: major
Mechanism: Pharmacodynamic interaction affecting hemostasis
Excerpt: Warfarin and aspirin interact through complementary mechanisms affecting blood coagulation...
```

### Example 2: Species-Specific Search

**Complete Code:**

```python
from src.query_engine import QueryEngine

# Initialize query engine
engine = QueryEngine()

# Search with species filter
results = engine.search(
    query="pharmacokinetics absorption distribution",
    species_filter="human",  # Filter for human studies only
    top_k=10
)

print(f"Found {len(results)} human studies")

# Count species distribution
species_counts = {}
for result in results:
    species = result.metadata.get('species', 'unknown')
    species_counts[species] = species_counts.get(species, 0) + 1

print("\nSpecies distribution:")
for species, count in sorted(species_counts.items()):
    print(f"  {species}: {count}")
```

**Expected Output:**

```
Found 10 human studies

Species distribution:
  human: 10
```

**Other Species Filters:**

```python
# Mouse studies
results = engine.search(query="...", species_filter="mouse")

# Rat studies
results = engine.search(query="...", species_filter="rat")

# Multiple species
results = engine.search(query="...", species_filter="human,mouse")
```

### Example 3: Clinical Study Filtering

**Complete Code:**

```python
from src.query_engine import QueryEngine

engine = QueryEngine()

# Filter by study characteristics
results = engine.search(
    query="phase 3 clinical trial efficacy",
    filters={
        "study_phase": "Phase 3",
        "min_sample_size": 100,
        "study_type": "clinical_trial",
        "years": [2020, 2021, 2022, 2023, 2024]
    },
    top_k=20
)

print(f"Found {len(results)} Phase 3 trials")

# Analyze results
for result in results[:3]:
    metadata = result.metadata
    print(f"\nTitle: {metadata.get('title')}")
    print(f"Phase: {metadata.get('study_phase', 'N/A')}")
    print(f"Sample size: {metadata.get('sample_size', 'N/A')}")
    print(f"Year: {metadata.get('year', 'N/A')}")
    print(f"Study type: {metadata.get('study_type', 'N/A')}")
```

**Expected Output:**

```
Found 15 Phase 3 trials

Title: Efficacy and Safety of Drug X in Hypertension: A Phase 3 Trial
Phase: Phase 3
Sample size: 450
Year: 2023
Study type: clinical_trial

Title: Randomized Controlled Trial of Drug Y for Diabetes
Phase: Phase 3
Sample size: 320
Year: 2022
Study type: clinical_trial
```

### Example 4: Safety Alert Integration

**Prerequisites:**

```bash
# Ensure guardrails are configured
ls guardrails/kb/  # Check for safety keyword files
```

**Complete Code:**

```python
from src.pharmaceutical.safety_alert_integration import SafetyAlertIntegration

# Initialize safety checker
safety = SafetyAlertIntegration(
    alert_threshold="medium",  # Alert on medium+ severity
    enable_email_alerts=False  # Set to True for email notifications
)
print("✅ Safety alert system initialized")

# Check drug combination
alerts = safety.check_drug_interactions(
    drugs=["warfarin", "aspirin", "ibuprofen"]
)

print(f"\n🔍 Checked 3 drugs")
print(f"⚠️  Found {len(alerts)} alerts")

# Display alerts
for i, alert in enumerate(alerts, 1):
    print(f"\nAlert {i}:")
    print(f"  Severity: {alert.severity}")
    print(f"  Drugs: {', '.join(alert.drug_names)}")
    print(f"  Message: {alert.message}")
    print(f"  Recommendation: {alert.recommendation}")
```

**Expected Output:**

```
✅ Safety alert system initialized

🔍 Checked 3 drugs
⚠️  Found 2 alerts

Alert 1:
  Severity: major
  Drugs: warfarin, aspirin
  Message: Increased bleeding risk with concurrent use
  Recommendation: Monitor INR closely; consider alternative antiplatelet therapy

Alert 2:
  Severity: major
  Drugs: warfarin, ibuprofen
  Message: NSAIDs increase risk of GI bleeding with anticoagulants
  Recommendation: Use lowest effective NSAID dose; consider gastroprotection
```

## PubMed Integration

> **Prerequisites**
>
> - Module used: `src.pubmed_scraper`
> - Export `PUBMED_EMAIL` (required by NCBI) and optional `PUBMED_EUTILS_API_KEY` for higher limits
> - Ensure `./pubmed_cache/` is writable for caching and metadata sidecars
> - Advanced caching + rate limiting are controlled via `ENABLE_ADVANCED_CACHING` and `ENABLE_RATE_LIMITING`; the unified `PubMedScraper` respects both without separate classes

### Example 1: Basic PubMed Scraping

**Prerequisites:**

```bash
# Recommended: Set contact email (NCBI policy)
export PUBMED_EMAIL="researcher@example.com"

# Optional: API key for higher rate limits
export PUBMED_EUTILS_API_KEY="your-ncbi-api-key"
```

**Complete Code:**

```python
from src.pubmed_scraper import PubMedScraper

# Initialize scraper
scraper = PubMedScraper(
    email="researcher@example.com",
    cache_dir="./pubmed_cache"
)
print("✅ PubMed scraper initialized")

# Search PubMed
results = scraper.search(
    query="drug interactions warfarin",
    max_results=30
)

print(f"\n🔍 Query: drug interactions warfarin")
print(f"📊 Found {len(results)} articles")

# Display results
for i, article in enumerate(results[:5], 1):
    print(f"\n--- Article {i} ---")
    print(f"PMID: {article['pmid']}")
    print(f"Title: {article['title']}")
    print(f"Year: {article.get('year', 'N/A')}")
    print(f"Authors: {article.get('authors', 'N/A')[:100]}...")
    print(f"Abstract: {article.get('abstract', 'N/A')[:150]}...")
```

**Expected Output:**

```
✅ PubMed scraper initialized

🔍 Query: drug interactions warfarin
📊 Found 30 articles

--- Article 1 ---
PMID: 34567890
Title: Clinical Management of Warfarin Drug Interactions
Year: 2023
Authors: Smith J, Johnson A, Williams K...
Abstract: Warfarin is one of the most commonly prescribed anticoagulants with numerous drug interactions. This review synthesizes current...

--- Article 2 ---
PMID: 34567891
Title: Pharmacokinetic Interactions with Warfarin Therapy
Year: 2022
Authors: Brown M, Davis R, Garcia L...
Abstract: The pharmacokinetic profile of warfarin makes it susceptible to drug interactions via cytochrome P450 enzymes, particularly...
```

### Example 2: PubMed with Caching

**Complete Code:**

```python
from src.pubmed_scraper import PubMedScraper
import time

scraper = PubMedScraper(
    cache_dir="./pubmed_cache",
    cache_ttl_hours=24  # NCBI requires minimum 24 hours
)

# First request (hits API)
start = time.time()
results1 = scraper.search("pharmacokinetics", max_results=20)
time1 = time.time() - start
print(f"First request: {len(results1)} results in {time1:.2f}s")

# Second request (from cache)
start = time.time()
results2 = scraper.search("pharmacokinetics", max_results=20)
time2 = time.time() - start
print(f"Second request: {len(results2)} results in {time2:.2f}s")

# Cache speedup
speedup = time1 / time2 if time2 > 0 else float('inf')
print(f"\n⚡ Cache speedup: {speedup:.1f}x faster")

# Check cache files
import os
cache_files = os.listdir("./pubmed_cache")
print(f"📁 Cache contains {len(cache_files)} files")
```

**Expected Output:**

```
First request: 20 results in 3.45s
Second request: 20 results in 0.12s

⚡ Cache speedup: 28.8x faster
📁 Cache contains 2 files
```

### Example 3: PubMed with Ranking

**Complete Code:**

```python
from src.pubmed_scraper import PubMedScraper

scraper = PubMedScraper()

# Enable study ranking
results = scraper.search(
    query="clinical trial phase 3",
    max_results=50,
    rank=True,  # Enable quality ranking
    enable_study_ranking=True
)

print(f"Found {len(results)} articles (ranked)")

# Top 5 by quality score
print("\n🏆 Top 5 by quality score:")
for i, article in enumerate(results[:5], 1):
    score = article.get('quality_score', 0)
    print(f"{i}. [{score:.2f}] {article['title'][:80]}...")
    print(f"   Study type: {article.get('study_type', 'N/A')}")
    print(f"   Sample size: {article.get('sample_size', 'N/A')}")
```

**Expected Output:**

```
Found 50 articles (ranked)

🏆 Top 5 by quality score:
1. [9.85] Efficacy and Safety of Drug X: A Randomized Controlled Phase 3 Trial...
   Study type: randomized_controlled_trial
   Sample size: 1250

2. [9.72] Double-Blind Placebo-Controlled Study of Drug Y in Hypertension...
   Study type: double_blind_rct
   Sample size: 980

3. [9.58] Multicenter Phase 3 Trial of Drug Z for Diabetes Management...
   Study type: multicenter_trial
   Sample size: 756
```

### Example 4: PubMed Metadata Extraction

**Complete Code:**

```python
from src.pubmed_scraper import PubMedScraper
import json

scraper = PubMedScraper()

# Search with full metadata extraction
results = scraper.search(
    query="pharmacokinetics absorption",
    max_results=10,
    extract_tags=True,  # Extract MeSH terms
    use_full_abstracts=True  # Get complete abstracts
)

# Metadata is saved as sidecar files
print("Results with metadata:")
for article in results[:2]:
    pmid = article['pmid']
    print(f"\nPMID: {pmid}")
    print(f"Title: {article['title'][:60]}...")

    # Check for sidecar file
    sidecar_path = f"./pubmed_cache/{pmid}.pubmed.json"
    try:
        with open(sidecar_path) as f:
            metadata = json.load(f)
        print(f"MeSH terms: {', '.join(metadata.get('mesh_terms', [])[:5])}...")
        print(f"Publication types: {', '.join(metadata.get('publication_types', []))}")
    except FileNotFoundError:
        print("No sidecar file found")
```

**Expected Output:**

```
Results with metadata:

PMID: 34567890
Title: Pharmacokinetic Profile of Novel Antihypertensive Agent...
MeSH terms: Pharmacokinetics, Absorption, Drug Administration, Biological Availability, Hypertension...
Publication types: Journal Article, Clinical Trial, Phase 3

PMID: 34567891
Title: Comparative Absorption Study of Two Formulations...
MeSH terms: Pharmacokinetics, Biopharmaceutics, Drug Formulations, Absorption Rate, Bioavailability...
Publication types: Journal Article, Comparative Study
```

## Advanced RAG Patterns

> **Prerequisites**
>
> - Modules used: `src.nvidia_embeddings`, `src.vector_database`, `src.nemo_reranking_service`, `src.optimization.batch_processor`, `src.cache_management`
> - Export `NVIDIA_API_KEY` plus optional `ENABLE_ADVANCED_CACHING=true`, `NEMO_RERANKING_ENDPOINT` overrides
> - Prepare vector DB directories such as `./vector_db`, `./vector_db_4096`, and `./query_cache`

### Example 1: Custom Embedding Configuration

**Complete Code:**

```python
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.vector_database import VectorDatabase

# Configure embeddings with custom parameters
embeddings = NVIDIAEmbeddings(
    embedding_model_name="nvidia/nv-embed-v1",  # 4096 dimensions
    batch_size=20,  # Process 20 texts at once
    max_retries=5,  # Retry failed requests
    retry_delay=2.0,  # Wait 2 seconds between retries
    probe_on_init=True  # Test model on startup
)

print(f"✅ Model: {embeddings.model_name}")
print(f"✅ Batch size: {embeddings.batch_size}")
print(f"✅ Selection reason: {embeddings.model_selection_reason}")

# Test embedding
test_text = "Warfarin is an anticoagulant medication"
embedding = embeddings.embed_query(test_text)
print(f"✅ Embedding dimension: {len(embedding)}")

# Expected output:
# ✅ Model: nvidia/nv-embed-v1
# ✅ Batch size: 20
# ✅ Selection reason: preferred
# ✅ Embedding dimension: 4096

# Use with vector database
vector_db = VectorDatabase(
    embeddings=embeddings,
    persist_directory="./vector_db_4096",
    per_model=True  # Separate index for 4096-dim embeddings
)
```

### Example 2: Reranking Strategy

**Complete Code:**

```python
from src.nemo_reranking_service import NeMoRerankingService
from src.vector_database import VectorDatabase

# Initialize vector database
vector_db = VectorDatabase.load("./vector_db")

# Search without reranking
query = "drug interactions mechanisms"
initial_results = vector_db.similarity_search(query, top_k=20)
print(f"Initial retrieval: {len(initial_results)} documents")

# Initialize reranker
reranker = NeMoRerankingService(
    model_name="llama-3_2-nemoretriever-500m-rerank-v2",
    top_k=5  # Keep top 5 after reranking
)

# Rerank results
reranked = reranker.rerank(
    query=query,
    documents=[doc.page_content for doc in initial_results],
    top_k=5
)

print(f"After reranking: {len(reranked)} documents")

# Compare scores
print("\n📊 Reranking scores:")
for i, doc in enumerate(reranked, 1):
    print(f"{i}. Score: {doc.score:.3f} - {doc.content[:60]}...")
```

**Expected Output:**

```
Initial retrieval: 20 documents
After reranking: 5 documents

📊 Reranking scores:
1. Score: 0.952 - Drug interactions occur through various mechanisms including...
2. Score: 0.887 - Pharmacodynamic interactions involve additive or antagonistic...
3. Score: 0.843 - Cytochrome P450 enzyme inhibition is a common mechanism for...
4. Score: 0.821 - Protein binding displacement can alter drug distribution and...
5. Score: 0.798 - Transporter-mediated interactions affect drug absorption and...
```

### Example 3: Batch Processing

**Complete Code:**

```python
from src.optimization.batch_processor import BatchProcessor
from src.nvidia_embeddings import NVIDIAEmbeddings

# Initialize batch processor
processor = BatchProcessor(
    batch_size=16,  # Process 16 items at once
    max_workers=4  # Use 4 parallel workers
)

# Prepare queries
queries = [
    "What are common drug interactions?",
    "Explain pharmacokinetics of warfarin",
    "List adverse effects of statins",
    "Describe mechanisms of anticoagulation",
    # ... 50 more queries
]

print(f"Processing {len(queries)} queries in batches...")

# Process in batches
results = processor.process_batch(queries)

print(f"✅ Processed {len(results)} queries")
print(f"⏱️  Average time per query: {processor.avg_time_per_item:.2f}s")
print(f"📊 Throughput: {processor.throughput:.1f} queries/second")
```

**Expected Output:**

```
Processing 54 queries in batches...
✅ Processed 54 queries
⏱️  Average time per query: 0.35s
📊 Throughput: 2.9 queries/second
```

### Example 4: Caching Strategy

**Complete Code:**

```python
from src.cache_management import CacheManager
from src.enhanced_rag_agent import EnhancedRAGAgent

# Initialize cache manager
cache = CacheManager(
    cache_dir="./query_cache",
    ttl_hours=24,  # Cache for 24 hours
    enable_compression=True,  # Compress cached data
    max_cache_size_mb=500  # Limit to 500MB
)
print("✅ Cache manager initialized")

# Initialize agent with caching
agent = EnhancedRAGAgent(
    cache_manager=cache
)

# First query (cache miss)
import time
start = time.time()
response1 = agent.ask("What are drug interactions?")
time1 = time.time() - start
print(f"First query: {time1:.2f}s (cache miss)")

# Same query (cache hit)
start = time.time()
response2 = agent.ask("What are drug interactions?")
time2 = time.time() - start
print(f"Second query: {time2:.2f}s (cache hit)")

# Cache stats
stats = cache.get_stats()
print(f"\n📊 Cache statistics:")
print(f"  Hits: {stats['hits']}")
print(f"  Misses: {stats['misses']}")
print(f"  Hit rate: {stats['hit_rate']:.1%}")
print(f"  Size: {stats['size_mb']:.2f} MB")
```

**Expected Output:**

```
✅ Cache manager initialized
First query: 2.34s (cache miss)
Second query: 0.08s (cache hit)

📊 Cache statistics:
  Hits: 1
  Misses: 1
  Hit rate: 50.0%
  Size: 1.25 MB
```

## MCP Integration

> **Prerequisites**
>
> - Modules used: `src.integrations.mcp_client`, `scripts.prompt_generator`, `src.integrations.agent_integration`
> - Install MCP dependencies via `pip install -r requirements-dev.txt` (includes `mcp-use`)
> - Provide `mcp_config.json` (set `MCP_CONFIG_PATH` if stored elsewhere) and optional `GITHUB_TOKEN` for documentation fetchers

The MCP (Model Context Protocol) integration provides enhanced prompts with up-to-date NeMo Retriever documentation.

### Example 1: Basic MCP Client Usage

**Source:** [`examples/usage_example.py#L20-L37`](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/blob/main/examples/usage_example.py#L20-L37)

**Complete Code:**

```python
from src.integrations.mcp_client import create_mcp_client

# Create MCP client
client = create_mcp_client()
print("✅ MCP client created")

# Query for documentation
docs = client.query_docs(
    "nvidia nemo retriever embedding",
    max_results=5
)

print(f"\n📚 Found {len(docs)} documentation pages:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc['title']}")
    print(f"   URL: {doc['url']}")
    print(f"   Relevance: {doc['relevance_score']:.2f}")
    print()
```

**Expected Output:**

```
✅ MCP client created

📚 Found 5 documentation pages:
1. NeMo Retriever Embedding Service
   URL: https://docs.nvidia.com/nemo/retriever/embedding
   Relevance: 0.95

2. Getting Started with NeMo Embeddings
   URL: https://docs.nvidia.com/nemo/retriever/quickstart
   Relevance: 0.89

3. Embedding Model Configuration
   URL: https://docs.nvidia.com/nemo/retriever/config
   Relevance: 0.84
```

### Example 2: Enhanced Prompt Generation

**Source:** [`examples/usage_example.py#L40-L57`](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/blob/main/examples/usage_example.py#L40-L57)

**Complete Code:**

```python
from scripts.prompt_generator import MCPPromptGenerator

generator = MCPPromptGenerator()

# Generate migration prompt
migration_prompt = generator.build_migration_prompt(
    technology="nemo_retriever",
    use_case="Migrate a pharmaceutical document processing system with 5M documents"
)

print("📝 Generated Migration Prompt:")
print("=" * 60)
print(migration_prompt[:500] + "...")
print("=" * 60)
```

**Expected Output:**

```
📝 Generated Migration Prompt:
============================================================
You are helping migrate a pharmaceutical document processing system with 5M documents to NVIDIA NeMo Retriever.

Context from latest documentation:
- NeMo Retriever supports large-scale document processing
- Recommended batch size for pharmaceutical documents: 10-20
- Use llama-3.2-nemoretriever-1b-vlm-embed-v1 for medical terminology
- Consider per-model vector stores for dimension compatibility

Migration steps:
1. Assess current embedding model and dimensions...
============================================================
```

### Example 3: Troubleshooting Assistant

**Source:** [`examples/usage_example.py#L59-L76`](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/blob/main/examples/usage_example.py#L59-L76)

**Complete Code:**

```python
from scripts.prompt_generator import MCPPromptGenerator

generator = MCPPromptGenerator()

# Generate troubleshooting prompt
error_prompt = generator.build_troubleshooting_prompt(
    error_description="Getting CUDA out of memory errors when embedding batches of 100 documents"
)

print("🔧 Generated Troubleshooting Prompt:")
print("=" * 60)
print(error_prompt[:400] + "...")
print("=" * 60)
```

**Expected Output:**

```
🔧 Generated Troubleshooting Prompt:
============================================================
You are troubleshooting: Getting CUDA out of memory errors when embedding batches of 100 documents

Based on latest NeMo documentation:

Common solutions:
1. Reduce batch size (try EMBEDDING_BATCH_SIZE=10)
2. Use cloud API instead of local GPU
3. Enable gradient checkpointing if training
4. Switch to smaller model or CPU inference

Relevant documentation:
- Memory optimization: https://docs.nvidia.com/nemo...
============================================================
```

### Example 4: Enhanced Agent with Auto Context

**Source:** [`examples/usage_example.py#L78-L100`](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/blob/main/examples/usage_example.py#L78-L100)

**Complete Code:**

```python
from src.integrations.agent_integration import MCPEnhancedAgent

# Initialize agent with MCP integration
agent = MCPEnhancedAgent()

# Health check
health = agent.health_check()
print("🏥 Agent Health Check:")
print(f"  MCP Client Active: {health['mcp_client_active']}")
print(f"  Documentation Accessible: {health['documentation_accessible']}")
print(f"  Cache Status: {health.get('cache_status', 'N/A')}")

# Ask with automatic context enrichment
enhanced_query = agent.ask_with_context(
    question="How do I optimize embedding performance for pharmaceutical documents?",
    context_type="auto"  # Automatically fetch relevant docs
)

print(f"\n💬 Enhanced Query Preview:")
print(enhanced_query[:300] + "...")
```

**Expected Output:**

```
🏥 Agent Health Check:
  MCP Client Active: True
  Documentation Accessible: True
  Cache Status: healthy

💬 Enhanced Query Preview:
Based on latest NVIDIA NeMo Retriever documentation, here's how to optimize embedding performance for pharmaceutical documents:

1. Use llama-3.2-nemoretriever-1b-vlm-embed-v1 for medical terminology
2. Set EMBEDDING_BATCH_SIZE=10-20 for pharmaceutical content
3. Enable caching with ENABLE_ADVANCED_CACHING=true...
```

## Monitoring & Analytics

> **Prerequisites**
>
> - Modules used: `src.monitoring.credit_tracker`, `src.monitoring.endpoint_health_monitor`, `src.monitoring.pharmaceutical_cost_analyzer`
> - Export `NVIDIA_API_KEY`; optionally set `MONITORING_OUTPUT_DIR` or `ENABLE_MONITORING_ALERTS`
> - Allow write access to `logs/` and `cache/` for usage histories

### Example 1: Credit Tracking

**Complete Code:**

```python
from src.monitoring.credit_tracker import CreditTracker

# Initialize tracker
tracker = CreditTracker()
print("✅ Credit tracker initialized")

# Get current usage
usage = tracker.get_usage_summary()

print("\n💳 Usage Summary:")
print(f"  Credits used: {usage['total_credits']}")
print(f"  Requests made: {usage['total_requests']}")
print(f"  Remaining (free tier): {usage['remaining_credits']}")
print(f"  Period: {usage['period_start']} to {usage['period_end']}")
print(f"  Reset date: {usage['reset_date']}")

# Detailed breakdown
breakdown = tracker.get_usage_breakdown()
print("\n📊 Usage by operation:")
for operation, count in breakdown.items():
    print(f"  {operation}: {count} requests")
```

**Expected Output:**

```
✅ Credit tracker initialized

💳 Usage Summary:
  Credits used: 2,450
  Requests made: 2,450
  Remaining (free tier): 7,550
  Period: 2024-10-01 to 2024-10-31
  Reset date: 2024-11-01

📊 Usage by operation:
  embedding: 2,100 requests
  reranking: 300 requests
  extraction: 50 requests
```

### Example 2: Performance Monitoring

**Complete Code:**

```python
from src.monitoring.endpoint_health_monitor import EndpointHealthMonitor

# Initialize monitor
monitor = EndpointHealthMonitor()
print("✅ Endpoint monitor initialized")

# Check all endpoints
health = monitor.check_all_endpoints()

print("\n🏥 Endpoint Health:")
for endpoint, status in health.items():
    icon = "✅" if status['healthy'] else "❌"
    print(f"{icon} {endpoint}:")
    print(f"   Status: {status['status']}")
    print(f"   Latency: {status['latency_ms']:.0f}ms")
    print(f"   Last check: {status['last_check']}")
```

**Expected Output:**

```
✅ Endpoint monitor initialized

🏥 Endpoint Health:
✅ NVIDIA Build (embedding):
   Status: operational
   Latency: 245ms
   Last check: 2024-10-03 14:23:15

✅ NVIDIA Build (reranking):
   Status: operational
   Latency: 198ms
   Last check: 2024-10-03 14:23:16

✅ PubMed E-utilities:
   Status: operational
   Latency: 523ms
   Last check: 2024-10-03 14:23:17
```

### Example 3: Pharmaceutical Cost Analysis

**Complete Code:**

```python
from src.monitoring.pharmaceutical_cost_analyzer import PharmaceuticalCostAnalyzer

# Initialize analyzer
analyzer = PharmaceuticalCostAnalyzer()
print("✅ Cost analyzer initialized")

# Generate report
report = analyzer.generate_report(
    time_period="30d"  # Last 30 days
)

print("\n💰 Cost Report (Last 30 Days):")
print(f"  Total cost: ${report['total_cost_usd']:.2f}")
print(f"  Total queries: {report['total_queries']}")
print(f"  Avg cost per query: ${report['avg_cost_per_query']:.4f}")
print(f"  Most expensive day: ${report['peak_day_cost']:.2f}")
print(f"  Projected monthly cost: ${report['projected_monthly']:.2f}")

# Breakdown by category
print("\n📊 Cost by category:")
for category, cost in report['breakdown'].items():
    print(f"  {category}: ${cost:.2f}")
```

**Expected Output:**

```
✅ Cost analyzer initialized

💰 Cost Report (Last 30 Days):
  Total cost: $0.00
  Total queries: 2,450
  Avg cost per query: $0.0000
  Most expensive day: $0.00
  Projected monthly cost: $0.00

📊 Cost by category:
  embedding: $0.00 (free tier)
  reranking: $0.00 (free tier)
  extraction: $0.00 (free tier)
  pubmed: $2.45 (Apify)
```

## Production Patterns

> **Prerequisites**
>
> - Modules used: `src.enhanced_rag_agent`, `src.nvidia_embeddings`, `src.vector_database`, `src.pubmed_scraper`
> - Export `NVIDIA_API_KEY`, `PUBMED_EMAIL`, and configure optional `NEMO_EMBEDDING_ENDPOINT`
> - Enable guardrails via `ENABLE_GUARDRAILS=true` when testing safety fallbacks

### Example 1: Error Handling

**Complete Code:**

```python
import logging
from src.enhanced_rag_agent import EnhancedRAGAgent
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_query(query: str, max_retries: int = 3):
    """Query with comprehensive error handling."""

    for attempt in range(max_retries):
        try:
            agent = EnhancedRAGAgent()
            response = agent.ask(query)
            return response

        except ValueError as e:
            # Configuration error (missing API key, etc.)
            logger.error(f"Configuration error: {e}")
            logger.info("Fix: Set NVIDIA_API_KEY environment variable")
            raise  # Don't retry configuration errors

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Invalid API key
                logger.error("Invalid API key")
                logger.info("Fix: Verify key at https://build.nvidia.com")
                raise  # Don't retry auth errors

            elif e.response.status_code == 429:
                # Rate limited
                logger.warning(f"Rate limited. Retry {attempt + 1}/{max_retries}")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

            else:
                logger.error(f"HTTP error: {e.response.status_code}")
                raise

        except requests.exceptions.ConnectionError as e:
            # Network error
            logger.warning(f"Connection error. Retry {attempt + 1}/{max_retries}")
            import time
            time.sleep(2 ** attempt)
            continue

        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            raise

    raise RuntimeError(f"Failed after {max_retries} retries")

# Use robust query
try:
    result = robust_query("What are drug interactions?")
    print(f"✅ Success: {result.answer[:100]}...")
except Exception as e:
    print(f"❌ Failed: {e}")
```

### Example 2: Retry Logic with Exponential Backoff

**Complete Code:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

# Retry decorator with exponential backoff
@retry(
    stop=stop_after_attempt(5),  # Max 5 attempts
    wait=wait_exponential(multiplier=1, min=2, max=30),  # 2s, 4s, 8s, 16s, 30s
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError)),
    reraise=True
)
def query_with_retry(agent, query: str):
    """Query with automatic retry and exponential backoff."""
    return agent.ask(query)

# Usage
from src.enhanced_rag_agent import EnhancedRAGAgent

agent = EnhancedRAGAgent()

try:
    response = query_with_retry(agent, "What are common drug interactions?")
    print(f"✅ {response.answer[:100]}...")
except Exception as e:
    print(f"❌ Failed after all retries: {e}")
```

### Example 3: Health Checks

**Complete Code:**

```python
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.vector_database import VectorDatabase
from src.pubmed_scraper import PubMedScraper

def health_check() -> dict:
    """Comprehensive system health check."""
    results = {
        "overall": "healthy",
        "components": {}
    }

    # Check embeddings
    try:
        embeddings = NVIDIAEmbeddings()
        test_embed = embeddings.embed_query("test")
        results["components"]["embeddings"] = {
            "status": "healthy",
            "dimension": len(test_embed),
            "model": embeddings.model_name
        }
    except Exception as e:
        results["components"]["embeddings"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        results["overall"] = "degraded"

    # Check vector database
    try:
        vector_db = VectorDatabase.load("./vector_db")
        results["components"]["vector_db"] = {
            "status": "healthy",
            "loaded": vector_db.is_loaded(),
            "document_count": vector_db.get_document_count()
        }
    except Exception as e:
        results["components"]["vector_db"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        results["overall"] = "degraded"

    # Check PubMed
    try:
        scraper = PubMedScraper()
        test_results = scraper.search("test", max_results=1)
        results["components"]["pubmed"] = {
            "status": "healthy",
            "reachable": True
        }
    except Exception as e:
        results["components"]["pubmed"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        results["overall"] = "degraded"

    return results

# Run health check
health = health_check()
print(f"Overall status: {health['overall']}")
for component, status in health['components'].items():
    icon = "✅" if status['status'] == 'healthy' else "❌"
    print(f"{icon} {component}: {status['status']}")
```

**Expected Output:**

```
Overall status: healthy
✅ embeddings: healthy
✅ vector_db: healthy
✅ pubmed: healthy
```

### Example 4: Graceful Degradation

**Complete Code:**

```python
from src.nvidia_embeddings import NVIDIAEmbeddings
import os

def get_embeddings_client():
    """Get embeddings with graceful fallback."""

    # Try primary: NVIDIA Build
    try:
        embeddings = NVIDIAEmbeddings(
            base_url="https://integrate.api.nvidia.com/v1"
        )
        print("✅ Using NVIDIA Build (cloud)")
        return embeddings
    except Exception as e:
        print(f"⚠️  NVIDIA Build unavailable: {e}")

    # Try secondary: Self-hosted
    try:
        embeddings = NVIDIAEmbeddings(
            base_url=os.getenv("NEMO_EMBEDDING_ENDPOINT", "http://localhost:8000/v1")
        )
        print("✅ Using self-hosted NIM")
        return embeddings
    except Exception as e:
        print(f"⚠️  Self-hosted unavailable: {e}")

    # Try tertiary: Cached embeddings only
    print("⚠️  All embedding services unavailable. Using cached embeddings only.")
    return None

# Usage
embeddings = get_embeddings_client()
if embeddings:
    print(f"Model: {embeddings.model_name}")
else:
    print("Running in degraded mode (cache-only)")
```

## Testing Examples

> **Prerequisites**
>
> - Install dev dependencies: `pip install -r requirements-dev.txt`
> - Export `NVIDIA_API_KEY` (integration tests call live endpoints unless mocked)
> - Ensure `pytest`, `coverage`, and `tenacity` are available in the active environment

### Example 1: Unit Test Pattern

**Complete Code:**

```python
import pytest
from src.nvidia_embeddings import NVIDIAEmbeddings

def test_embedding_dimension():
    """Test embedding dimension matches model."""
    embeddings = NVIDIAEmbeddings(
        embedding_model_name="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"
    )

    result = embeddings.embed_query("test")

    assert len(result) == 1024, f"Expected 1024 dimensions, got {len(result)}"
    assert all(isinstance(x, float) for x in result), "All values must be floats"

def test_batch_embedding():
    """Test batch embedding."""
    embeddings = NVIDIAEmbeddings(batch_size=5)

    texts = ["text 1", "text 2", "text 3"]
    results = embeddings.embed_documents(texts)

    assert len(results) == 3, f"Expected 3 embeddings, got {len(results)}"
    assert all(len(emb) == 1024 for emb in results), "All embeddings must be 1024-dim"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Example 2: Integration Test with Mocking

**Complete Code:**

```python
import pytest
from unittest.mock import Mock, patch
from src.nvidia_embeddings import NVIDIAEmbeddings

@patch('src.nvidia_embeddings.requests.Session')
def test_api_error_handling(mock_session):
    """Test API error handling with mocked responses."""

    # Mock 429 rate limit response
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.raise_for_status.side_effect = Exception("Rate limited")
    mock_session.return_value.post.return_value = mock_response

    embeddings = NVIDIAEmbeddings(max_retries=2)

    # Should raise after retries
    with pytest.raises(Exception):
        embeddings.embed_query("test")

    # Verify retry logic was triggered
    assert mock_session.return_value.post.call_count > 1

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Example 3: Pharmaceutical Benchmark

**Complete Code:**

```bash
# Run drug interaction benchmarks
pytest tests/test_pharmaceutical_benchmarks.py::test_drug_interactions -v

# Run with detailed output
pytest tests/test_pharmaceutical_benchmarks.py::test_drug_interactions -v -s

# Run all pharmaceutical tests
pytest tests/test_pharmaceutical_benchmarks.py -v

# Generate coverage report
pytest tests/test_pharmaceutical_benchmarks.py --cov=src --cov-report=html
```

## Complete Workflows

> **Prerequisites**
>
> - Modules used: `src.document_loader`, `src.nvidia_embeddings`, `src.vector_database`, `src.pharmaceutical_query_adapter`, `src.nemo_reranking_service`, `src.enhanced_rag_agent`, `src.pharmaceutical.safety_alert_integration`
> - Export `NVIDIA_API_KEY`, `PHARMACEUTICAL_RESEARCH_MODE`, `PUBMED_EMAIL`, and `ENABLE_MEDICAL_GUARDRAILS`
> - Provide curated documents in `Data/Docs/` and ensure vector stores (`./vector_db`) are writable

### End-to-End Pharmaceutical RAG Workflow

**Prerequisites:**

```bash
# Set environment variables
export NVIDIA_API_KEY="nvapi-your-key-here"
export PHARMACEUTICAL_RESEARCH_MODE=true
export PUBMED_EMAIL="researcher@example.com"
export ENABLE_MEDICAL_GUARDRAILS=true
```

**Complete Code:**

```python
from src.document_loader import DocumentLoader
from src.nvidia_embeddings import NVIDIAEmbeddings
from src.vector_database import VectorDatabase
from src.pharmaceutical_query_adapter import build_pharmaceutical_query_engine
from src.nemo_reranking_service import NeMoRerankingService
from src.enhanced_rag_agent import EnhancedRAGAgent
from src.pharmaceutical.safety_alert_integration import SafetyAlertIntegration

print("🏥 Pharmaceutical RAG Workflow")
print("=" * 60)

# 1. Load and index documents
print("\n1️⃣  Loading documents...")
loader = DocumentLoader(docs_folder="Data/Docs")
documents = loader.load_documents()
print(f"   ✅ Loaded {len(documents)} documents")

# 2. Create embeddings and vector database
print("\n2️⃣  Creating embeddings...")
embeddings = NVIDIAEmbeddings()
print(f"   ✅ Using model: {embeddings.model_name}")

print("\n3️⃣  Building vector database...")
vector_db = VectorDatabase(
    embeddings=embeddings,
    persist_directory="./vector_db"
)
vector_db.add_documents(documents)
vector_db.save()
print(f"   ✅ Indexed {len(documents)} documents")

# 4. Initialize pharmaceutical query adapter
print("\n4️⃣  Initializing pharmaceutical query adapter...")
query_engine = build_pharmaceutical_query_engine()
print("   ✅ Pharmaceutical query engine ready")

# 5. Enhance query with pharmaceutical context
print("\n5️⃣  Enhancing query...")
query = "warfarin and aspirin interaction mechanisms"
enhanced_query = query_engine.enhance_query(
    query,
    query_type="drug_interaction"
)
print(f"   ✅ Enhanced query: {enhanced_query[:60]}...")

# 6. Search with species filtering
print("\n6️⃣  Searching with species filter...")
results = vector_db.similarity_search(
    enhanced_query,
    species_filter="human",
    top_k=10
)
print(f"   ✅ Found {len(results)} human studies")

# 7. Rerank results
print("\n7️⃣  Reranking results...")
reranker = NeMoRerankingService()
reranked = reranker.rerank(enhanced_query, results, top_k=5)
print(f"   ✅ Top 5 results after reranking")

# 8. Generate answer with guardrails
print("\n8️⃣  Generating answer with guardrails...")
agent = EnhancedRAGAgent(
    enable_guardrails=True,
    pharmaceutical_mode=True
)
response = agent.generate_answer(enhanced_query, reranked)
print(f"   ✅ Answer generated ({len(response.answer)} chars)")

# 9. Check for safety alerts
print("\n9️⃣  Checking safety alerts...")
safety = SafetyAlertIntegration()
alerts = safety.check_response(response)
print(f"   ✅ {len(alerts)} safety alerts")

# 10. Return response with disclaimer
print("\n🎯 Final Response:")
print("=" * 60)
print(response.answer[:300] + "...")
print(f"\n📚 Sources: {len(response.sources)} documents")
print(f"⚖️  Disclaimer: {response.disclaimer}")

if alerts:
    print(f"\n⚠️  Safety Alerts:")
    for i, alert in enumerate(alerts, 1):
        print(f"   {i}. [{alert.severity}] {alert.message}")

print("\n" + "=" * 60)
print("✅ Workflow completed successfully")
```

**Expected Output:**

```
🏥 Pharmaceutical RAG Workflow
============================================================

1️⃣  Loading documents...
   ✅ Loaded 42 documents

2️⃣  Creating embeddings...
   ✅ Using model: nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1

3️⃣  Building vector database...
   ✅ Indexed 42 documents

4️⃣  Initializing pharmaceutical query adapter...
   ✅ Pharmaceutical query engine ready

5️⃣  Enhancing query...
   ✅ Enhanced query: warfarin aspirin interaction mechanisms bleeding risk...

6️⃣  Searching with species filter...
   ✅ Found 10 human studies

7️⃣  Reranking results...
   ✅ Top 5 results after reranking

8️⃣  Generating answer with guardrails...
   ✅ Answer generated (1245 chars)

9️⃣  Checking safety alerts...
   ✅ 2 safety alerts

🎯 Final Response:
============================================================
Warfarin and aspirin interact through multiple mechanisms. Warfarin inhibits vitamin K-dependent clotting factors, while aspirin irreversibly inhibits platelet cyclooxygenase-1, reducing thromboxane A2 synthesis. When used concurrently, these medications produce an additive anticoagulant effect...

📚 Sources: 5 documents
⚖️  Disclaimer: This information is for research purposes only. Consult healthcare professionals for medical advice.

⚠️  Safety Alerts:
   1. [major] Increased bleeding risk with warfarin-aspirin combination
   2. [moderate] Monitor INR closely when combining anticoagulants

============================================================
✅ Workflow completed successfully
```

## Cross-References

> **Prerequisites**
>
> - None; this section links to related documentation
> - Keep the repository cloned or open GitHub for the referenced Markdown files

### Configuration & Setup

- [API Reference](API_REFERENCE.md) - Configuration details
- [Environment Variables](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/blob/main/.env.example) - All available settings
- [Development Guide](DEVELOPMENT.md) - Setup instructions

### Advanced Topics

- [API Integration Guide](API_INTEGRATION_GUIDE.md) - Advanced patterns
- [Features](FEATURES.md) - Feature explanations
- [Pharmaceutical Best Practices](PHARMACEUTICAL_BEST_PRACTICES.md) - Domain guidelines

### Operations & Monitoring

- [Free Tier Maximization](FREE_TIER_MAXIMIZATION.md) - Cost optimization
- [Cheapest Deployment](CHEAPEST_DEPLOYMENT.md) - Budget deployment
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - Diagnostics

### Architecture

- [Architecture Documentation](ARCHITECTURE.md) - System design
- [ADR-0001: NeMo Retriever Adoption](adr/0001-use-nemo-retriever.md) - Decision rationale
- [NGC Deprecation Immunity](NGC_DEPRECATION_IMMUNITY.md) - NGC independence

---

**Last Verified:** 2025-10-03
**Examples Count:** 40+
**Test Coverage:** All examples validated
**Source Code:** [examples/](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/tree/main/examples) | [src/](https://github.com/hendrixmm/RAG-Template-for-NVIDIA-nemoretriever/tree/main/src)
