# Keyword Lab 🔬

A powerful CLI tool that transforms seed topics into validated, ranked SEO keywords optimized for both traditional search and **Generative Engine Optimization (GEO)**. It extracts, validates, clusters, and scores keywords—giving you actionable content strategies.

## ✨ Features

### Core Capabilities
- **Keyword Discovery**: Extract long-tail keywords from documents, URLs, or SERP results
- **Google Autocomplete Validation**: Verify keywords against real search suggestions
- **Semantic Clustering**: Group keywords using Q&A-optimized embeddings
- **Parent Topic Assignment**: Automatic hub-spoke silo architecture via LLM
- **GEO-Centric Intent Classification**: 5 intent categories optimized for AI search

### New in v2.0
- 🔍 **Google Autocomplete Validation** - Confirm keywords have real search volume
- 📊 **Percentile Ranking** - Fairer scoring that preserves long-tail value
- 🧠 **GEO Intent Categories** - `direct_answer`, `complex_research`, `transactional`, `local`, `comparative`
- 🏗️ **Parent Topic Silos** - Automatic pillar page architecture
- 📝 **Content Brief Generation** - LLM-powered briefs with H1, questions, entities
- ⚠️ **Entity Co-Occurrence** - Detect thin content risk in clusters
- 🧹 **QA Validation** - Automated cleanup (cluster size, word count limits)
- 🚫 **Blacklist Filtering** - Exclude competitor/spam patterns
- 🔍 **PAA Extraction** - People Also Ask questions from SerpAPI
- ⚡ **Weighted HTML Extraction** - Priority extraction of H1-H3, meta, alt text

## 🚀 Quick Start

### Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"  # Includes sentence-transformers for better clustering
python -m nltk.downloader stopwords
```

### Basic Usage
```bash
# Run keyword discovery
keyword-lab run --seed-topic "best coffee beans" --audience "home baristas"

# Generate content brief for a cluster
keyword-lab brief keywords.json --cluster "cluster-0"

# Run QA validation
keyword-lab qa keywords.json --min-cluster-size 3 --max-words 6 --report
```

### Environment Variables
```bash
# Optional: Enable LLM features
export GEMINI_API_KEY="your-key"      # For Gemini (recommended)
export OPENAI_API_KEY="your-key"      # For OpenAI
export ANTHROPIC_API_KEY="your-key"   # For Anthropic
export SERPAPI_KEY="your-key"         # For PAA extraction
```

## 📋 Output Schema

Each keyword item includes:

| Field | Type | Description |
|-------|------|-------------|
| `keyword` | string | The keyword phrase (lowercase) |
| `cluster` | string | Semantic cluster name |
| `parent_topic` | string | Pillar topic for silo architecture |
| `intent` | enum | Intent category (see below) |
| `funnel_stage` | enum | `TOFU`, `MOFU`, or `BOFU` |
| `search_volume` | 0–1 | Relative search volume (percentile) |
| `difficulty` | 0–1 | Competition difficulty estimate |
| `estimated` | boolean | True if metrics are estimated |
| `validated` | boolean | True if confirmed via Autocomplete |
| `opportunity_score` | 0–1 | Combined opportunity metric |

### Intent Categories

**Traditional SEO:**
- `informational` - Educational queries
- `commercial` - Research/comparison queries
- `transactional` - Purchase intent
- `navigational` - Brand/site navigation

**GEO-Centric (for AI search):**
- `direct_answer` - Simple factual queries (AI snippets)
- `complex_research` - Multi-faceted exploration
- `comparative` - Comparison/recommendation
- `local` - Location-based queries

## 📖 CLI Commands

### `keyword-lab run`
Main pipeline for keyword discovery.

```bash
keyword-lab run \
  --seed-topic "coffee brewing" \
  --audience "home baristas" \
  --geo "us" \
  --language "en" \
  --sources ./docs \
  --config config.yaml \
  --output keywords.json \
  --verbose
```

### `keyword-lab brief`
Generate content briefs from keyword clusters.

```bash
keyword-lab brief keywords.json \
  --cluster "cluster-0" \
  --output brief.md \
  --format markdown
```

### `keyword-lab qa`
Validate and clean pipeline output.

```bash
keyword-lab qa keywords.json \
  --min-cluster-size 3 \
  --max-words 6 \
  --min-opportunity 0.1 \
  --report \
  --dry-run
```

## ⚙️ Configuration

Create a `config.yaml` to customize behavior:

```yaml
# Intent detection rules
intent_rules:
  informational: [how, what, why, guide, tutorial]
  commercial: [best, top, review, compare, vs]
  transactional: [buy, discount, deal, price]

# NLP settings
nlp:
  ngram_min_df: 2
  top_terms_per_doc: 10

# Clustering
cluster:
  use_silhouette: true
  max_clusters: 8

# LLM provider
llm:
  provider: auto  # auto, gemini, openai, anthropic, none

# Parent topic silos
parent_topics:
  enabled: true
  max_topics: 10

# Validation
validation:
  autocomplete_enabled: true
  max_autocomplete_checks: 100

# Filtering
filtering:
  blacklist: [login, support, my account]
  max_words: 6
  min_words: 2

# QA validation
qa:
  min_cluster_size: 3
  max_word_count: 6
  min_opportunity_score: 0.0
```

## 🔧 Advanced Features

### Google Autocomplete Validation
Keywords are validated against Google's autocomplete API to confirm real search demand:
```python
from keyword_lab.scrape import validate_keywords_with_autocomplete

validated = validate_keywords_with_autocomplete(
    keywords=["best coffee beans", "how to brew coffee"],
    language="en",
    country="us"
)
# Returns: {"best coffee beans": True, "how to brew coffee": True}
```

### Entity Co-Occurrence Analysis
Detect clusters at risk of thin content:
```python
from keyword_lab.cluster import get_cluster_entity_summary

analysis, healthy, at_risk = get_cluster_entity_summary(clusters)
for cluster in at_risk:
    print(f"⚠️ {cluster}: Thin content risk")
```

### Content Brief Generation
Generate LLM-powered content briefs:
```python
from keyword_lab.cli import generate_brief_with_llm

brief = generate_brief_with_llm(
    cluster_name="coffee-brewing",
    keywords=["how to brew coffee", "best coffee maker"],
    intents={"how to brew coffee": "informational"}
)
```

## 🏗️ Architecture

```
keyword_lab/
├── cli.py          # Typer CLI (run, brief, qa commands)
├── pipeline.py     # Main orchestration
├── scrape.py       # Document acquisition, Autocomplete, PAA
├── nlp.py          # Candidate generation, text processing
├── cluster.py      # Semantic clustering, entity analysis
├── metrics.py      # Percentile ranking, opportunity scoring
├── llm.py          # LLM integration, intent classification
├── qa.py           # Quality assurance validation
├── schema.py       # JSON Schema validation
├── io.py           # Output handling (JSON, CSV, XLSX)
└── config.py       # Configuration management
```

## 📊 Pipeline Flow

```
1. Acquire Documents
   └─> Local files, URLs, SERP results

2. Generate Candidates
   └─> TF-IDF extraction, question generation, LLM expansion

3. Filter & Validate
   └─> Blacklist, length limits, Autocomplete validation

4. Cluster & Classify
   └─> Semantic clustering, intent classification, parent topics

5. Score & Rank
   └─> Percentile ranking, opportunity scoring

6. Output & QA
   └─> JSON/CSV export, QA validation, content briefs
```

## ⚠️ Limitations & Ethics

- **No direct SERP scraping**: Uses APIs and respects robots.txt
- **Estimated metrics**: Marked with `estimated: true` unless from trusted sources
- **Rate limiting**: Built-in delays for Autocomplete validation
- **Language**: English-first for NLP, but runs for other languages

## 📚 Documentation

- [QuickStart Guide](./quickstart.md) - Detailed setup instructions
- [Configuration Reference](./config.sample.yaml) - All configuration options

## 📜 License

MIT License - See LICENSE file for details.

---

## ✨ New in v2.0

### Validation & Quality
- **Google Autocomplete Validation** - Confirm keywords have real search volume by checking against Google's autocomplete suggestions
- **Percentile Ranking** - Fairer scoring using percentile ranking that preserves long-tail keyword value
- **QA Validation Command** - `keyword-lab qa` to clean output (remove small clusters, long keywords)
- **Entity Co-Occurrence Analysis** - Detect thin content risk by analyzing entity density in clusters

### GEO (Generative Engine Optimization)
- **GEO Intent Categories** - 5 AI-search-optimized intents: `direct_answer`, `complex_research`, `transactional`, `local`, `comparative`
- **Parent Topic Silos** - Automatic hub-spoke architecture with LLM-assigned pillar topics
- **Content Brief Generation** - `keyword-lab brief` command for LLM-powered content briefs

### Data Quality
- **Blacklist Filtering** - Exclude competitor/navigational/spam patterns via config
- **PAA Extraction** - Extract "People Also Ask" questions from SerpAPI
- **Weighted HTML Extraction** - Priority extraction of H1-H3, meta descriptions, alt text

### New Output Fields
The output schema now includes:
- `parent_topic` - Pillar topic for silo architecture  
- `validated` - Boolean indicating Autocomplete confirmation

## Additional CLI Commands

### `keyword-lab brief`
Generate content briefs from keyword clusters:
```bash
keyword-lab brief keywords.json --cluster "cluster-0" --output brief.md
```

### `keyword-lab qa`
Validate and clean pipeline output:
```bash
keyword-lab qa keywords.json --min-cluster-size 3 --max-words 6 --report
```
