# Reggie 🏛️

An AI-powered agent for exploring and analyzing public comments on federal regulations from Regulations.gov.

## Overview

Reggie is a sophisticated command-line tool that helps healthcare companies and policy analysts extract insights from public comments on federal regulations. It leverages LangChain, OpenAI's GPT models, and vector embeddings to:

1. **Load & Process**: Fetch documents and comments from Regulations.gov
2. **Categorize**: Automatically classify commenters by type (e.g., physicians, hospitals, advocacy groups)
3. **Analyze Sentiment**: Determine if commenters support, oppose, or have mixed feelings about regulations
4. **Enable Search**: Use semantic search via embeddings to find relevant comments on specific topics
5. **Track Everything**: Store all data in PostgreSQL with pgvector for efficient similarity search

## Features

- 🚀 **Async-first architecture** for fast, parallel processing
- 🤖 **LangChain integration** with structured output for reliable categorization
- 📊 **15 commenter categories** tailored for healthcare regulations
- 💭 **Sentiment analysis** (for, against, mixed, unclear)
- 🔍 **Vector embeddings** for semantic search across comments
- 📦 **Chunking with overlap** to preserve context in long comments
- 🎯 **LangSmith integration** for tracing and evaluation
- ⚡ **Exponential backoff** for rate limiting and error handling
- 💾 **PostgreSQL + pgvector** for scalable storage

## Quick Start

See [SETUP.md](SETUP.md) for detailed installation instructions.

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Initialize the database
reggie init

# 4. Load a document
reggie load CMS-2025-0304-0009

# 5. List loaded documents
reggie list
```

## Architecture

### Component Overview

```
reggie/
├── api/           # Async client for Regulations.gov API
├── db/            # PostgreSQL schema and connection utilities
├── pipeline/      # LangChain processing pipelines
│   ├── categorizer.py  # Comment classification with structured output
│   ├── embedder.py     # Text chunking and embedding
│   └── loader.py       # Main orchestration logic
└── cli/           # Command-line interface
```

### Data Flow

1. **Fetch**: Async API client retrieves document + comments
2. **Categorize**: LangChain + GPT-5-nano classifies each comment
3. **Chunk**: Text splitter creates 1000-token chunks with 200-token overlap
4. **Embed**: OpenAI text-embedding-3-small generates vectors
5. **Store**: PostgreSQL stores comments, categories, sentiments, and embeddings

### Database Schema

- **documents**: Metadata for loaded documents
- **comments**: Full comment text with category and sentiment
- **comment_chunks**: Text chunks with vector embeddings (1536-dim)

## CLI Commands

### `reggie init`
Initialize the database schema.

```bash
reggie init
reggie init --force  # Re-initialize (drops existing data)
```

### `reggie load <document_id>`
Load a document and all its comments.

```bash
reggie load CMS-2025-0304-0009
reggie load CMS-2025-0304-0009 --batch-size 20
```

### `reggie list`
List all loaded documents with statistics.

```bash
reggie list
```

### `reggie discuss [document_id]`
Interactive discussion mode (coming soon).

```bash
reggie discuss                    # Discuss all loaded documents
reggie discuss CMS-2025-0304-0009 # Discuss specific document
```

## Configuration

All configuration is done via environment variables. See `.env.example` for the full list.

### Required
- `OPENAI_API_KEY`: Your OpenAI API key

### Optional
- `REG_API_KEY`: Regulations.gov API key (defaults to DEMO_KEY)
- `POSTGRES_*`: Database connection settings
- `LANGSMITH_API_KEY`: LangSmith API key for tracing
- `LANGSMITH_PROJECT`: LangSmith project name

## Commenter Categories

Reggie classifies commenters into 15 categories:

1. Physicians & Surgeons
2. Other Licensed Clinicians
3. Healthcare Practice Staff
4. Patients & Caregivers
5. Patient/Disability Advocates & Advocacy Organizations
6. Professional Associations
7. Hospitals Health Systems & Networks
8. Healthcare Companies & Corporations
9. Pharmaceutical & Biotech Companies
10. Medical Device & Digital Health Companies
11. Government & Public Programs
12. Academic & Research Institutions
13. Nonprofits & Foundations
14. Individuals / Private Citizens
15. Anonymous / Not Specified

## Technology Stack

- **LangChain**: Orchestration and structured output
- **OpenAI**: GPT-5-nano for categorization, text-embedding-3-small for vectors
- **PostgreSQL + pgvector**: Vector database for semantic search
- **asyncio + httpx**: Async I/O for parallel processing
- **Click + Rich**: Beautiful CLI with progress indicators
- **LangSmith**: Tracing and evaluation (optional)

## Development

### Running Tests
```bash
pytest
```

### Code Quality
```bash
black reggie/
ruff reggie/
mypy reggie/
```

### Project Structure
```
reggie/
├── reggie/              # Main package
│   ├── api/            # API client
│   ├── db/             # Database utilities
│   ├── pipeline/       # LangChain pipelines
│   ├── cli/            # CLI interface
│   └── config.py       # Configuration
├── scripts/            # Utility scripts
├── requirements.txt    # Dependencies
├── setup.py           # Package setup
└── SETUP.md           # Setup guide
```

## Roadmap

- [x] Document loading pipeline
- [x] Comment categorization
- [x] Sentiment analysis
- [x] Vector embeddings with chunking
- [x] CLI interface
- [x] LangSmith integration
- [ ] RAG agent for querying (`reggie discuss`)
- [ ] Advanced analytics and reporting
- [ ] Support for non-healthcare dockets
- [ ] Web interface

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
