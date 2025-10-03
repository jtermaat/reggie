# Reggie Setup Guide

This guide will help you set up Reggie, an AI agent for exploring regulations.gov comments.

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 14 or higher with pgvector extension
- OpenAI API key

## Step 1: Set Up PostgreSQL with pgvector

### Option A: Using Docker (Recommended)

```bash
docker run --name reggie-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=reggie \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### Option B: Install Locally

1. Install PostgreSQL 14+
2. Install the pgvector extension:
   ```bash
   # On macOS with Homebrew
   brew install pgvector

   # On Ubuntu/Debian
   sudo apt-get install postgresql-16-pgvector
   ```

3. Create the database:
   ```bash
   psql -U postgres -c "CREATE DATABASE reggie;"
   ```

## Step 2: Clone and Set Up the Project

```bash
# Navigate to the project directory
cd /path/to/reggie

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Step 3: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   # Required
   OPENAI_API_KEY=sk-...your-key...

   # Optional but recommended
   REG_API_KEY=...your-regulations-gov-key...

   # Database (adjust if needed)
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=reggie
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password

   # Optional: LangSmith for tracing
   LANGSMITH_API_KEY=...your-langsmith-key...
   LANGSMITH_PROJECT=reggie
   ```

### Getting API Keys

**OpenAI API Key (Required)**
- Sign up at https://platform.openai.com/
- Navigate to API Keys section
- Create a new API key

**Regulations.gov API Key (Optional but Recommended)**
- Visit https://open.gsa.gov/api/regulationsgov/
- Sign up for a free API key
- Without this, the DEMO_KEY will be used (has strict rate limits)

**LangSmith API Key (Optional)**
- Sign up at https://smith.langchain.com/
- Create a new API key
- This enables tracing and evaluation of LangChain operations

## Step 4: Initialize the Database

```bash
reggie init
```

This will create all necessary tables and set up the pgvector extension.

## Step 5: Verify Installation

```bash
# Check the CLI is working
reggie --help

# List documents (should be empty initially)
reggie list
```

## Step 6: Load Your First Document

```bash
# Load a document by its ID
# Example: CMS (Centers for Medicare & Medicaid Services) document
reggie load CMS-2025-0304-0009
```

This will:
1. Fetch the document metadata
2. Fetch all comments on the document
3. Categorize each comment (sentiment + commenter type)
4. Chunk and embed all comments
5. Store everything in PostgreSQL

The process may take several minutes depending on the number of comments.

## Usage Examples

### Load a Document
```bash
reggie load CMS-2025-0304-0009
```

### List All Loaded Documents
```bash
reggie list
```

### Start Discussion (Coming Soon)
```bash
reggie discuss
reggie discuss CMS-2025-0304-0009
```

## Troubleshooting

### Database Connection Issues

If you see `connection refused` errors:
1. Ensure PostgreSQL is running
2. Check your `.env` file has correct database credentials
3. Verify the database exists: `psql -U postgres -l | grep reggie`

### Rate Limiting Errors

If you see rate limit errors from regulations.gov:
1. Get a free API key from https://open.gsa.gov/api/regulationsgov/
2. Add it to your `.env` file as `REG_API_KEY`

### OpenAI API Errors

If you see OpenAI authentication errors:
1. Verify your API key is correct in `.env`
2. Check your OpenAI account has available credits
3. Ensure you have access to the gpt-5-nano model

### pgvector Extension Issues

If you see `extension "vector" does not exist`:
1. Ensure pgvector is properly installed
2. Run: `psql -U postgres -d reggie -c "CREATE EXTENSION vector;"`

## Architecture

Reggie consists of several components:

- **API Client** (`reggie/api/`): Async client for regulations.gov API
- **Database** (`reggie/db/`): PostgreSQL schema and connection utilities
- **Pipeline** (`reggie/pipeline/`):
  - Categorizer: LangChain-based comment classification
  - Embedder: Text chunking and embedding with OpenAI
  - Loader: Main orchestration logic
- **CLI** (`reggie/cli/`): Command-line interface

## Next Steps

- Load more documents to build your database
- Explore the loaded comments using `reggie list`
- Wait for the `reggie discuss` feature to query and analyze comments

## Getting Help

For issues or questions:
1. Check the troubleshooting section above
2. Review the error logs (displayed in the terminal)
3. Ensure all environment variables are set correctly
