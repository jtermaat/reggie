# Reggie: An Agentic RAG Tool for Analyzing Comments on Regulations.gov

Reggie is an end-to-end tool for loading, processing, and analyzing comments on regulation documents on [Regulations.gov](https://www.regulations.gov/), with a particular focus on Healthcare regulations.

Data is imported from bulk downloads (or loaded via API) and processed by tagging with a lightweight LLM and chunking/embedding for vector search.

This data is saved in a local PostgreSQL database with pgvector for semantic search, and exposed through visualizations and query tools an agent can use to make statistical queries or text-based RAG searches (with optional filtering on the tagged metadata).  

## RAG Graph

![Agent Graph](reggie-graph.png)

## Quick Start

### Prerequisites

- Python 3.9+
- Docker Desktop (for easy setup) OR PostgreSQL 16+ with pgvector
- OpenAI API key
- Regulations.gov API Key

### Setup: Option A - Docker Compose (Recommended)

This is the easiest way to get started. Docker Compose will automatically set up PostgreSQL with the pgvector extension.

#### 1. Install Reggie

```bash
# Clone or navigate to the reggie directory
cd /path/to/reggie

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install reggie and all dependencies
pip install -e .
```

#### 2. Start PostgreSQL Database

```bash
# Start PostgreSQL with pgvector using Docker Compose
docker compose up -d

# The database will be ready in a few seconds
# Connection details: postgresql://reggie:reggie@localhost:5432/reggie
```

#### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# Required:
# OPENAI_API_KEY=[your OpenAI API key]
# REG_API_KEY=[your Regulations.gov API key]
#
# DATABASE_URL is already configured for Docker Compose (postgresql://reggie:reggie@localhost:5432/reggie)
```

### Setup: Option B - Manual PostgreSQL Installation

If you prefer to install PostgreSQL manually or use an existing PostgreSQL server:

#### 1. Install PostgreSQL with pgvector

**macOS (Homebrew):**
```bash
brew install postgresql@16
brew install pgvector
brew services start postgresql@16
```

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql-16 postgresql-16-pgvector
sudo systemctl start postgresql
```

#### 2. Create Database and User

```bash
# Connect to PostgreSQL
psql postgres

# In PostgreSQL shell:
CREATE DATABASE reggie;
CREATE USER reggie WITH PASSWORD 'reggie';
GRANT ALL PRIVILEGES ON DATABASE reggie TO reggie;
\q
```

#### 3. Install Reggie & Configure

```bash
# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure environment
cp .env.example .env

# Edit .env and set:
# OPENAI_API_KEY=[your key]
# REG_API_KEY=[your key]
# DATABASE_URL=postgresql://reggie:reggie@localhost:5432/reggie
```

### 3. Import and Process Comments

The database schema will be automatically initialized on first use.

The **recommended approach** is to download comments in bulk from Regulations.gov and import them:

1. Go to the document page on [Regulations.gov](https://www.regulations.gov/) and use the "Download" feature to export comments as a CSV file
2. Import the CSV into Reggie:

```bash
reggie import ~/Downloads/comments-export.csv
```

3. Process the imported comments:

```bash
reggie process CMS-2025-0304-0001
```

**What each command does:**

- `reggie import`: Loads comments from the bulk download CSV into the database. This is fast since it bypasses API rate limits.
- `reggie process`: Tags each comment with AI (category, sentiment, topics) and creates embeddings for semantic search. Shows a cost report when complete.

The tagging categories and topics can be viewed in [comment.py](https://github.com/jtermaat/reggie/blob/main/reggie/models/comment.py).

### 4. Start a Discussion

  ```bash
  reggie discuss CMS-2025-0304-0001
  ```

  Opens a dialogue with an agent.  The agent can query the comment data with vector search and filter by the tags we added during processing.  The agent can also make statistical queries to answer questions about the general support level of various types of commenters, or what sorts of topics were raised by whom.

  - "What do physicians think about this rule?"
  - "What concerns were raised about reimbursement?"

  **Model Selection**: By default, the discussion agent uses `gpt-5-mini` for high-quality responses with streaming enabled. You can override this with the `--model` flag:

  ```bash
  reggie discuss CMS-2025-0304-0001 --model gpt-4o-mini
  ```

  **Streaming**: Reggie automatically attempts to stream responses for a better interactive experience. If streaming is unavailable (e.g., your organization needs verification), it will gracefully fall back to non-streaming mode.

  **Note**: When the agent makes statistical queries, visualizations are automatically displayed showing bar charts of the results with color-coded sentiment breakdowns.

### 5. Visualize Opposition/Support

  ```bash
  reggie visualize CMS-2025-0304-0001
  ```

  Displays a comprehensive opposition vs. support visualization showing sentiment breakdown across all commenter categories. The centered bar chart shows:
  - Categories sorted by total comment count
  - Opposition (red bars, left) and support (green bars, right) percentages
  - Total comments per category
  - Percentages based on total category comments (allowing users to deduce mixed/unclear sentiment)

### 6. Alternative Workflows

If you don't have a bulk download CSV, you can use the API-based approaches below.

**Stream (download + process in one step):**
```bash
reggie stream CMS-2025-0304-0001
```

Downloads comments from the Regulations.gov API and processes them immediately. Shows real-time cost tracking. Rate-limited to ~1000 requests/hour, so large documents may take hours. Good for quick tests with small documents like [CMS-2025-0304-0001](https://www.regulations.gov/document/CMS-2025-0304-0001) (10 comments).

**Load + Process (separate steps):**
```bash
reggie load CMS-2025-0304-0001   # Download only
reggie process CMS-2025-0304-0001  # Process separately
```

Use this when you want to download first and process later, or when experimenting with different processing parameters. The `process` command supports `--batch-size` (default: 10) and `--skip-processed` to resume interrupted runs.

### 7. Database Management

**List documents:**
```bash
reggie list
```

Shows all documents in the database with their document ID, title, docket ID, comment count, and load date.

**Clear a document:**
```bash
reggie clear CMS-2025-0304-0001
```

Removes a document and all associated data (comments, chunks, embeddings). Requires confirmation, or use `--force` to skip. Useful when re-importing a document with fresh data.


## Design Notes

### Tagging

During processing, each comment is classified by a lightweight LLM (default: gpt-5-nano) along three dimensions:

- **Category**: Who is commenting (e.g., "Physicians & Surgeons", "Patient Advocates", "Hospitals & Health Systems")
- **Sentiment**: Position on the regulation ("for", "against", "mixed", "unclear")
- **Topics**: What they discuss (e.g., "reimbursement_payment", "access_to_care", "administrative_burden")

Detailed enums can be viewed in [comment.py](https://github.com/jtermaat/reggie/blob/main/reggie/models/comment.py).

These tags are stored as structured metadata in PostgreSQL (as JSONB) alongside the vector embeddings (using pgvector).

### Tools: Statistical Queries and Filtered RAG

This tagging enables the agent to use two tools: A statistical query tool, and a text-based RAG search with optional filtering on tagged metadata.

The statistical queries allow it to answer questions like:

> "What do doctors generally think about this rule?"

Filtered RAG queries allow it to answer questions like:

> "What do physicians think about reimbursement?"

The agent can filter the vector search to only comments where:
- `category = "Physicians & Surgeons"`
- `topics = ["reimbursement_payment"]`

### Separated RAG Sub-Agent

Rather than having the discussion agent directly call vector search, we use a **separate LangGraph sub-agent** ([rag_graph.py](https://github.com/jtermaat/reggie/blob/main/reggie/agent/rag_graph.py)) for retrieval:


**Context Savings for the main agent**

The RAG sub-agent evaluates chunk snippets and decides which comments are relevant *before* returning results to the main agent. The discussion agent only sees the final, filtered set of complete comments—not all the intermediate search results and assessments, avoiding context pollution.


### Evaluation Framework

The evaluation dataset can be viewed [here along with evaluation runs](https://smith.langchain.com/public/97d8f43d-6ad7-4cc9-9b02-c9c37ef4768d/d).

Queries are ranked Easy, Medium, and Hard, with evaluators for completeness, accuracy, relevance, filter application, and tool choice.


## Model Selection & Configuration

Reggie uses different models optimized for different tasks:

- **Categorization/Tagging**: `gpt-5-nano` - Optimized for cost efficiency when processing thousands of comments
- **Discussion Agent**: `gpt-5-mini` (default) - Balanced performance and quality for interactive conversations
- **RAG Sub-Agent**: `gpt-5-mini` (default) - Ensures high-quality retrieval and evaluation

You can customize the discussion model using the `--model` flag to experiment with different OpenAI models based on your needs and budget.

## License

MIT License - See [LICENSE](LICENSE) for details. 
