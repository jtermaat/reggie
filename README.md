# Reggie: An Agentic RAG Tool for Analyzing Comments on Regulations.gov

Reggie is an end-to-end tool for loading, processing, and analyzing comments on regulation documents on [Regulations.gov](https://www.regulations.gov/), with a particular focus on Healthcare regulations.

Data is loaded from the public API and processed by tagging with a lightweight LLM and chunking/embedding for vector search.

This data is saved in a local SQLite database and exposed through visualizations and query tools an agent can use to make statistical queries or text-based RAG searches (with optional filtering on the tagged metadata).  

## RAG Graph

![Agent Graph](reggie-graph.png)

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Regulations.gov API Key

### 1. Install Reggie

```bash
# Clone or navigate to the reggie directory
cd /path/to/reggie

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install reggie and all dependencies
pip install -e .
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Minimum required:
# OPENAI_API_KEY=[your OpenAI API key here]
# REG_API_KEY=[your regulations.gov API Key]
```

### 3. Process a Document

The database will be automatically initialized on first use at `~/.reggie/reggie.db`.

```bash
reggie stream CMS-2025-0304-0001
```

This is the **recommended approach** for ingesting and processing a document's comments. The `stream` command downloads all comments from Regulations.gov and processes them (including chunking, embedding, and AI-powered tagging) in a single efficient operation.

**What it does:**
- Downloads comments from the Regulations.gov API
- Tags each comment by category (who is commenting), sentiment (for/against), and topics discussed
- Creates embeddings for semantic search
- Shows real-time cost tracking as processing occurs

**Note**: This may take multiple hours depending on the number of comments, due to API rate limiting. For a quick test, choose a document with a low number of comments. For example, [CMS-2025-0304-0001](https://www.regulations.gov/document/CMS-2025-0304-0001) has only 10 comments.

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

### 6. Alternative: Advanced Workflow

For more control over the download and processing phases, you can use separate commands:

**Download comments only:**
```bash
reggie load CMS-2025-0304-0001
```

This fetches all comments for the document and stores them in the database without processing.

**Process previously downloaded comments:**
```bash
reggie process CMS-2025-0304-0001
```

This processes the comments that were already downloaded with `reggie load`, performing the AI tagging and embedding. You can customize the batch size with `--batch-size` (default: 10) and skip already-processed comments with `--skip-processed`.

**When to use this workflow:**
- When you want to download first and process later
- When experimenting with different processing parameters
- When re-processing comments with different AI models


## Design Notes

### Tagging

During processing, each comment is classified by a lightweight LLM (default: gpt-5-nano) along three dimensions:

- **Category**: Who is commenting (e.g., "Physicians & Surgeons", "Patient Advocates", "Hospitals & Health Systems")
- **Sentiment**: Position on the regulation ("for", "against", "mixed", "unclear")
- **Topics**: What they discuss (e.g., "reimbursement_payment", "access_to_care", "administrative_burden")

Detailed enums can be viewed in [comment.py](https://github.com/jtermaat/reggie/blob/main/reggie/models/comment.py).

These tags are stored as structured metadata in SQLite alongside the embeddings.

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
