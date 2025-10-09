# Reggie: An Agentic RAG Tool for Analyzing Comments on Regulations.gov

Reggie is an end-to-end tool for loading, processing, and analyzing comments on regulation documents on [Regulations.gov](https://www.regulations.gov/), with a particular focus on Healthcare regulations.

Data is loaded from the public API and processed by tagging with a lightweight LLM and chunking/embedding for vector search.

This data is saved in postgresql and exposed through query tools an agent can use to make statistical queries or text-based RAG searches (with optional filtering on the tagged metadata).  

## RAG Graph

![Agent Graph](reggie-graph.png)


## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL with pgvector (or use Docker)
- OpenAI API key

### 1. Start PostgreSQL (Docker)

```bash
docker run --name reggie-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=reggie \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### 2. Install Reggie

```bash
# Clone or navigate to the reggie directory
cd /path/to/reggie

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install reggie and all dependencies
pip install -e .
```

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Minimum required:
# OPENAI_API_KEY=sk-...your-key-here...
```

### 4. Initialize Database

```bash
reggie init
```

You should see: "✓ Database initialized successfully!"

### 5. Load a Document

```bash
reggie load CMS-2025-0304-0001
```

This will fetch all the comments for that document.

**Note**: This may multiple hours depending on the number of comments, due to rate limiting. For a quick test, choose a document with a low number of comments. For example, [CMS-2025-0304-0001](https://www.regulations.gov/document/CMS-2025-0304-0001) has only 10 comments.

### 6. Process the Comments

  ```bash
  reggie process CCMS-2025-0304-0001
  ```

  Processes comment data for the given document, including chunking and embedding, and tagging the comments by who is commenting, the topics they commented on, and their sentiment. The enums in [comment.py](https://github.com/jtermaat/reggie/blob/main/reggie/models/comment.py) show the kinds of tags that can be used.  
  Note: Processing uses OpenAI API calls and may take time depending on the number of comments.

### 7. Start a Discussion
  
  ```bash
  reggie discuss CMS-2025-0304-0001
  ```

  Opens a dialogue with an agent.  The agent can query the comment data with vector search and filter by the tags we added during processing.  The agent can also make statistical queries to answer questions about the general support level of various types of commenters, or what sorts of topics were raised by whom.

  - "What do physicians think about this rule?"
  - "What concerns were raised about reimbursement?"


## Next Steps

Time was the most challenging constraint in this project, and there are plenty of things left to do.  Here are some critical areas:

- Right now, I'm using gpt-5-nano to tag the comments, but given this that this is a simple classification task, we could likely switch to a lightweight open source model.  One candidate to try is [Extract-0](https://github.com/herniqeu/extract0)

- We should do more experiments to tune the parameters in config.py. Chunk_size and chunk_offset could likely benefit from smaller values, but there hasn't been time to effectively experiment with this.

- Right now, we're stuck with a tradeoff where we can either use gpt-5-mini for the agent, or we can have streaming responses from OpenAI.  Experiments show that gpt-5-mini is superior.  We need to verify with OpenAI to enable streaming with gpt-5-mini.

- Since this tool could benefit from both visualizations and links to comment pages on regulations.gov, it would be beneficial to build a web front-end rather than having it live on the command-line. 
