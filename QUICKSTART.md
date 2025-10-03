# Reggie Quick Start Guide

Get up and running with Reggie in 5 minutes.

## Prerequisites

- Python 3.9+
- PostgreSQL with pgvector (or use Docker)
- OpenAI API key

## 1. Start PostgreSQL (Docker)

```bash
docker run --name reggie-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=reggie \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

## 2. Install Reggie

```bash
# Clone or navigate to the reggie directory
cd /path/to/reggie

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Minimum required:
# OPENAI_API_KEY=sk-...your-key-here...
```

## 4. Initialize Database

```bash
reggie init
```

You should see: "✓ Database initialized successfully!"

## 5. Load a Document

```bash
reggie load CMS-2025-0304-0009
```

This will:
- Fetch the document and all comments
- Categorize each comment
- Generate embeddings
- Store everything in PostgreSQL

**Note**: This may take several minutes depending on the number of comments.

## 6. View Loaded Documents

```bash
reggie list
```

You'll see a table with document statistics.

## Example Output

```
Loading document: CMS-2025-0304-0009

Fetching document metadata...
Fetching comments...
Fetched 245 comments
Categorizing comments...
Chunking and embedding comments...
Storing data in database...

✓ Document loaded successfully!

Statistics:
  • Comments processed: 245
  • Chunks created: 412
  • Errors: 0
  • Duration: 183.2s
```

## Common Issues

### "connection refused" Error

PostgreSQL isn't running. Start it with:
```bash
docker start reggie-postgres
```

### "OPENAI_API_KEY not set" Error

Edit your `.env` file and add:
```
OPENAI_API_KEY=sk-your-key-here
```

### Rate Limiting from Regulations.gov

Get a free API key from https://open.gsa.gov/api/regulationsgov/ and add to `.env`:
```
REG_API_KEY=your-key-here
```

## Next Steps

- Load more documents: `reggie load ANOTHER-DOC-ID`
- View all documents: `reggie list`
- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand how it works
- Read [SETUP.md](SETUP.md) for advanced configuration

## Getting Document IDs

Find document IDs on regulations.gov:

1. Visit https://www.regulations.gov/
2. Search for regulations
3. Click on a document
4. The document ID is in the URL:
   ```
   https://www.regulations.gov/document/CMS-2025-0304-0009
                                        ^^^^^^^^^^^^^^^^^^^
                                        This is the document ID
   ```

## Example Document IDs to Try

- `CMS-2025-0304-0009` - CMS healthcare regulation
- `FDA-2023-N-1234-0001` - FDA notice (check regulations.gov for current IDs)
- Any document ID from regulations.gov with public comments

## Help

```bash
reggie --help           # Show all commands
reggie load --help      # Show load command options
reggie init --help      # Show init command options
```
