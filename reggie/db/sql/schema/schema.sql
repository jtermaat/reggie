-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    object_id TEXT NOT NULL,
    docket_id TEXT,
    document_type TEXT,
    posted_date TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_docket_id ON documents(docket_id);
CREATE INDEX IF NOT EXISTS idx_documents_object_id ON documents(object_id);

-- Comments table
CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    comment_text TEXT,
    category TEXT,
    sentiment TEXT,
    topics TEXT[],
    first_name TEXT,
    last_name TEXT,
    organization TEXT,
    posted_date TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comments_document_id ON comments(document_id);
CREATE INDEX IF NOT EXISTS idx_comments_category ON comments(category);
CREATE INDEX IF NOT EXISTS idx_comments_sentiment ON comments(sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_category_sentiment ON comments(category, sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_topics ON comments USING GIN (topics);

-- Comment chunks table for embeddings
CREATE TABLE IF NOT EXISTS comment_chunks (
    id SERIAL PRIMARY KEY,
    comment_id TEXT NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(1536),  -- dimension for text-embedding-3-small
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comment_chunks_comment_id ON comment_chunks(comment_id);
CREATE INDEX IF NOT EXISTS idx_comment_chunks_embedding ON comment_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_comments_updated_at BEFORE UPDATE ON comments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
