-- PostgreSQL schema for reggie

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    title TEXT,
    object_id VARCHAR(255) NOT NULL,
    docket_id VARCHAR(255),
    document_type VARCHAR(100),
    posted_date TIMESTAMP,
    metadata JSONB,
    aggregated_keywords JSONB DEFAULT '{"keywords_phrases": [], "entities": []}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_docket_id ON documents(docket_id);
CREATE INDEX IF NOT EXISTS idx_documents_object_id ON documents(object_id);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);

-- Comments table
CREATE TABLE IF NOT EXISTS comments (
    id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    comment_text TEXT,
    category VARCHAR(100),
    sentiment VARCHAR(50),
    topics JSONB,
    doctor_specialization VARCHAR(255),
    licensed_professional_type VARCHAR(255),
    keywords_entities JSONB DEFAULT '{"keywords_phrases": [], "entities": []}',
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    organization TEXT,
    posted_date TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comments_document_id ON comments(document_id);
CREATE INDEX IF NOT EXISTS idx_comments_category ON comments(category);
CREATE INDEX IF NOT EXISTS idx_comments_sentiment ON comments(sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_category_sentiment ON comments(category, sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_doctor_specialization ON comments(doctor_specialization);
CREATE INDEX IF NOT EXISTS idx_comments_licensed_professional_type ON comments(licensed_professional_type);
CREATE INDEX IF NOT EXISTS idx_comments_topics ON comments USING GIN(topics);

-- Comment chunks table for embeddings
CREATE TABLE IF NOT EXISTS comment_chunks (
    id BIGSERIAL PRIMARY KEY,
    comment_id VARCHAR(255) NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(1536),
    chunk_text_tsv tsvector,  -- Pre-computed full-text search vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comment_chunks_comment_id ON comment_chunks(comment_id);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_comment_chunks_fts ON comment_chunks USING GIN(chunk_text_tsv);

-- Trigger function to auto-update tsvector on insert/update
CREATE OR REPLACE FUNCTION update_chunk_text_tsv()
RETURNS TRIGGER AS $$
BEGIN
    NEW.chunk_text_tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to populate tsvector automatically
DROP TRIGGER IF EXISTS chunk_text_tsv_update ON comment_chunks;
CREATE TRIGGER chunk_text_tsv_update
    BEFORE INSERT OR UPDATE OF chunk_text ON comment_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_chunk_text_tsv();

-- IVFFlat index for vector similarity search (cosine distance)
-- Note: This index is created after data is loaded for better performance
-- CREATE INDEX IF NOT EXISTS idx_comment_chunks_embedding ON comment_chunks
-- USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers to update updated_at timestamp
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_comments_updated_at ON comments;
CREATE TRIGGER update_comments_updated_at
    BEFORE UPDATE ON comments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
