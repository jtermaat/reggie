-- SQLite schema for reggie

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    object_id TEXT NOT NULL,
    docket_id TEXT,
    document_type TEXT,
    posted_date TEXT,
    metadata TEXT,  -- JSON stored as TEXT
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
    topics TEXT,  -- JSON array stored as TEXT
    doctor_specialization TEXT,
    licensed_professional_type TEXT,
    first_name TEXT,
    last_name TEXT,
    organization TEXT,
    posted_date TEXT,
    metadata TEXT,  -- JSON stored as TEXT
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comments_document_id ON comments(document_id);
CREATE INDEX IF NOT EXISTS idx_comments_category ON comments(category);
CREATE INDEX IF NOT EXISTS idx_comments_sentiment ON comments(sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_category_sentiment ON comments(category, sentiment);
CREATE INDEX IF NOT EXISTS idx_comments_doctor_specialization ON comments(doctor_specialization);
CREATE INDEX IF NOT EXISTS idx_comments_licensed_professional_type ON comments(licensed_professional_type);

-- Comment chunks table for embeddings
CREATE TABLE IF NOT EXISTS comment_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id TEXT NOT NULL REFERENCES comments(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding BLOB,  -- 1536-dimensional float32 vector stored as binary
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comment_chunks_comment_id ON comment_chunks(comment_id);

-- Triggers to update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_documents_updated_at
    AFTER UPDATE ON documents
    FOR EACH ROW
BEGIN
    UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_comments_updated_at
    AFTER UPDATE ON comments
    FOR EACH ROW
BEGIN
    UPDATE comments SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
