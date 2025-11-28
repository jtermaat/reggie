-- Migration: Add full-text search capabilities to comment_chunks
-- This enables BM25-like lexical search alongside vector similarity search

-- 1. Add tsvector column for pre-computed text search vectors
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'comment_chunks' AND column_name = 'chunk_text_tsv'
    ) THEN
        ALTER TABLE comment_chunks ADD COLUMN chunk_text_tsv tsvector;
        RAISE NOTICE 'Added chunk_text_tsv column';
    END IF;
END $$;

-- 2. Populate tsvector for existing data
UPDATE comment_chunks
SET chunk_text_tsv = to_tsvector('english', chunk_text)
WHERE chunk_text_tsv IS NULL;

-- 3. Create GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_comment_chunks_fts
ON comment_chunks USING GIN(chunk_text_tsv);

-- 4. Create trigger function to auto-update tsvector on insert/update
CREATE OR REPLACE FUNCTION update_chunk_text_tsv()
RETURNS TRIGGER AS $$
BEGIN
    NEW.chunk_text_tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. Create trigger (drop first to avoid conflicts)
DROP TRIGGER IF EXISTS chunk_text_tsv_update ON comment_chunks;
CREATE TRIGGER chunk_text_tsv_update
    BEFORE INSERT OR UPDATE OF chunk_text ON comment_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_chunk_text_tsv();
