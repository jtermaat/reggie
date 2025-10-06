-- Migration to add topics column to comments table
-- Run this on existing databases to add the topics feature

-- Add topics column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'comments' AND column_name = 'topics'
    ) THEN
        ALTER TABLE comments ADD COLUMN topics TEXT[];
        CREATE INDEX idx_comments_topics ON comments USING GIN (topics);
        RAISE NOTICE 'Added topics column and index to comments table';
    ELSE
        RAISE NOTICE 'Topics column already exists';
    END IF;
END $$;
