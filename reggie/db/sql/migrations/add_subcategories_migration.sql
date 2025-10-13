-- Migration to add sub-category columns to comments table
-- Run this on existing databases to add the doctor specialization and licensed professional type features

-- Add doctor_specialization column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'comments' AND column_name = 'doctor_specialization'
    ) THEN
        ALTER TABLE comments ADD COLUMN doctor_specialization TEXT;
        CREATE INDEX idx_comments_doctor_specialization ON comments(doctor_specialization);
        RAISE NOTICE 'Added doctor_specialization column and index to comments table';
    ELSE
        RAISE NOTICE 'doctor_specialization column already exists';
    END IF;
END $$;

-- Add licensed_professional_type column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'comments' AND column_name = 'licensed_professional_type'
    ) THEN
        ALTER TABLE comments ADD COLUMN licensed_professional_type TEXT;
        CREATE INDEX idx_comments_licensed_professional_type ON comments(licensed_professional_type);
        RAISE NOTICE 'Added licensed_professional_type column and index to comments table';
    ELSE
        RAISE NOTICE 'licensed_professional_type column already exists';
    END IF;
END $$;
