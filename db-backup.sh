#!/bin/bash
# Backup reggie PostgreSQL database

# Create backup directory if it doesn't exist
mkdir -p reggie/db/backups

# Generate timestamp for backup filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="reggie/db/backups/reggie_backup_${TIMESTAMP}.sql"

# Run pg_dump using Docker container to match server version
docker exec reggie-postgres pg_dump -U johntermaat -d reggie > "$BACKUP_FILE"

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "Database backup successful: $BACKUP_FILE"
else
    echo "Database backup failed!"
    exit 1
fi
