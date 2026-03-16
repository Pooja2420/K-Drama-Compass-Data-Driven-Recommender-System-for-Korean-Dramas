#!/bin/bash
# Creates multiple PostgreSQL databases from the POSTGRES_MULTIPLE_DATABASES env var.
# Usage: POSTGRES_MULTIPLE_DATABASES=db1,db2,db3

set -e

function create_db() {
    local database=$1
    echo "Creating database: $database"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $database;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL
}

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Multiple database creation requested: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
        create_db "$db"
    done
    echo "Multiple databases created."
fi
