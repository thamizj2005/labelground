#!/usr/bin/env python3
"""
Database Migration Script

This script safely applies schema updates to the SQLite database. It checks for the 
existence of tables/columns and adds them incrementally if missing.

Usage:
    python scripts/migrate_db.py
"""

import sqlite3
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "workspace" / "meta.db"

def add_column(cursor, table, column, col_type):
    """Safely add a column to a table if it doesn't already exist"""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        print(f"✅ Added column '{column}' to table '{table}'")
    except sqlite3.OperationalError:
        print(f"ℹ️ Column '{column}' already exists in '{table}' (or table is missing)")

def main():
    if not DB_PATH.exists():
        print("ℹ️ No existing database found. A fresh database will be initialized automatically on start.")
        return

    print("=" * 60)
    print(f"🚀 Running Database Migrations on: {DB_PATH.name}")
    print("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Add User columns
    add_column(cursor, 'projects', 'owner_id', 'INTEGER REFERENCES users(id)')
    add_column(cursor, 'annotations', 'user_id', 'INTEGER REFERENCES users(id)')

    # Create Users table if missing
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR NOT NULL UNIQUE,
            email VARCHAR NOT NULL UNIQUE,
            hashed_password VARCHAR NOT NULL,
            role VARCHAR,
            is_active BOOLEAN,
            created_at DATETIME
        )
        """)
        print("✅ Users table checked/created")
    except Exception as e:
        print(f"❌ Error creating users table: {e}")

    # Create Project Assignments table if missing
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            project_id INTEGER NOT NULL REFERENCES projects(id),
            can_edit BOOLEAN,
            can_export BOOLEAN,
            UNIQUE(user_id, project_id)
        )
        """)
        print("✅ Project Assignments table checked/created")
    except Exception as e:
        print(f"❌ Error creating project_assignments table: {e}")

    # Create Activity Logs table if missing
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            project_id INTEGER REFERENCES projects(id),
            action VARCHAR NOT NULL,
            details JSON,
            created_at DATETIME
        )
        """)
        print("✅ Activity Logs table checked/created")
    except Exception as e:
        print(f"❌ Error creating activity_logs table: {e}")

    # Add verification_status to images
    add_column(cursor, 'images', 'verification_status', "VARCHAR DEFAULT 'unverified'")

    # Add level and traceback to activity_logs
    add_column(cursor, 'activity_logs', 'level', "VARCHAR DEFAULT 'info'")
    add_column(cursor, 'activity_logs', 'traceback', 'TEXT')

    # Add created_by_id and security questions to users
    add_column(cursor, 'users', 'created_by_id', 'INTEGER REFERENCES users(id)')
    add_column(cursor, 'users', 'security_question', 'VARCHAR')
    add_column(cursor, 'users', 'security_answer', 'VARCHAR')

    conn.commit()
    conn.close()
    print("=" * 60)
    print("🎉 Database migration completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
