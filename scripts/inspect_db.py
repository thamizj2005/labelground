#!/usr/bin/env python3
"""
Inspect Database Script

This helper script inspects the SQLite database to print out statistics about the projects, 
images, users, and annotations currently registered in the system.

Usage:
    python scripts/inspect_db.py
"""

import sqlite3
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "workspace" / "meta.db"

def main():
    if not DB_PATH.exists():
        print(f"❌ Database not found at: {DB_PATH}")
        print("Please run start.sh or run the app first to initialize the database.")
        return

    print("=" * 60)
    print(f"🔍 Inspecting Database: {DB_PATH.name}")
    print(f"📂 Full Path: {DB_PATH}")
    print("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    print("\n📦 Tables in Database:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"  • {table[0]:<25} | Rows: {count}")

    print("\n📁 Projects:")
    try:
        cursor.execute("SELECT id, name, annotation_type FROM projects")
        projects = cursor.fetchall()
        if not projects:
            print("  (No projects created yet)")
        for p in projects:
            cursor.execute(f"SELECT COUNT(*) FROM images WHERE project_id = {p[0]}")
            img_count = cursor.fetchone()[0]
            print(f"  • ID: {p[0]:<2} | Name: {p[1]:<20} | Type: {p[2]:<10} | Images: {img_count}")
    except sqlite3.OperationalError:
        print("  ❌ Could not read projects table. May not be initialized.")

    print("\n📊 Annotation Stats:")
    try:
        cursor.execute("SELECT created_by, COUNT(*) FROM annotations GROUP BY created_by")
        rows = cursor.fetchall()
        if not rows:
            print("  (No annotations saved yet)")
        for row in rows:
            print(f"  • Created By: {row[0]:<10} | Count: {row[1]}")
    except sqlite3.OperationalError:
        print("  ❌ Could not read annotations table.")

    conn.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
