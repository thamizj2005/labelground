import sqlite3
import os

db_path = "/home/thamizh/Documents/vision/workspace/meta.db"
if not os.path.exists(db_path):
    print("Database not found")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("--- Tables ---")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(table[0])
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    count = cursor.fetchone()[0]
    print(f"  Rows: {count}")

print("\n--- Projects ---")
cursor.execute("SELECT id, name, annotation_type FROM projects")
for p in cursor.fetchall():
    print(f"ID: {p[0]}, Name: {p[1]}, Type: {p[2]}")

print("\n--- Image Stats ---")
cursor.execute("SELECT COUNT(*) FROM images")
print(f"Total Images: {cursor.fetchone()[0]}")

print("\n--- Annotation Stats ---")
cursor.execute("SELECT created_by, COUNT(*) FROM annotations GROUP BY created_by")
for row in cursor.fetchall():
    print(f"Created By: {row[0]}, Count: {row[1]}")

conn.close()
