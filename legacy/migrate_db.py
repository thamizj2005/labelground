import sqlite3
import os

db_path = 'workspace/meta.db'

if not os.path.exists(db_path):
    print("No existing database found. Fresh start will be fine.")
    exit(0)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def add_column(table, column, type):
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type}")
        print(f"Added column {column} to table {table}")
    except sqlite3.OperationalError:
        print(f"Column {column} already exists in {table} or table missing")

# Add User columns
add_column('projects', 'owner_id', 'INTEGER REFERENCES users(id)')
add_column('annotations', 'user_id', 'INTEGER REFERENCES users(id)')

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
    print("Users table checked/created")
except Exception as e:
    print(f"Error creating users table: {e}")

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
    print("ProjectAssignments table checked/created")
except Exception as e:
    print(f"Error creating project_assignments table: {e}")
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
    print("ActivityLogs table checked/created")
except Exception as e:
    print(f"Error creating activity_logs table: {e}")

# Add verification_status to images
add_column('images', 'verification_status', "VARCHAR DEFAULT 'unverified'")

# Add level and traceback to activity_logs
add_column('activity_logs', 'level', "VARCHAR DEFAULT 'info'")
add_column('activity_logs', 'traceback', 'TEXT')

# Add created_by_id to users
add_column('users', 'created_by_id', 'INTEGER REFERENCES users(id)')
add_column('users', 'security_question', 'VARCHAR')
add_column('users', 'security_answer', 'VARCHAR')

conn.commit()
conn.close()
print("Migration completed successfully.")
