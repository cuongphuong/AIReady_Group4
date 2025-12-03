import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import os

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), "bugclassifier.db")

def init_db():
    """Initialize database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Chat sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Chat messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            file_upload_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (file_upload_id) REFERENCES file_uploads(id) ON DELETE SET NULL
        )
    """)
    
    # File uploads table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            total_rows INTEGER,
            classified_rows INTEGER,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE SET NULL
        )
    """)
    
    # Classification results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classification_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_upload_id INTEGER NOT NULL,
            bug_text TEXT NOT NULL,
            label TEXT NOT NULL,
            reason TEXT,
            team TEXT,
            severity TEXT,
            FOREIGN KEY (file_upload_id) REFERENCES file_uploads(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_uploads_session ON file_uploads(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_upload ON classification_results(file_upload_id)")
    
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


def create_chat_session(session_id: str, title: str = "Untitled") -> bool:
    """Create a new chat session"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, title) VALUES (?, ?)",
            (session_id, title)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Session already exists


def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get chat session by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_chat_session_title(session_id: str, title: str) -> bool:
    """Update chat session title"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
        (title, session_id)
    )
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success


def delete_chat_session(session_id: str) -> bool:
    """Delete chat session and all related data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return success


def get_all_chat_sessions() -> List[Dict[str, Any]]:
    """Get all chat sessions ordered by updated_at DESC"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_chat_message(session_id: str, role: str, content: str, file_upload_id: Optional[int] = None) -> int:
    """Add a message to chat session, returns message ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Update session updated_at
    cursor.execute(
        "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
        (session_id,)
    )
    
    # Insert message
    cursor.execute(
        "INSERT INTO chat_messages (session_id, role, content, file_upload_id) VALUES (?, ?, ?, ?)",
        (session_id, role, content, file_upload_id)
    )
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id


def get_chat_messages(session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all messages for a chat session"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_file_upload(
    session_id: Optional[str],
    filename: str,
    file_path: str,
    file_size: int,
    total_rows: int,
    classified_rows: int
) -> int:
    """Save file upload metadata, returns upload ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO file_uploads 
           (session_id, filename, file_path, file_size, total_rows, classified_rows)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, filename, file_path, file_size, total_rows, classified_rows)
    )
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return upload_id


def save_classification_results(file_upload_id: int, results: List[Dict[str, Any]]) -> bool:
    """Save classification results for a file upload"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for result in results:
            cursor.execute(
                """INSERT INTO classification_results 
                   (file_upload_id, bug_text, label, reason, team, severity)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    file_upload_id,
                    result.get('text', ''),
                    result.get('label', ''),
                    result.get('raw', ''),
                    result.get('team'),
                    result.get('severity')
                )
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving classification results: {e}")
        return False


def get_file_uploads(session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get file uploads, optionally filtered by session_id"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute(
            "SELECT * FROM file_uploads WHERE session_id = ? ORDER BY upload_time DESC LIMIT ?",
            (session_id, limit)
        )
    else:
        cursor.execute(
            "SELECT * FROM file_uploads ORDER BY upload_time DESC LIMIT ?",
            (limit,)
        )
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_classification_results(file_upload_id: int) -> List[Dict[str, Any]]:
    """Get classification results for a specific file upload"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM classification_results WHERE file_upload_id = ?",
        (file_upload_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_statistics() -> Dict[str, Any]:
    """Get overall statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total sessions
    cursor.execute("SELECT COUNT(*) FROM chat_sessions")
    total_sessions = cursor.fetchone()[0]
    
    # Total messages
    cursor.execute("SELECT COUNT(*) FROM chat_messages")
    total_messages = cursor.fetchone()[0]
    
    # Total uploads
    cursor.execute("SELECT COUNT(*) FROM file_uploads")
    total_uploads = cursor.fetchone()[0]
    
    # Total bugs classified
    cursor.execute("SELECT SUM(classified_rows) FROM file_uploads")
    total_bugs = cursor.fetchone()[0] or 0
    
    # Most common labels
    cursor.execute("""
        SELECT label, COUNT(*) as count 
        FROM classification_results 
        GROUP BY label 
        ORDER BY count DESC 
        LIMIT 10
    """)
    top_labels = [{"label": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_uploads": total_uploads,
        "total_bugs_classified": total_bugs,
        "top_labels": top_labels
    }


# Initialize database on module import
if __name__ == "__main__":
    init_db()
    print("Database tables created successfully!")
