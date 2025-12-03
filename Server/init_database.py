"""
Script to initialize database and test basic operations
"""
from models.database import init_db, create_chat_session, add_chat_message, get_chat_messages, get_statistics
from utils.file_storage import init_upload_directory
import uuid

def main():
    print("=== Initializing BugClassifier Database ===\n")
    
    # Initialize database
    init_db()
    print()
    
    # Initialize upload directory
    init_upload_directory()
    print()
    
    # Test: Create a sample chat session
    session_id = str(uuid.uuid4())
    print(f"Creating test chat session: {session_id}")
    success = create_chat_session(session_id, "Chat demo")
    print(f"Success: {success}\n")
    
    # Show statistics
    print("=== Database Statistics ===")
    stats = get_statistics()
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Total uploads: {stats['total_uploads']}")
    print(f"Total bugs classified: {stats['total_bugs_classified']}")
    print()
    
    print("✅ Database initialization and test completed successfully!")

if __name__ == "__main__":
    main()
