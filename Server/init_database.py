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
    
    # Test: Add messages
    print("Adding test messages...")
    add_chat_message(session_id, "user", "Nút bị lệch trên trang chủ")
    add_chat_message(session_id, "assistant", "1. UI — Nút bị lệch so với thiết kế, ảnh hưởng đến giao diện trang chủ.\nNút bị lệch trên trang chủ")
    print("Messages added\n")
    
    # Test: Get messages
    print("Retrieving messages...")
    messages = get_chat_messages(session_id)
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content']}")
    print()
    
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
