"""
Script test để verify database integration với UI
"""
import requests
import json
from uuid import uuid4

BASE_URL = "http://localhost:8000"

def test_full_flow():
    print("=== Testing Full Database Integration Flow ===\n")
    
    # 1. Tạo chat session mới
    session_id = str(uuid4())
    print(f"1. Creating new chat session: {session_id}")
    
    response = requests.post(
        f"{BASE_URL}/chat/sessions",
        json={
            "session_id": session_id,
            "title": "Test Chat from Script"
        }
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # 2. Thêm user message
    print("2. Adding user message")
    response = requests.post(
        f"{BASE_URL}/chat/sessions/{session_id}/messages",
        json={
            "role": "user",
            "content": "Nút login bị lỗi không nhấn được"
        }
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # 3. Thêm assistant message
    print("3. Adding assistant message")
    response = requests.post(
        f"{BASE_URL}/chat/sessions/{session_id}/messages",
        json={
            "role": "assistant",
            "content": "Phân loại: UI\nTeam: Frontend\nSeverity: High"
        }
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # 4. Lấy tất cả messages
    print("4. Getting all messages")
    response = requests.get(
        f"{BASE_URL}/chat/sessions/{session_id}/messages?limit=10"
    )
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Total messages: {len(data['messages'])}")
    for msg in data['messages']:
        print(f"   - [{msg['role']}]: {msg['content'][:50]}...")
    print()
    
    # 5. Lấy tất cả sessions
    print("5. Getting all chat sessions")
    response = requests.get(f"{BASE_URL}/chat/sessions")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Total sessions: {len(data['sessions'])}")
    for session in data['sessions'][:3]:  # Show first 3
        print(f"   - {session['title']} (ID: {session['session_id'][:8]}...)")
    print()
    
    # 6. Update session title
    print("6. Updating session title")
    response = requests.put(
        f"{BASE_URL}/chat/sessions/{session_id}",
        json={"title": "Updated Test Chat"}
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # 7. Get statistics
    print("7. Getting statistics")
    response = requests.get(f"{BASE_URL}/statistics")
    print(f"   Status: {response.status_code}")
    stats = response.json()
    print(f"   Total sessions: {stats['total_sessions']}")
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Total uploads: {stats['total_uploads']}")
    print(f"   Total bugs classified: {stats['total_bugs_classified']}\n")
    
    # 8. Delete session (cleanup)
    print("8. Deleting test session (cleanup)")
    response = requests.delete(f"{BASE_URL}/chat/sessions/{session_id}")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not responding. Please start the server first.")
            print("   Run: uvicorn api:app --reload --port 8000")
            exit(1)
        
        test_full_flow()
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server at http://localhost:8000")
        print("   Please start the server first:")
        print("   cd Server && uvicorn api:app --reload --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
