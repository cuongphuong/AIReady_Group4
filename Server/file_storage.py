import os
from pathlib import Path

# Directory to store uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

def init_upload_directory():
    """Create uploads directory if it doesn't exist"""
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Upload directory ready at: {UPLOAD_DIR}")

def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save uploaded file to disk
    Returns: absolute file path
    """
    init_upload_directory()
    
    # Generate unique filename with timestamp to avoid collisions
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{filename}"
    
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    return file_path

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)

def cleanup_old_files(days: int = 30):
    """Delete files older than specified days"""
    import time
    
    if not os.path.exists(UPLOAD_DIR):
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            file_mtime = os.path.getmtime(file_path)
            if file_mtime < cutoff_time:
                os.remove(file_path)
                print(f"Deleted old file: {filename}")

if __name__ == "__main__":
    init_upload_directory()
