"""
Vector Store Model Layer
Quản lý ChromaDB operations và data migrations
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Import ChromaDB service
try:
    from services.chroma_service import get_chroma_service, is_chromadb_available
    CHROMA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ChromaDB service not available: {e}")
    CHROMA_AVAILABLE = False

# Import config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BUG_LABELS, FEW_SHOT_EXAMPLES


def generate_bug_id(bug_text: str, label: str) -> str:
    """
    Generate unique ID cho bug dựa trên content hash
    
    Args:
        bug_text: Bug description
        label: Classification label
    
    Returns:
        Unique bug ID (hash-based)
    """
    content = f"{bug_text}_{label}_{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def init_vector_store(use_local_embeddings: bool = False) -> bool:
    """
    Khởi tạo vector store với data ban đầu
    - Load few-shot examples
    - Load label descriptions
    
    Args:
        use_local_embeddings: True để dùng local embeddings, False cho OpenAI
    
    Returns:
        True nếu thành công
    """
    if not CHROMA_AVAILABLE:
        logger.warning("ChromaDB not available, skipping vector store init")
        return False
    
    try:
        logger.info("🔧 Initializing vector store with seed data...")
        chroma = get_chroma_service(use_local_embeddings=use_local_embeddings)
        if chroma is None:
            logger.warning("ChromaDB service not available")
            return False
        
        # 1. Load few-shot examples
        logger.info("📚 Loading few-shot examples...")
        for i, example in enumerate(FEW_SHOT_EXAMPLES):
            example_id = f"example_{i:03d}"
            chroma.add_few_shot_example(
                example_id=example_id,
                description=example['description'],
                label=example['label']
            )
        
        logger.info(f"✅ Loaded {len(FEW_SHOT_EXAMPLES)} few-shot examples")
        
        # 2. Load label descriptions
        logger.info("🏷️ Loading label descriptions...")
        for label, config in BUG_LABELS.items():
            chroma.add_label_description(
                label=label,
                description=config.get('desc', ''),
                keywords=config.get('keywords', []),
                examples=config.get('examples', [])
            )
        
        logger.info(f"✅ Loaded {len(BUG_LABELS)} label descriptions")
        
        # 3. Show stats
        stats = chroma.get_statistics()
        logger.info(f"📊 Vector store stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize vector store: {e}")
        return False


def migrate_bugs_to_vector_store(
    bugs: List[Dict[str, Any]],
    batch_size: int = 100,
    use_local_embeddings: bool = False
) -> Dict[str, int]:
    """
    Migrate bugs từ SQLite sang ChromaDB
    
    Args:
        bugs: List of bug dicts với keys: text, label, reason, team, severity
        batch_size: Batch size cho processing
        use_local_embeddings: True để dùng local embeddings
    
    Returns:
        Dict với success/failed counts
    """
    if not CHROMA_AVAILABLE:
        logger.warning("ChromaDB not available")
        return {"success": 0, "failed": 0}
    
    logger.info(f"🔄 Migrating {len(bugs)} bugs to vector store...")
    chroma = get_chroma_service(use_local_embeddings=use_local_embeddings)
    if chroma is None:
        logger.warning("ChromaDB service not available")
        return {"success": 0, "failed": 0, "total": len(bugs)}
    
    success_count = 0
    failed_count = 0
    
    for i, bug in enumerate(bugs):
        try:
            # Generate unique ID
            bug_id = generate_bug_id(
                bug.get('text', ''),
                bug.get('label', '')
            )
            
            # Add to vector store
            result = chroma.add_bug(
                bug_id=bug_id,
                bug_text=bug.get('text', ''),
                label=bug.get('label', ''),
                reason=bug.get('reason', ''),
                team=bug.get('team'),
                severity=bug.get('severity'),
                metadata={
                    'source': 'migration',
                    'original_id': bug.get('id')
                }
            )
            
            if result:
                success_count += 1
            else:
                failed_count += 1
            
            # Log progress
            if (i + 1) % batch_size == 0:
                logger.info(f"Progress: {i + 1}/{len(bugs)} bugs processed")
        
        except Exception as e:
            logger.error(f"Failed to migrate bug {i}: {e}")
            failed_count += 1
    
    logger.info(f"✅ Migration complete: {success_count} success, {failed_count} failed")
    
    return {
        "success": success_count,
        "failed": failed_count,
        "total": len(bugs)
    }


def add_classified_bug_to_vector_store(
    bug_text: str,
    label: str,
    reason: str,
    team: Optional[str] = None,
    severity: Optional[str] = None,
    file_upload_id: Optional[int] = None,
    use_local_embeddings: bool = False
) -> bool:
    """
    Thêm bug vừa classify vào vector store
    
    Args:
        bug_text: Bug description
        label: Classification label
        reason: Classification reason
        team: Assigned team
        severity: Bug severity
        file_upload_id: Reference to file upload
        use_local_embeddings: True để dùng local embeddings
    
    Returns:
        True nếu thành công
    """
    if not CHROMA_AVAILABLE:
        return False
    
    try:
        chroma = get_chroma_service(use_local_embeddings=use_local_embeddings)
        if chroma is None:
            logger.warning("ChromaDB service not available")
            return False
        
        # Generate unique ID
        bug_id = generate_bug_id(bug_text, label)
        
        # Add to vector store
        metadata = {}
        if file_upload_id:
            metadata['file_upload_id'] = file_upload_id
        
        return chroma.add_bug(
            bug_id=bug_id,
            bug_text=bug_text,
            label=label,
            reason=reason,
            team=team,
            severity=severity,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Failed to add bug to vector store: {e}")
        return False


def search_similar_classified_bugs(
    query: str,
    top_k: int = 5,
    label_filter: Optional[str] = None,
    similarity_threshold: float = 0.8,
    use_local_embeddings: bool = False
) -> List[Dict[str, Any]]:
    """
    Tìm bugs tương tự đã được classify
    
    Args:
        query: Bug description cần tìm
        top_k: Số lượng kết quả
        label_filter: Lọc theo label
        similarity_threshold: Ngưỡng similarity (0-1, càng cao càng giống)
        use_local_embeddings: True để dùng local embeddings
    
    Returns:
        List of similar bugs
    """
    if not CHROMA_AVAILABLE:
        return []
    
    try:
        chroma = get_chroma_service(use_local_embeddings=use_local_embeddings)
        if chroma is None:
            logger.warning("ChromaDB service not available")
            return []
        
        # Convert similarity threshold to distance threshold
        # Distance = 1 - similarity (cosine distance)
        distance_threshold = 1 - similarity_threshold
        
        return chroma.search_similar_bugs(
            query=query,
            top_k=top_k,
            label_filter=label_filter,
            distance_threshold=distance_threshold
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def get_dynamic_few_shot_examples(
    bug_text: str,
    top_k: int = 5,
    use_local_embeddings: bool = False
) -> List[Dict[str, Any]]:
    """
    Lấy few-shot examples động dựa trên bug cần classify
    
    Args:
        bug_text: Bug description
        top_k: Số lượng examples
        use_local_embeddings: True để dùng local embeddings
    
    Returns:
        List of relevant examples
    """
    if not CHROMA_AVAILABLE:
        return []
    
    try:
        chroma = get_chroma_service(use_local_embeddings=use_local_embeddings)
        if chroma is None:
            logger.warning("ChromaDB service not available")
            return []
        return chroma.get_relevant_examples(bug_text, top_k=top_k)
    
    except Exception as e:
        logger.error(f"Failed to get examples: {e}")
        return []


def get_vector_store_stats(use_local_embeddings: bool = False) -> Dict[str, Any]:
    """
    Lấy thống kê vector store
    
    Args:
        use_local_embeddings: True để xem stats của local embeddings
    """
    if not CHROMA_AVAILABLE:
        return {
            "available": False,
            "message": "ChromaDB not initialized"
        }
    
    try:
        chroma = get_chroma_service(use_local_embeddings=use_local_embeddings)
        if chroma is None:
            return {
                "available": False,
                "message": "ChromaDB service not initialized"
            }
        stats = chroma.get_statistics()
        stats['available'] = True
        return stats
    
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


# Initialize vector store on module import (lazy loading)
if __name__ == "__main__":
    # Run initialization script
    print("Initializing vector store...")
    success = init_vector_store()
    if success:
        print("✅ Vector store initialized successfully!")
        stats = get_vector_store_stats()
        print(f"📊 Stats: {stats}")
    else:
        print("❌ Failed to initialize vector store")
