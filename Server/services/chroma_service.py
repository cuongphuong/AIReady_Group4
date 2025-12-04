"""
ChromaDB Service
Vector database service cho semantic search và similarity matching
Sử dụng ChromaDB với embeddings từ OpenAI hoặc sentence-transformers
"""

import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    # ChromaDB 1.3.5+ uses chromadb.utils.embedding_functions
    try:
        from chromadb.utils import embedding_functions
    except ImportError:
        # Fallback for older versions
        import chromadb.utils.embedding_functions as embedding_functions
    CHROMADB_AVAILABLE = True
    logger.info("✅ ChromaDB available")
except ImportError as e:
    CHROMADB_AVAILABLE = False
    logger.warning(f"⚠️ ChromaDB not available: {e}")

# OpenAI for embeddings
try:
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False


class ChromaService:
    """
    Service quản lý ChromaDB vector store
    
    Features:
    - Store bug classifications với embeddings
    - Semantic search để tìm bugs tương tự
    - Dynamic few-shot examples retrieval
    - Cache classifications để giảm LLM calls
    """
    
    def __init__(self, use_openai_embeddings: bool = True):
        """
        Khởi tạo ChromaDB service
        
        Args:
            use_openai_embeddings: Dùng OpenAI embeddings (True) hoặc local sentence-transformers (False)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        self.use_openai = use_openai_embeddings and OPENAI_AVAILABLE
        
        # Khởi tạo ChromaDB client
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        
        logger.info(f"🔧 Initializing ChromaDB at: {db_path}")
        
        self.client = chromadb.PersistentClient(
            path=db_path
        )
        
        # Chọn embedding function
        # Đọc cấu hình OpenAI embeddings từ .env
        db_api_key = os.getenv("DB_OPENAI_API_KEY")
        db_api_base = os.getenv("DB_OPENAI_API_BASE_URL")
        db_model = os.getenv("DB_MODEL_NAME", "text-embedding-3-small")
        
        if use_openai_embeddings and db_api_key and db_api_base:
            # ChromaDB 1.3.5: Custom OpenAI embedding function
            try:
                from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
                try:
                    self.embedding_function = OpenAIEmbeddingFunction(
                        api_key=db_api_key,
                        api_base=db_api_base,
                        model_name=db_model
                    )
                except TypeError:
                    logger.warning("api_base not supported, using default endpoint")
                    self.embedding_function = OpenAIEmbeddingFunction(
                        api_key=db_api_key,
                        model_name=db_model
                    )
            except ImportError:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=db_api_key,
                    model_name=db_model
                )
            self.use_openai = True
        else:
            # ChromaDB 1.3.5: SentenceTransformer embedding function
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                self.embedding_function = SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
            except ImportError:
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
            self.use_openai = False
        
        # Tạo hoặc lấy collections
        self._init_collections()
        
        logger.info("✅ ChromaDB service initialized")
    
    def _init_collections(self):
        """Khởi tạo các collections cần thiết"""
        
        # Suffix dựa trên embedding type để có collections riêng biệt
        suffix = "_openai" if self.use_openai else "_local"
        
        # Collection 1: Bug Classifications (bugs đã được classify)
        self.bugs_collection = self.client.get_or_create_collection(
            name=f"bug_classifications{suffix}",
            embedding_function=self.embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "description": "Classified bug reports with labels",
                "embedding_type": "openai" if self.use_openai else "local"
            }
        )
        logger.info(f"📦 Collection 'bug_classifications{suffix}': {self.bugs_collection.count()} items")
        
        # Collection 2: Few-Shot Examples (training examples)
        self.examples_collection = self.client.get_or_create_collection(
            name=f"few_shot_examples{suffix}",
            embedding_function=self.embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "description": "Few-shot learning examples",
                "embedding_type": "openai" if self.use_openai else "local"
            }
        )
        logger.info(f"📦 Collection 'few_shot_examples{suffix}': {self.examples_collection.count()} items")
        
        # Collection 3: Label Descriptions (semantic label matching)
        self.labels_collection = self.client.get_or_create_collection(
            name=f"label_descriptions{suffix}",
            embedding_function=self.embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "description": "Bug label descriptions and keywords",
                "embedding_type": "openai" if self.use_openai else "local"
            }
        )
        logger.info(f"📦 Collection 'label_descriptions{suffix}': {self.labels_collection.count()} items")
    
    def add_bug(
        self,
        bug_id: str,
        bug_text: str,
        label: str,
        reason: str,
        team: Optional[str] = None,
        severity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Thêm bug đã classify vào vector store
        
        Args:
            bug_id: Unique ID (có thể là DB id hoặc hash)
            bug_text: Nội dung bug description
            label: Classification label
            reason: Lý do phân loại
            team: Team chịu trách nhiệm
            severity: Mức độ nghiêm trọng
            metadata: Metadata bổ sung
        
        Returns:
            True nếu thành công
        """
        try:
            # Chuẩn bị metadata
            bug_metadata = {
                "label": label,
                "reason": reason,
                "team": team or "",
                "severity": severity or "",
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                bug_metadata.update(metadata)
            
            # Add to collection
            self.bugs_collection.add(
                documents=[bug_text],
                metadatas=[bug_metadata],
                ids=[bug_id]
            )
            
            logger.info(f"✅ Added bug {bug_id} to vector store (label: {label})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add bug {bug_id}: {e}")
            return False
    
    def search_similar_bugs(
        self,
        query: str,
        top_k: int = 5,
        label_filter: Optional[str] = None,
        distance_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bugs tương tự semantic
        
        Args:
            query: Bug description cần tìm
            top_k: Số lượng kết quả trả về
            label_filter: Lọc theo label cụ thể
            distance_threshold: Ngưỡng distance (< threshold thì relevant)
        
        Returns:
            List of similar bugs với metadata và distance scores
        """
        try:
            # Build where clause nếu có filter
            where = {"label": label_filter} if label_filter else None
            
            # Query collection
            results = self.bugs_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            # Format results
            similar_bugs = []
            if results['ids'] and results['ids'][0]:
                for i, bug_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i] if results['distances'] else None
                    
                    # Skip nếu distance > threshold
                    if distance_threshold and distance and distance > distance_threshold:
                        continue
                    
                    similar_bugs.append({
                        'id': bug_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance,
                        'similarity': 1 - distance if distance else None
                    })
            
            logger.info(f"🔍 Found {len(similar_bugs)} similar bugs for query (top_k={top_k})")
            return similar_bugs
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_relevant_examples(
        self,
        bug_text: str,
        top_k: int = 3,
        label_hint: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy few-shot examples phù hợp nhất với bug cần classify
        
        Args:
            bug_text: Bug description
            top_k: Số lượng examples
            label_hint: Ưu tiên examples của label này (optional)
        
        Returns:
            List of relevant examples
        """
        try:
            where = {"label": label_hint} if label_hint else None
            
            results = self.examples_collection.query(
                query_texts=[bug_text],
                n_results=top_k,
                where=where
            )
            
            examples = []
            if results['ids'] and results['ids'][0]:
                for i, example_id in enumerate(results['ids'][0]):
                    examples.append({
                        'id': example_id,
                        'description': results['documents'][0][i],
                        'label': results['metadatas'][0][i].get('label'),
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            logger.info(f"📚 Retrieved {len(examples)} relevant examples")
            return examples
            
        except Exception as e:
            logger.error(f"❌ Failed to get examples: {e}")
            return []
    
    def add_few_shot_example(
        self,
        example_id: str,
        description: str,
        label: str
    ) -> bool:
        """
        Thêm few-shot example vào collection
        
        Args:
            example_id: Unique ID
            description: Bug description
            label: Classification label
        
        Returns:
            True nếu thành công
        """
        try:
            self.examples_collection.add(
                documents=[description],
                metadatas=[{"label": label}],
                ids=[example_id]
            )
            logger.info(f"✅ Added few-shot example {example_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add example: {e}")
            return False
    
    def add_label_description(
        self,
        label: str,
        description: str,
        keywords: List[str] = None,
        examples: List[str] = None
    ) -> bool:
        """
        Thêm label description vào collection để semantic matching
        
        Args:
            label: Label name
            description: Mô tả chi tiết label
            keywords: Keywords liên quan
            examples: Example use cases
        
        Returns:
            True nếu thành công
        """
        try:
            # Combine description + keywords + examples thành document
            doc_parts = [description]
            
            if keywords:
                doc_parts.append("Keywords: " + ", ".join(keywords))
            
            if examples:
                doc_parts.append("Examples: " + "; ".join(examples))
            
            document = "\n".join(doc_parts)
            
            self.labels_collection.upsert(
                documents=[document],
                metadatas=[{
                    "label": label,
                    "keywords": ",".join(keywords) if keywords else "",
                    "num_examples": len(examples) if examples else 0
                }],
                ids=[f"label_{label}"]
            )
            
            logger.info(f"✅ Added/updated label description: {label}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add label description: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê về vector store"""
        db_model = os.getenv("DB_MODEL_NAME", "text-embedding-3-small")
        return {
            "bugs_count": self.bugs_collection.count(),
            "examples_count": self.examples_collection.count(),
            "labels_count": self.labels_collection.count(),
            "embedding_model": f"openai/{db_model}" if self.use_openai else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
    
    def reset_collections(self, confirm: bool = False):
        """Xóa tất cả data trong collections (NGUY HIỂM!)"""
        if not confirm:
            raise ValueError("Must set confirm=True to reset collections")
        
        suffix = "_openai" if self.use_openai else "_local"
        logger.warning(f"⚠️ RESETTING ALL COLLECTIONS{suffix}!")
        
        try:
            self.client.delete_collection(f"bug_classifications{suffix}")
        except:
            pass
        try:
            self.client.delete_collection(f"few_shot_examples{suffix}")
        except:
            pass
        try:
            self.client.delete_collection(f"label_descriptions{suffix}")
        except:
            pass
        
        self._init_collections()
        logger.info("✅ Collections reset complete")


# Global singleton instances (separate for OpenAI and Local)
_chroma_service_openai = None
_chroma_service_local = None
_chroma_service_lock = False

def get_chroma_service(use_local_embeddings: bool = False) -> Optional[ChromaService]:
    """
    Get ChromaDB service instance (singleton pattern)
    
    Args:
        use_local_embeddings: True để dùng local sentence-transformers,
                             False để dùng OpenAI embeddings (default)
    
    Returns:
        ChromaService instance với embedding type tương ứng, hoặc None nếu không available
    """
    global _chroma_service_openai, _chroma_service_local, _chroma_service_lock
    
    if use_local_embeddings:
        if _chroma_service_local is None and not _chroma_service_lock:
            _chroma_service_lock = True
            try:
                logger.info("🔧 Initializing ChromaDB service singleton (LOCAL embeddings)...")
                _chroma_service_local = ChromaService(use_openai_embeddings=False)
                logger.info("✅ ChromaDB service singleton ready (LOCAL)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ChromaDB service: {e}")
                _chroma_service_local = None
                # Don't raise - return None để caller có thể handle gracefully
            finally:
                _chroma_service_lock = False
        return _chroma_service_local
    else:
        if _chroma_service_openai is None and not _chroma_service_lock:
            _chroma_service_lock = True
            try:
                logger.info("🔧 Initializing ChromaDB service singleton (OPENAI embeddings)...")
                _chroma_service_openai = ChromaService(use_openai_embeddings=True)
                logger.info("✅ ChromaDB service singleton ready (OPENAI)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ChromaDB service: {e}")
                _chroma_service_openai = None
                # Don't raise - return None để caller có thể handle gracefully
            finally:
                _chroma_service_lock = False
        return _chroma_service_openai


# Convenience functions
def is_chromadb_available() -> bool:
    """Check if ChromaDB is available and initialized"""
    try:
        service = get_chroma_service()
        return service is not None
    except:
        return False
