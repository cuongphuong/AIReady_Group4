import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bugrecordindex")
PINECONE_HOST = os.getenv("PINECONE_HOST")

# Initialize Pinecone (v3+ API - Serverless)
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is not set in .env file.")

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

class PineconeService:
    def __init__(self, index_name: str):
        self.index_name = index_name
        
        # List existing indexes
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Index '{self.index_name}' không tồn tại. Đang tạo mới...")
            # Create serverless index with dimension 768 and cosine metric
            # Note: dimension must match the embedding model output dimension
            pc.create_index(
                name=self.index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        # Connect to index using host if provided
        if PINECONE_HOST:
            self.index = pc.Index(self.index_name, host=PINECONE_HOST)
        else:
            self.index = pc.Index(self.index_name)
            
        logger.info(f"Pinecone Service initialized with index: {self.index_name}")

    def upsert_vectors(self, vectors: list):
        """Thêm hoặc cập nhật vectors vào index
        
        Args:
            vectors: List of tuples (id, embedding, metadata) or list of dicts
        """
        logger.info(f"Upserting {len(vectors)} vectors vào index '{self.index_name}'...")
        
        # Convert tuple format to dict format for Pinecone v8+
        formatted_vectors = []
        for v in vectors:
            if isinstance(v, tuple) and len(v) >= 2:
                # Format: (id, values, metadata)
                vector_dict = {
                    'id': str(v[0]),
                    'values': v[1]
                }
                if len(v) >= 3 and v[2]:  # Has metadata
                    vector_dict['metadata'] = v[2]
                formatted_vectors.append(vector_dict)
            elif isinstance(v, dict):
                formatted_vectors.append(v)
            else:
                logger.warning(f"Skipping invalid vector format: {type(v)}")
        
        if formatted_vectors:
            self.index.upsert(vectors=formatted_vectors)

    def query_vectors(self, vector: list, top_k: int = 10):
        """Tìm kiếm vectors gần nhất"""
        logger.info(f"Querying top {top_k} vectors từ index '{self.index_name}'...")
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )

    def delete_vectors(self, ids: list):
        logger.info(f"Deleting {len(ids)} vectors từ index '{self.index_name}'...")
        self.index.delete(ids)

# Global instance (lazy loading)
_pinecone_service = None

def get_pinecone_service(index_name: str | None = None) -> PineconeService:
    """Get singleton Pinecone service instance.

    If `index_name` is None, the function will use `PINECONE_INDEX_NAME` from environment
    (default `default`).
    """
    global _pinecone_service
    if index_name is None:
        index_name = PINECONE_INDEX_NAME
    if _pinecone_service is None:
        _pinecone_service = PineconeService(index_name)
    return _pinecone_service