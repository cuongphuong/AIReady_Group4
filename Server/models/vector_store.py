import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Try to use Pinecone service
try:
    from services.pinecone_service import get_pinecone_service
    PINECONE_AVAILABLE = True
    logger.info("✅ Pinecone service available")
except Exception as e:
    PINECONE_AVAILABLE = False
    logger.warning(f"⚠️ Pinecone service not available: {e}")

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BUG_LABELS, FEW_SHOT_EXAMPLES


def generate_bug_id(bug_text: str, label: str) -> str:
    content = f"{bug_text}_{label}_{datetime.now().isoformat()}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _hash_to_vector(text: str, dim: int = 768) -> List[float]:
    """Deterministic fallback embedding based on MD5 digest."""
    m = hashlib.md5(text.encode()).digest()
    # Expand to dim bytes by repeating digest
    reps = (dim + len(m) - 1) // len(m)
    b = (m * reps)[:dim]
    # Convert bytes to floats in range [-1,1]
    vec = [((x / 255.0) * 2.0 - 1.0) for x in b]
    return vec


def _get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    return [_hash_to_vector(t) for t in texts]


def init_vector_store(use_local_embeddings: bool = False) -> bool:
    if not PINECONE_AVAILABLE:
        logger.warning("Pinecone not available, skipping vector store init")
        return False

    try:
        pine = get_pinecone_service()
        # Upsert few-shot examples
        texts = [ex['description'] for ex in FEW_SHOT_EXAMPLES]
        ids = [f"example_{i:03d}" for i in range(len(texts))]
        metadatas = [{'type': 'example', 'label': ex['label']} for ex in FEW_SHOT_EXAMPLES]
        embeddings = _get_embeddings(texts)
        pine.upsert_vectors([(ids[i], embeddings[i], metadatas[i]) for i in range(len(ids))])

        # Upsert label descriptions as docs too
        label_texts = []
        label_ids = []
        label_metas = []
        for label, cfg in BUG_LABELS.items():
            doc = cfg.get('desc', '') + '\nKeywords: ' + ','.join(cfg.get('keywords', []))
            label_texts.append(doc)
            label_ids.append(f"label_{label}")
            label_metas.append({'type': 'label', 'label': label})

        if label_texts:
            label_emb = _get_embeddings(label_texts)
            pine.upsert_vectors([(label_ids[i], label_emb[i], label_metas[i]) for i in range(len(label_ids))])

        logger.info("✅ Vector store (Pinecone) initialized with seed data")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Pinecone vector store: {e}")
        return False


def add_classified_bug_to_vector_store(
    bug_text: str,
    label: str,
    reason: str,
    team: Optional[str] = None,
    severity: Optional[str] = None,
    file_upload_id: Optional[int] = None,
    use_local_embeddings: bool = False
) -> bool:
    if not PINECONE_AVAILABLE:
        return False
    try:
        pine = get_pinecone_service()
        bug_id = generate_bug_id(bug_text, label)
        meta = {
            'type': 'bug',
            'label': label,
            'reason': reason,
            'team': team or '',
            'severity': severity or '',
            'timestamp': datetime.now().isoformat(),
        }
        if file_upload_id:
            meta['file_upload_id'] = file_upload_id

        emb = _get_embeddings([bug_text])[0]
        pine.upsert_vectors([(bug_id, emb, meta)])
        logger.info(f"✅ Added bug to Pinecone index (id={bug_id}, label={label})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to add bug to Pinecone: {e}")
        return False


def search_similar_classified_bugs(
    query: str,
    top_k: int = 5,
    label_filter: Optional[str] = None,
    similarity_threshold: float = 0.8,
    use_local_embeddings: bool = False
) -> List[Dict[str, Any]]:
    if not PINECONE_AVAILABLE:
        return []
    try:
        pine = get_pinecone_service()
        q_emb = _get_embeddings([query])[0]
        resp = pine.query_vectors(q_emb, top_k=top_k)

        # Normalize response handling for different Pinecone client versions
        matches = []
        if hasattr(resp, 'matches'):
            raw_matches = resp.matches
        elif isinstance(resp, dict) and 'matches' in resp:
            raw_matches = resp['matches']
        else:
            raw_matches = resp.get('matches', []) if isinstance(resp, dict) else []

        for m in raw_matches:
            # Extract metadata safely
            if hasattr(m, 'metadata'):
                meta = m.metadata if m.metadata else {}
            elif isinstance(m, dict):
                meta = m.get('metadata', {})
            else:
                meta = {}
            
            # Ensure metadata is a dict
            if not isinstance(meta, dict):
                meta = {}
            
            # Extract score
            score = getattr(m, 'score', None) or (m.get('score') if isinstance(m, dict) else None)
            
            # Pinecone sometimes uses distance (lower is better). Convert to similarity if needed
            similarity = None
            if score is not None:
                # If score looks like distance (0..1) we convert
                if 0 <= score <= 1:
                    distance_val = m.get('distance') if isinstance(m, dict) else None
                    similarity = 1 - score if distance_val is not None else score
                else:
                    # score may already be a similarity in 0..100 or some other scale
                    similarity = score

            # Normalize similarity to range [0.0, 1.0]
            try:
                if similarity is not None:
                    sim_val = float(similarity)
                    # If scale appears to be percent (0..100), convert to 0..1
                    if sim_val > 1.0 and sim_val <= 100.0:
                        sim_val = sim_val / 100.0
                    # Clamp to [0,1]
                    sim_val = max(0.0, min(1.0, sim_val))
                    similarity = sim_val
            except Exception:
                # If conversion fails, keep similarity as None to avoid wrong comparisons
                similarity = None

            text = meta.get('text', '') or (getattr(m, 'document', None) or '')
            
            # Filter by label if requested
            if label_filter and meta.get('label') != label_filter:
                continue
            if similarity is not None and similarity < similarity_threshold:
                continue

            matches.append({
                'id': getattr(m, 'id', None) or (m.get('id') if isinstance(m, dict) else None),
                'text': text,
                'metadata': meta,
                'score': score,
                'similarity': similarity
            })

        logger.info(f"🔍 Pinecone search returned {len(matches)} matches (top_k={top_k})")
        return matches
    except Exception as e:
        logger.error(f"❌ Pinecone search failed: {e}")
        return []


def get_dynamic_few_shot_examples(
    bug_text: str,
    top_k: int = 5,
    use_local_embeddings: bool = False
) -> List[Dict[str, Any]]:
    if not PINECONE_AVAILABLE:
        return []
    try:
        # Search and return only items with metadata.type == 'example'
        results = search_similar_classified_bugs(bug_text, top_k=top_k, use_local_embeddings=use_local_embeddings, similarity_threshold=0.0)
        examples = []
        for r in results:
            meta = r.get('metadata', {})
            if meta.get('type') == 'example':
                examples.append({'id': r.get('id'), 'description': r.get('text'), 'label': meta.get('label')})
        return examples[:top_k]
    except Exception as e:
        logger.error(f"❌ Failed to get dynamic examples from Pinecone: {e}")
        return []


def get_vector_store_stats(use_local_embeddings: bool = False) -> Dict[str, Any]:
    if not PINECONE_AVAILABLE:
        return {"available": False, "message": "Pinecone not initialized"}
    try:
        # Minimal stats
        return {"available": True, "index": os.getenv('PINECONE_INDEX_NAME', 'default')}
    except Exception as e:
        return {"available": False, "error": str(e)}


# Module-run helper
if __name__ == "__main__":
    print("Initializing Pinecone-backed vector store...")
    ok = init_vector_store()
    print("OK" if ok else "FAILED")
