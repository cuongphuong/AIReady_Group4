"""
Bug Classifier Service
Orchestrator service để phân loại bug reports
Điều phối giữa GPT và Llama models
"""

import os
import asyncio
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tắt log HTTP requests từ httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# Enable debug logs for semantic search details
# Uncomment dòng dưới để thấy chi tiết similarity scores:
# logger.setLevel(logging.DEBUG)

# Import configuration từ package config
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BUG_LABELS, TEAM_GROUPS, LABEL_TO_TEAM, FEW_SHOT_EXAMPLES

# Import model services
try:
    from services.gpt_service import get_gpt_service
    GPT_AVAILABLE = True
    logger.info("✅ GPT service available")
except ImportError as e:
    GPT_AVAILABLE = False
    logger.warning(f"⚠️ GPT service not available: {e}")

try:
    from services.llama_service import get_llama_service
    LLAMA_AVAILABLE = True
    logger.info("✅ Llama service available")
    # Pre-initialize singleton để tránh load model mỗi request
    try:
        _llama_singleton = get_llama_service()
        logger.info("✅ Llama singleton pre-loaded")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-load Llama model: {e}")
except ImportError as e:
    LLAMA_AVAILABLE = False
    logger.warning(f"⚠️ Llama service not available: {e}")
except Exception as e:
    LLAMA_AVAILABLE = False
    logger.error(f"❌ Error importing Llama service: {e}")

# Vector store (previously ChromaDB) - now backed by Pinecone via models/vector_store
try:
    from models.vector_store import (
        search_similar_classified_bugs,
        get_dynamic_few_shot_examples,
        add_classified_bug_to_vector_store,
        get_vector_store_stats
    )
    VECTOR_STORE_AVAILABLE = True
    logger.info("✅ Vector store available (Pinecone-backed)")
except ImportError as e:
    VECTOR_STORE_AVAILABLE = False
    logger.warning(f"⚠️ Vector store not available: {e}")


# Helper functions
def _label_line(label, v):
    kws = v.get("keywords") or []
    kw_text = f" (keywords: {', '.join(kws)})" if kws else ""
    return f"- {label}: {v.get('desc', '')}{kw_text}"

label_descriptions = "\n".join(
    [_label_line(label, v) for label, v in BUG_LABELS.items()]
)

# Build example text for few-shot learning
example_text = "\n".join(
    [
        f"Bug report: \"{ex['description']}\"\nPhân loại: {ex['label']}"
        for ex in FEW_SHOT_EXAMPLES
    ]
)

def _quick_heuristic_for_text(description: str):
    """Phân loại nhanh bằng keyword matching - yêu cầu >60% từ trong câu match với keywords"""
    import re
    
    desc_lower = (description or "").lower()
    
    # Tách các từ trong description (bỏ ký tự đặc biệt)
    desc_words = re.findall(r'\b\w+\b', desc_lower)
    total_words = len(desc_words)
    
    if total_words == 0:
        return None
    
    keyword_scores = {}
    keyword_matches = {}
    match_percentages = {}

    for label, v in BUG_LABELS.items():
        kws = v.get("keywords") or []
        matched_words = set()
        matches = []
        
        for kw in kws:
            if not kw:
                continue
            kw_lower = kw.lower()
            # Match whole word
            pattern = r'\b' + re.escape(kw_lower) + r'\b'
            if re.search(pattern, desc_lower):
                matches.append(kw)
                # Đếm các từ trong keyword được match
                kw_words = re.findall(r'\b\w+\b', kw_lower)
                matched_words.update(kw_words)
        
        # Tính % từ trong description được match bởi keywords
        matched_desc_words = sum(1 for word in desc_words if word in matched_words)
        match_percentage = (matched_desc_words / total_words) * 100
        
        keyword_scores[label] = len(matches)
        match_percentages[label] = match_percentage
        if matches:
            keyword_matches[label] = matches

    # Tìm label có % match cao nhất và > 60%
    if match_percentages:
        best_label = max(match_percentages, key=lambda k: match_percentages[k])
        best_percentage = match_percentages[best_label]
        
        if best_percentage > 60:
            team = LABEL_TO_TEAM.get(best_label)
            return {
                "label": best_label,
                "reason": f"Matched {best_percentage:.0f}% keywords: {', '.join(keyword_matches.get(best_label, []))} (heuristic)",
                "team": team,
                }
    return None


async def classify_bug(description: str, model: str = "GPT-5"):
    """
    Phân loại bug report với multi-layer approach:
    1. Keyword heuristic (nhanh nhất)
    2. Semantic search từ vector store (cached bugs)
    3. LLM classification với dynamic few-shot examples
    
    Args:
        description: Mô tả bug
        model: "Llama" hoặc "GPT-5"
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 CLASSIFY_BUG - Model: {model}")
    logger.info(f"📝 Input: {description[:100]}..." if len(description) > 100 else f"📝 Input: {description}")
    
    # Bước 1: Thử keyword heuristic (nhanh nhất)
    heuristic_result = _quick_heuristic_for_text(description)
    if heuristic_result:
        logger.info(f"⚡ Heuristic match: {heuristic_result}")
        # Không lưu vào vector store - quá rõ ràng, chỉ dựa vào keywords
        return heuristic_result
    
    # Xác định embedding type dựa trên model
    use_local = (model == "Llama")
    
    # Track if high similarity match exists (>= 85%)
    has_high_similarity_match = False
    
    # Bước 2: Semantic search trong vector store (bugs đã classify)
    if VECTOR_STORE_AVAILABLE:
        try:
            similar_bugs = search_similar_classified_bugs(
                query=description,
                top_k=1,
                similarity_threshold=0.85,  # High similarity threshold (85%)
                use_local_embeddings=use_local
            )
            
            if similar_bugs and len(similar_bugs) > 0:
                best_match = similar_bugs[0]
                similarity = best_match.get('similarity', 0)
                metadata = best_match.get('metadata', {})
                
                if similarity >= 0.85:
                    has_high_similarity_match = True  # Mark that we found high similarity
                    
                    if metadata.get('label'):  # Very similar bug found with label
                        result = {
                            'label': metadata.get('label'),
                            'reason': f"Similar to: '{best_match.get('text', '')[:60]}...' (semantic: {similarity:.0%})",
                            'team': metadata.get('team'),
                            'severity': metadata.get('severity')
                        }
                        logger.info(f"🎯 Semantic match: {result}")
                        # Không lưu vào vector store - đã có bug tương tự trong DB
                        return result
        except Exception as e:
            logger.warning(f"⚠️ Vector store search failed: {e}")
    
    # Bước 3: Get dynamic few-shot examples từ vector store
    dynamic_examples = FEW_SHOT_EXAMPLES  # Default
    if VECTOR_STORE_AVAILABLE:
        try:
            retrieved_examples = get_dynamic_few_shot_examples(
                description,
                top_k=5,
                use_local_embeddings=use_local
            )
            if retrieved_examples:
                # Convert to format compatible với existing code
                dynamic_examples = [
                    {'description': ex['description'], 'label': ex['label']}
                    for ex in retrieved_examples
                ]
                logger.info(f"✅ Using {len(dynamic_examples)} dynamic examples")
        except Exception as e:
            logger.warning(f"⚠️ Failed to get dynamic examples: {e}")
    
    # Bước 4: LLM classification theo model được chọn
    if model == "Llama":
        # Xử lý LLAMA
        if not LLAMA_AVAILABLE:
            logger.error(f"❌ Llama không khả dụng")
            return {"label": "", "reason": "Llama model not available", "team": None, "severity": None}
        
        try:
            logger.info("🦙 Đang xử lý bằng Llama...")
            llama_service = get_llama_service()
            result = await asyncio.to_thread(
                llama_service.classify_bug,
                description,
                list(BUG_LABELS.keys()),
                dynamic_examples  # Use dynamic examples
            )
            # Map team
            if not result.get('team') and result.get('label'):
                result['team'] = LABEL_TO_TEAM.get(result['label'])
            logger.info(f"✅ Llama result: {result}")
            
            # Lưu vào vector store nếu không có match >= 85%
            if VECTOR_STORE_AVAILABLE and result.get('label') and not has_high_similarity_match:
                try:
                    add_classified_bug_to_vector_store(
                        bug_text=description,
                        label=result['label'],
                        reason=result.get('reason', ''),
                        team=result.get('team'),
                        severity=result.get('severity'),
                        use_local_embeddings=True  # Local embeddings cho Llama
                    )
                    logger.info("💾 Saved to vector store (LOCAL embeddings)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save to vector store: {e}")
            elif has_high_similarity_match:
                logger.info("⏭️  Skipped saving - high similarity match exists (>= 85%)")
            
            return result
        except Exception as e:
            logger.error(f"❌ Llama lỗi: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"label": "", "reason": f"Llama error: {str(e)}", "team": None, "severity": None}
    
    elif model == "GPT-5":
        # Xử lý GPT
        if not GPT_AVAILABLE:
            logger.error("❌ GPT không khả dụng")
            return {"label": "", "reason": "GPT model not available", "team": None, "severity": None}
        
        try:
            logger.info("🤖 Đang xử lý bằng GPT...")
            gpt_service = get_gpt_service()
            result = await gpt_service.classify_bug(
                description=description,
                labels=list(BUG_LABELS.keys()),
                label_descriptions=label_descriptions,
                example_text=example_text,
                team_groups=list(TEAM_GROUPS.keys())
            )
            # Map team
            if not result.get('team') and result.get('label'):
                result['team'] = LABEL_TO_TEAM.get(result['label'])
            logger.info(f"✅ GPT result: {result}")
            
            # Lưu vào vector store nếu không có match >= 85%
            if VECTOR_STORE_AVAILABLE and result.get('label') and not has_high_similarity_match:
                try:
                    add_classified_bug_to_vector_store(
                        bug_text=description,
                        label=result['label'],
                        reason=result.get('reason', ''),
                        team=result.get('team'),
                        severity=result.get('severity'),
                        use_local_embeddings=False  # OpenAI embeddings cho GPT
                    )
                    logger.info("💾 Saved to vector store (OPENAI embeddings)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save to vector store: {e}")
            elif has_high_similarity_match:
                logger.info("⏭️  Skipped saving - high similarity match exists (>= 85%)")
            
            return result
        except Exception as e:
            logger.error(f"❌ GPT lỗi: {e}")
            return {"label": "", "reason": f"GPT error: {str(e)}", "team": None, "severity": None}
    
    else:
        # Model không hỗ trợ
        logger.error(f"❌ Model '{model}' không được hỗ trợ")
        return {"label": "", "reason": f"Unsupported model: {model}", "team": None, "severity": None}


async def batch_classify(descriptions: List[str], model: str = "GPT-5"):
    logger.info(f"\n{'='*80}")
    logger.info(f"📦 BATCH_CLASSIFY - Model: {model}, Count: {len(descriptions)}")
    results: List[Optional[dict]] = [None] * len(descriptions)

    # Xác định embedding type
    use_local = (model == "Llama")
    
    # TIER 1: Heuristic pass (keyword matching >60%)
    remaining_indexes = []
    for i, desc in enumerate(descriptions):
        h = _quick_heuristic_for_text(desc)
        if h:
            results[i] = h
        else:
            remaining_indexes.append(i)
    
    logger.info(f"⚡ Tier 1 Heuristic: {len(descriptions) - len(remaining_indexes)}/{len(descriptions)} matched")

    if not remaining_indexes:
        # Heuristic match 100% → Không cần lưu vào vector store (quá rõ ràng)
        return results
    
    # TIER 2: Vector store Semantic Search (>85% similarity)
    # Track which bugs have high similarity matches (>= 85%)
    high_similarity_bugs = set()  # Set of bug indexes with >= 85% similarity
    semantic_remaining = []
    if VECTOR_STORE_AVAILABLE:
        logger.info(f"🔍 Tier 2 Semantic Search: Checking {len(remaining_indexes)} bugs...")
        for idx in remaining_indexes:
            try:
                query_text = descriptions[idx]
                logger.debug(f"Searching for bug {idx}: {query_text[:100]}...")
                
                similar_bugs = search_similar_classified_bugs(
                    query=query_text,
                    top_k=3,  # Lấy 3 kết quả để debug
                    similarity_threshold=0.0,  # Tắt threshold để thấy tất cả
                    use_local_embeddings=use_local
                )
                
                if similar_bugs and len(similar_bugs) > 0:
                    best_match = similar_bugs[0]
                    similarity = best_match.get('similarity', 0)
                    
                    # Ensure similarity is a number
                    try:
                        similarity = float(similarity) if similarity is not None else 0
                    except (ValueError, TypeError):
                        logger.warning(f"   Bug {idx}: Invalid similarity value: {similarity} (type: {type(similarity)})")
                        similarity = 0
                    
                    metadata = best_match.get('metadata', {})
                    
                    # Debug: In ra top 3 matches
                    if len(similar_bugs) > 1:
                        logger.info(f"   Bug {idx} top matches: " + ", ".join([f"{s.get('similarity', 0):.1%}" for s in similar_bugs[:3]]))
                    
                    # Debug similarity value and type
                    logger.debug(f"   Bug {idx}: similarity = {similarity}, type = {type(similarity)}, >= 0.85? {similarity >= 0.85 if similarity is not None else 'None'}")
                    
                    if similarity >= 0.85:
                        high_similarity_bugs.add(idx)  # Mark as high similarity
                        
                        if metadata.get('label'):
                            results[idx] = {
                                'label': metadata.get('label'),
                                'reason': f"Similar: '{best_match.get('text', '')[:40]}...' ({similarity:.0%})",
                                'team': metadata.get('team'),
                                'severity': metadata.get('severity')
                            }
                            continue
                        else:
                            # Debug: why no label?
                            logger.info(f"   Bug {idx}: High similarity ({similarity:.1%}) but metadata missing 'label' field")
                            logger.debug(f"      metadata = {metadata}")
                            logger.debug(f"      best_match = {best_match}")
                    else:
                        logger.info(f"   Bug {idx}: Found similar but only {similarity:.1%} < 85% threshold")
            except Exception as e:
                logger.error(f"❌ Semantic search failed for bug {idx}: {e}", exc_info=True)
            
            semantic_remaining.append(idx)
        
        semantic_matched = len(remaining_indexes) - len(semantic_remaining)
        logger.info(f"✅ Tier 2 Semantic: {semantic_matched}/{len(remaining_indexes)} matched")
        remaining_indexes = semantic_remaining
    
    if not remaining_indexes:
        # Semantic match từ vector store → Không cần lưu lại (đã có trong DB)
        return results
    
    # TIER 3 & 4: LLM Classification với dynamic few-shot examples
    logger.info(f"🤖 Tier 3+4 LLM: Processing {len(remaining_indexes)} bugs with {model}...")
    if model == "Llama":
        # Xử lý LLAMA batch
        if not LLAMA_AVAILABLE:
            logger.error(f"❌ Llama không khả dụng")
            for idx in remaining_indexes:
                results[idx] = {"label": "", "reason": "Llama model not available", "team": None, "severity": None}
            return results
        
        logger.info("🦙 Đang xử lý batch bằng Llama...")
        llama_service = get_llama_service()
        
        # Get descriptions for remaining bugs
        remaining_descriptions = [descriptions[idx] for idx in remaining_indexes]
        
        # Try batch classification first (more efficient)
        try:
            batch_results = await asyncio.to_thread(
                llama_service.batch_classify_bugs,
                remaining_descriptions,
                list(BUG_LABELS.keys()),
                FEW_SHOT_EXAMPLES
            )
            
            # Map results back
            for i, idx in enumerate(remaining_indexes):
                if i < len(batch_results):
                    result = batch_results[i]
                    # Add team mapping
                    if not result.get('team') and result.get('label'):
                        result['team'] = LABEL_TO_TEAM.get(result['label'])
                    results[idx] = result
            
            logger.info(f"✅ Llama batch classification complete")
        except Exception as e:
            logger.error(f"❌ Llama batch error: {e}, falling back to individual classification")
            # Fallback: classify individually
            for idx in remaining_indexes:
                try:
                    result = await classify_bug(descriptions[idx], model="Llama")
                    results[idx] = result
                except Exception as e2:
                    logger.error(f"❌ Llama lỗi bug {idx}: {e2}")
                    results[idx] = {"label": "", "reason": f"Llama error: {str(e2)}", "team": None, "severity": None}
        
    
    elif model == "GPT-5":
        # Xử lý GPT batch
        if not GPT_AVAILABLE:
            logger.error("❌ GPT không khả dụng")
            for idx in remaining_indexes:
                results[idx] = {"label": "", "reason": "GPT model not available", "team": None, "severity": None}
            return results
        
        logger.info("🤖 Đang xử lý batch bằng GPT...")
        gpt_service = get_gpt_service()
        
        # Lấy descriptions và indexes còn lại
        remaining_descriptions = [descriptions[idx] for idx in remaining_indexes]
        
        # Gọi GPT batch API
        batch_results = await gpt_service.batch_classify(
            descriptions=remaining_descriptions,
            indexes=remaining_indexes,
            labels=list(BUG_LABELS.keys()),
            label_descriptions=label_descriptions,
            example_text=example_text,
            team_groups=list(TEAM_GROUPS.keys())
        )
        
        # Map kết quả về
        for idx, result in batch_results.items():
            if 0 <= idx < len(results):
                if not result.get('team') and result.get('label'):
                    result['team'] = LABEL_TO_TEAM.get(result['label'])
                results[idx] = result
    
    else:
        # Model không hỗ trợ
        logger.error(f"❌ Model '{model}' không được hỗ trợ")
        for idx in remaining_indexes:
            results[idx] = {"label": "", "reason": f"Unsupported model: {model}", "team": None, "severity": None}
        return results

    # Fallback individual classification for None entries
    none_count = sum(1 for r in results if r is None)
    if none_count > 0:
        logger.info(f"🔄 Fallback individual classification for {none_count} bugs")
    
    for i in range(len(results)):
        if results[i] is None:
            try:
                results[i] = await classify_bug(descriptions[i], model=model)
            except Exception as e:
                logger.error(f"❌ Failed to classify bug {i}: {e}")
                results[i] = {
                    "label": "",
                    "reason": "classification_failed",
                    "team": None,
                }
    
    # TIER 5: Lưu kết quả vào vector store (bỏ qua những bug có high similarity >= 85%)
    if VECTOR_STORE_AVAILABLE:
        saved_count = 0
        skipped_count = 0
        for i, result in enumerate(results):
            if result and result.get('label'):
                # Skip saving if bug has high similarity match (>= 85%)
                if i in high_similarity_bugs:
                    skipped_count += 1
                    logger.debug(f"Skipped bug {i} - high similarity match exists")
                    continue
                    
                try:
                    add_classified_bug_to_vector_store(
                        bug_text=descriptions[i],
                        label=result['label'],
                        reason=result.get('reason', ''),
                        team=result.get('team'),
                        severity=result.get('severity'),
                        use_local_embeddings=use_local
                    )
                    saved_count += 1
                except Exception as e:
                    logger.debug(f"Failed to save bug {i}: {e}")
        
        logger.info(f"💾 Saved {saved_count}/{len(results)} results to vector store (skipped {skipped_count} with high similarity)")
    
    logger.info(f"✅ Batch classification complete: {len(results)} results")
    logger.info(f"{'='*80}\n")
    return results


# CLI interface khi chạy trực tiếp
if __name__ == "__main__":
    bug_report = input("Nhập nội dung bug report: ")

    try:
        res = asyncio.run(classify_bug(bug_report))
    except Exception as e:
        print(f"Classification error: {e}")
        res = None

    if isinstance(res, dict):
        print(
            f"\nBug report: {bug_report}\nPhân loại: {res.get('label')}\nLý do: {res.get('reason')}"
        )
    else:
        print(f"\nBug report: {bug_report}\nPhân loại: {res}")
    input(".")
