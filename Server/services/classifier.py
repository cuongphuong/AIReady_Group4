"""
Bug Classifier Service
Orchestrator service để phân loại bug reports
Điều phối giữa GPT và Llama models
"""

from config import BUG_LABELS, TEAM_GROUPS, LABEL_TO_TEAM, FEW_SHOT_EXAMPLES
import sys
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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Note: Old GPT and Llama services are now replaced by unified_llm_service
# Keeping imports for backward compatibility (not actively used)
GPT_AVAILABLE = False
LLAMA_AVAILABLE = False

# Unified LLM Service (LangChain integration for GPT-5, Llama, Vector Store)
try:
    from services.unified_llm_service import (
        get_unified_service,
        similarity_search,
        add_bug_to_vectorstore,
        get_dynamic_few_shot_examples,
        get_vector_store_stats
    )

    UNIFIED_SERVICE_AVAILABLE = True
    VECTOR_STORE_AVAILABLE = True
    logger.info("✅ Unified LLM Service available (LangChain + Pinecone)")
except ImportError as e:
    UNIFIED_SERVICE_AVAILABLE = False
    VECTOR_STORE_AVAILABLE = False
    logger.warning(f"⚠️ Unified service not available: {e}")


# Helper functions
def _build_label_descriptions():
    """Build label descriptions with keywords."""
    lines = []
    for label, v in BUG_LABELS.items():
        kws = v.get("keywords") or []
        kw_text = f" (keywords: {', '.join(kws)})" if kws else ""
        lines.append(f"- {label}: {v.get('desc', '')}{kw_text}")
    return "\n".join(lines)


def _build_example_text(examples):
    """Build few-shot example text."""
    return "\n".join([
        f"Bug report: \"{ex['description']}\"\nPhân loại: {ex['label']}"
        for ex in examples
    ])


label_descriptions = _build_label_descriptions()
example_text = _build_example_text(FEW_SHOT_EXAMPLES)


def _create_error_result(reason: str) -> dict:
    """Create standardized error result."""
    return {"label": "", "reason": reason, "team": None, "severity": None}


def _save_to_vectorstore(description: str, result: dict):
    """Save classification result to vector store."""
    try:
        add_bug_to_vectorstore(
            bug_text=description,
            label=result["label"],
            reason=result.get("reason", ""),
            team=result.get("team"),
            severity=result.get("severity"),
        )
        logger.info("💾 Saved to vector store")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save to vector store: {e}")


def _quick_heuristic_for_text(description: str):
    """Phân loại nhanh bằng keyword matching - yêu cầu >60% từ trong câu match với keywords"""
    import re

    desc_lower = (description or "").lower()
    desc_words = re.findall(r"\b\w+\b", desc_lower)
    
    if not desc_words:
        return None

    total_words = len(desc_words)
    match_percentages = {}
    keyword_matches = {}

    for label, v in BUG_LABELS.items():
        kws = v.get("keywords") or []
        matched_words = set()
        matches = []

        for kw in kws:
            if kw and re.search(r"\b" + re.escape(kw.lower()) + r"\b", desc_lower):
                matches.append(kw)
                matched_words.update(re.findall(r"\b\w+\b", kw.lower()))

        if matches:
            matched_desc_words = sum(1 for word in desc_words if word in matched_words)
            match_percentages[label] = (matched_desc_words / total_words) * 100
            keyword_matches[label] = matches

    # Tìm label có % match cao nhất và > 60%
    if match_percentages:
        best_label = max(match_percentages, key=match_percentages.get)
        best_percentage = match_percentages[best_label]

        if best_percentage > 60:
            return {
                "label": best_label,
                "reason": f"Matched {best_percentage:.0f}% keywords: {', '.join(keyword_matches[best_label])} (heuristic)",
                "team": LABEL_TO_TEAM.get(best_label),
            }
    return None


def _try_semantic_search(description: str, k: int = 5):
    """Try semantic search in vector store. Returns (result_dict or None, has_high_similarity_match)."""
    if not VECTOR_STORE_AVAILABLE:
        return None, False
    
    try:
        similar_bugs = similarity_search(
            query=description,
            k=k,
            filter={"type": "bug"},
        )

        if similar_bugs:
            best_match = similar_bugs[0]
            similarity = best_match.get("similarity", 0)
            metadata = best_match.get("metadata", {})

            if similarity >= 0.85:
                if metadata.get("label"):
                    result = {
                        "label": metadata["label"],
                        "reason": f"Similar to: '{best_match.get('text', '')[:60]}...' (semantic: {similarity:.0%})",
                        "team": metadata.get("team"),
                        "severity": metadata.get("severity"),
                    }
                    return result, True
                return None, True  # High similarity but no label
    except Exception as e:
        logger.warning(f"⚠️ Vector store search failed: {e}")
    
    return None, False


async def classify_bug(description: str, model: str = "GPT-5"):
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 CLASSIFY_BUG - Model: {model}")
    logger.info(f"📝 Input: {description[:100]}..." if len(description) > 100 else f"📝 Input: {description}")

    # Bước 1: Thử keyword heuristic (nhanh nhất)
    heuristic_result = _quick_heuristic_for_text(description)
    if heuristic_result:
        logger.info(f"⚡ Heuristic match: {heuristic_result}")
        return heuristic_result

    # Bước 2: Semantic search trong vector store (bugs đã classify)
    has_high_similarity_match = False
    if model == "GPT-5":
        semantic_result, has_high_similarity_match = _try_semantic_search(description)
        if semantic_result:
            logger.info(f"🎯 Semantic match: {semantic_result}")
            return semantic_result

    # Bước 3: Prepare few-shot examples
    dynamic_examples = FEW_SHOT_EXAMPLES
    if model == "GPT-5" and VECTOR_STORE_AVAILABLE:
        try:
            retrieved = get_dynamic_few_shot_examples(description, top_k=5, use_local_embeddings=False)
            if retrieved:
                dynamic_examples = [{"description": ex["description"], "label": ex["label"]} for ex in retrieved]
                logger.info(f"✅ Using {len(dynamic_examples)} dynamic examples")
        except Exception as e:
            logger.warning(f"⚠️ Failed to get dynamic examples: {e}")
    else:
        logger.info(f"✅ Using {len(FEW_SHOT_EXAMPLES)} static examples")

    # Bước 4: LLM classification
    if not UNIFIED_SERVICE_AVAILABLE:
        return _create_error_result("Unified service not available")
    
    try:
        unified_service = get_unified_service()
        result = await unified_service.classify_bug(
            description=description,
            labels=list(BUG_LABELS.keys()),
            model=model,
            label_descriptions=label_descriptions,
            example_text=example_text,
            team_groups=list(TEAM_GROUPS.keys()),
            examples=dynamic_examples
        )
        
        # Map team if not set
        if not result.get("team") and result.get("label"):
            result["team"] = LABEL_TO_TEAM.get(result["label"])
        
        logger.info(f"✅ {model} result: {result}")
        
        # Lưu vào vector store (chỉ GPT-5, bỏ qua nếu có match >= 85%)
        if model == "GPT-5" and VECTOR_STORE_AVAILABLE and result.get("label") and not has_high_similarity_match:
            _save_to_vectorstore(description, result)
        elif has_high_similarity_match:
            logger.info("⏭️  Skipped saving - high similarity match exists")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ {model} classification error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return _create_error_result(f"{model} error: {str(e)}")


def _batch_heuristic_pass(descriptions: List[str], results: List[Optional[dict]]) -> List[int]:
    """Apply heuristic matching. Returns remaining indexes."""
    remaining = []
    for i, desc in enumerate(descriptions):
        h = _quick_heuristic_for_text(desc)
        if h:
            results[i] = h
        else:
            remaining.append(i)
    
    matched = len(descriptions) - len(remaining)
    logger.info(f"⚡ Tier 1 Heuristic: {matched}/{len(descriptions)} matched")
    return remaining


def _batch_semantic_pass(descriptions: List[str], results: List[Optional[dict]], remaining_indexes: List[int]) -> tuple[List[int], set]:
    """Apply semantic search. Returns (remaining_indexes, high_similarity_bugs)."""
    if not VECTOR_STORE_AVAILABLE:
        return remaining_indexes, set()
    
    logger.info(f"🔍 Tier 2 Semantic Search: Checking {len(remaining_indexes)} bugs...")
    high_similarity_bugs = set()
    semantic_remaining = []

    for idx in remaining_indexes:
        logger.info(f"🔎 Semantic search for bug {idx}...")
        try:
            semantic_result, is_high_sim = _try_semantic_search(descriptions[idx], k=3)
            
            if is_high_sim:
                high_similarity_bugs.add(idx)
            
            if semantic_result:
                results[idx] = semantic_result
                continue
                
        except Exception as e:
            logger.error(f"❌ Semantic search failed for bug {idx}: {e}")

        semantic_remaining.append(idx)

    matched = len(remaining_indexes) - len(semantic_remaining)
    logger.info(f"✅ Tier 2 Semantic: {matched}/{len(remaining_indexes)} matched")
    return semantic_remaining, high_similarity_bugs


async def batch_classify(descriptions: List[str], model: str = "GPT-5"):
    logger.info(f"\n{'='*80}")
    logger.info(f"📦 BATCH_CLASSIFY - Model: {model}, Count: {len(descriptions)}")
    results: List[Optional[dict]] = [None] * len(descriptions)

    # TIER 1: Heuristic pass
    remaining_indexes = _batch_heuristic_pass(descriptions, results)
    if not remaining_indexes:
        return results

    # TIER 2: Semantic Search (GPT-5 only)
    high_similarity_bugs = set()
    if model == "GPT-5":
        remaining_indexes, high_similarity_bugs = _batch_semantic_pass(descriptions, results, remaining_indexes)
    
    if not remaining_indexes:
        return results

    # TIER 3 & 4: LLM Classification
    logger.info(f"🤖 Tier 3+4 LLM: Processing {len(remaining_indexes)} bugs with {model}...")
    
    if not UNIFIED_SERVICE_AVAILABLE:
        logger.error("❌ Unified service not available")
        for idx in remaining_indexes:
            results[idx] = _create_error_result("Unified service not available")
        return results
    
    try:
        await _batch_llm_classify(descriptions, results, remaining_indexes, model)
    except Exception as e:
        logger.error(f"❌ Batch classification error: {e}, falling back to individual")
        await _fallback_individual_classify(descriptions, results, remaining_indexes, model)

    # Fallback for None entries
    await _handle_none_results(descriptions, results, model)

    # TIER 5: Save to vector store (GPT-5 only, skip high similarity)
    if VECTOR_STORE_AVAILABLE and model == "GPT-5":
        _batch_save_to_vectorstore(descriptions, results, high_similarity_bugs)

    logger.info(f"✅ Batch classification complete: {len(results)} results")
    logger.info(f"{'='*80}\n")
    return results


async def _batch_llm_classify(descriptions: List[str], results: List[Optional[dict]], remaining_indexes: List[int], model: str):
    """Perform batch LLM classification."""
    unified_service = get_unified_service()
    remaining_descriptions = [descriptions[idx] for idx in remaining_indexes]
    
    if model == "GPT-5":
        batch_results = await unified_service.batch_classify(
            descriptions=remaining_descriptions,
            labels=list(BUG_LABELS.keys()),
            model="GPT-5",
            indexes=remaining_indexes,
            label_descriptions=label_descriptions,
            example_text=example_text
        )
        # Map results (GPT returns Dict[int, Dict])
        for idx, result in batch_results.items():
            if 0 <= idx < len(results):
                if not result.get("team") and result.get("label"):
                    result["team"] = LABEL_TO_TEAM.get(result["label"])
                results[idx] = result
        logger.info(f"✅ GPT-5 batch classification complete")
        
    elif model == "Llama":
        batch_results = await unified_service.batch_classify(
            descriptions=remaining_descriptions,
            labels=list(BUG_LABELS.keys()),
            model="Llama",
            examples=FEW_SHOT_EXAMPLES
        )
        # Map results (Llama returns List[Dict])
        for i, idx in enumerate(remaining_indexes):
            if i < len(batch_results):
                result = batch_results[i]
                if not result.get("team") and result.get("label"):
                    result["team"] = LABEL_TO_TEAM.get(result["label"])
                results[idx] = result
        logger.info(f"✅ Llama batch classification complete")
    else:
        logger.error(f"❌ Model '{model}' không được hỗ trợ")
        for idx in remaining_indexes:
            results[idx] = _create_error_result(f"Unsupported model: {model}")


async def _fallback_individual_classify(descriptions: List[str], results: List[Optional[dict]], remaining_indexes: List[int], model: str):
    """Fallback to individual classification."""
    for idx in remaining_indexes:
        try:
            results[idx] = await classify_bug(descriptions[idx], model=model)
        except Exception as e:
            logger.error(f"❌ Error bug {idx}: {e}")
            results[idx] = _create_error_result(f"Error: {str(e)}")


async def _handle_none_results(descriptions: List[str], results: List[Optional[dict]], model: str):
    """Handle None results with individual classification."""
    none_count = sum(1 for r in results if r is None)
    if none_count > 0:
        logger.info(f"🔄 Fallback individual classification for {none_count} bugs")
        for i in range(len(results)):
            if results[i] is None:
                try:
                    results[i] = await classify_bug(descriptions[i], model=model)
                except Exception as e:
                    logger.error(f"❌ Failed to classify bug {i}: {e}")
                    results[i] = _create_error_result("classification_failed")


def _batch_save_to_vectorstore(descriptions: List[str], results: List[Optional[dict]], high_similarity_bugs: set):
    """Save batch results to vector store."""
    saved_count = 0
    skipped_count = 0
    logger.info(f"💾 Saving results to vector store...")
    
    for i, result in enumerate(results):
        if result and result.get("label"):
            if i in high_similarity_bugs:
                skipped_count += 1
                logger.debug(f"Skipped bug {i} - high similarity match exists")
                continue

            try:
                add_bug_to_vectorstore(
                    bug_text=descriptions[i],
                    label=result["label"],
                    reason=result.get("reason", ""),
                    team=result.get("team"),
                    severity=result.get("severity"),
                )
                saved_count += 1
            except Exception as e:
                logger.debug(f"Failed to save bug {i}: {e}")

    logger.info(f"💾 Saved {saved_count}/{len(results)} (skipped {skipped_count} high similarity)")


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
