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
    """Phân loại nhanh bằng keyword matching (whole word only)"""
    import re
    
    desc_lower = (description or "").lower()
    keyword_scores = {}
    keyword_matches = {}

    for label, v in BUG_LABELS.items():
        kws = v.get("keywords") or []
        score = 0
        matches = []
        for kw in kws:
            if not kw:
                continue
            # Chỉ match whole word để tránh false positive (VD: "load" trong "Download")
            # Use word boundary \b to match complete words only
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            if re.search(pattern, desc_lower):
                score += 1
                matches.append(kw)
        keyword_scores[label] = score
        if matches:
            keyword_matches[label] = matches

    if keyword_scores:
        best_label = max(keyword_scores, key=lambda k: keyword_scores[k])
        # Yêu cầu ít nhất 2 keywords match để tin tưởng hơn (hoặc 1 keyword nếu match duy nhất)
        if keyword_scores[best_label] >= 2:
            top_scores = [
                s for s in keyword_scores.values() if s == keyword_scores[best_label]
            ]
            if len(top_scores) == 1:
                team = LABEL_TO_TEAM.get(best_label)
                return {
                    "label": best_label,
                    "reason": f"Matched keywords: {', '.join(keyword_matches.get(best_label, []))} (heuristic)",
                    "team": team,
                }
        # Nếu chỉ có 1 keyword match và không có label nào khác match, chấp nhận
        elif keyword_scores[best_label] == 1:
            total_matches = sum(1 for s in keyword_scores.values() if s > 0)
            if total_matches == 1:  # Chỉ có 1 label match duy nhất
                team = LABEL_TO_TEAM.get(best_label)
                return {
                    "label": best_label,
                    "reason": f"Matched keyword: {', '.join(keyword_matches.get(best_label, []))} (heuristic)",
                    "team": team,
                }
    return None


async def classify_bug(description: str, model: str = "GPT-5"):
    """
    Phân loại bug report
    
    Args:
        description: Mô tả bug
        model: "Llama" hoặc "GPT-5"
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 CLASSIFY_BUG - Model: {model}")
    logger.info(f"📝 Input: {description[:100]}..." if len(description) > 100 else f"📝 Input: {description}")
    
    # Bước 1: Thử heuristic matching (nhanh nhất)
    heuristic_result = _quick_heuristic_for_text(description)
    if heuristic_result:
        logger.info(f"⚡ Heuristic match: {heuristic_result}")
        return heuristic_result
    
    # Bước 2: Xử lý theo model được chọn
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
                FEW_SHOT_EXAMPLES
            )
            # Map team
            if not result.get('team') and result.get('label'):
                result['team'] = LABEL_TO_TEAM.get(result['label'])
            logger.info(f"✅ Llama result: {result}")
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

    # Heuristic pass
    remaining_indexes = []
    for i, desc in enumerate(descriptions):
        h = _quick_heuristic_for_text(desc)
        if h:
            results[i] = h
        else:
            remaining_indexes.append(i)
    
    logger.info(f"⚡ Heuristic matched: {len(descriptions) - len(remaining_indexes)}/{len(descriptions)}")
    logger.info(f"🔄 Remaining for model: {len(remaining_indexes)}")

    if not remaining_indexes:
        return results
    
    # Xử lý theo model được chọn
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
        
        logger.info(f"✅ Batch classification complete: {len(results)} results")
        return results
    
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
    
    logger.info(f"✅ Batch classification complete: {len(results)} results")
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
