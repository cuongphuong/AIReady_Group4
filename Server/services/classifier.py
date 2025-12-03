"""
Bug Classifier Service
Phân loại bug reports sử dụng LLM với Function Calling
"""

import os
import json
import re
import asyncio
from typing import List, Optional
from dotenv import load_dotenv
import openai

# Import configuration từ package config
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BUG_LABELS, TEAM_GROUPS, LABEL_TO_TEAM, FEW_SHOT_EXAMPLES

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE_URL")
model_name = os.getenv("MODEL_NAME", "gpt-5")

# Khởi tạo OpenAI client
client = openai.OpenAI(base_url=base_url, api_key=api_key)


# Helper functions
def _label_line(label, v):
    """Format label description với keywords"""
    kws = v.get("keywords") or []
    kw_text = f" (keywords: {', '.join(kws)})" if kws else ""
    return f"- {label}: {v.get('desc', '')}{kw_text}"


label_descriptions = "\n".join(
    [_label_line(label, v) for label, v in BUG_LABELS.items()]
)


async def _call_model_with_retries(
    call_kwargs: dict, retries: int = 3, backoff_factor: float = 0.5
):
    """Gọi LLM với retry logic và exponential backoff"""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create, **call_kwargs
            )
            return resp
        except Exception as e:
            last_exc = e
            wait = backoff_factor * (2 ** (attempt - 1))
            await asyncio.sleep(wait)
    raise last_exc


def _quick_heuristic_for_text(description: str):
    """Phân loại nhanh bằng keyword matching"""
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
            if kw.lower() in desc_lower:
                score += 1
                matches.append(kw)
        keyword_scores[label] = score
        if matches:
            keyword_matches[label] = matches

    if keyword_scores:
        best_label = max(keyword_scores, key=lambda k: keyword_scores[k])
        if keyword_scores[best_label] > 0:
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
    return None


async def classify_bug(description: str):
    """
    Phân loại một bug report
    Returns: dict với keys: label, reason, team, severity
    """
    # Quick heuristic path
    h = _quick_heuristic_for_text(description)
    if h:
        return h

    # Build few-shot examples
    example_text = "\n".join(
        [
            f"Bug report: \"{ex['description']}\"\nPhân loại: {ex['label']}"
            for ex in FEW_SHOT_EXAMPLES
        ]
    )

    # Optimized prompt (4-part structure, no output format as we use Function Calling)
    prompt = f"""
=== VAI TRÒ ===
Bạn là chuyên gia QA với 10+ năm kinh nghiệm, chuyên phân tích và phân loại bug cho các hệ thống phần mềm lớn.

=== NHIỆM VỤ ===
Phân loại báo cáo bug dưới đây vào CHÍNH XÁC MỘT nhãn phù hợp nhất từ danh sách cho trước.
Đánh giá mức độ nghiêm trọng (severity) và xác định team chịu trách nhiệm.

=== NGỮ CẢNH ===
Các nhãn phân loại có sẵn:
{label_descriptions}

Các ví dụ minh họa:
{example_text}

=== LẬP LUẬN ===
1. Đọc kỹ mô tả bug, xác định từ khóa chính (keywords).
2. So sánh với các nhãn có sẵn, tìm nhãn khớp nhất về mặt ngữ nghĩa.
3. Nếu có nhiều nhãn phù hợp, ưu tiên nhãn cụ thể hơn (VD: "Backend" > "Functional").
4. Đánh giá tác động: Critical (hệ thống sập/bảo mật) > High (chức năng chính lỗi) > Medium (trải nghiệm kém) > Low (hiển thị sai nhỏ).

=== QUY TẮC ===
- KHÔNG bịa ra nhãn mới ngoài danh sách.
- Lý do phải ngắn gọn (< 30 từ) và bằng tiếng Việt.
- Phải chọn đúng team dựa trên nhãn phân loại.

Báo cáo bug cần phân loại:
<<<
{description}
>>>
    """

    # Function definition cho structured output
    classify_function = {
        "name": "classify_bug_report",
        "description": "Phân loại bug report vào một trong các nhãn định sẵn",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": list(BUG_LABELS.keys()),
                    "description": "Nhãn phân loại bug",
                },
                "reason": {
                    "type": "string",
                    "description": "Lý do phân loại (ngắn gọn, < 30 từ, tiếng Việt)",
                },
                "team": {
                    "type": "string",
                    "enum": list(TEAM_GROUPS.keys()),
                    "description": "Team chịu trách nhiệm",
                },
                "severity": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High", "Critical"],
                    "description": "Mức độ nghiêm trọng",
                },
            },
            "required": ["label", "reason"],
        },
    }

    call_kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior QA expert with 10+ years of experience. Follow the structured prompt precisely and output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "functions": [classify_function],
        "function_call": {"name": "classify_bug_report"},
        "max_tokens": 1500,
    }
    if not model_name.startswith("gpt-5"):
        call_kwargs["temperature"] = 0.0

    response = await _call_model_with_retries(call_kwargs)

    # Extract function call result
    message = response.choices[0].message
    if message.function_call:
        try:
            args = json.loads(message.function_call.arguments)
            label = args.get("label")
            reason = args.get("reason") or ""
            team = args.get("team") or LABEL_TO_TEAM.get(label)
            severity = args.get("severity")

            return {
                "label": label,
                "reason": reason.strip(),
                "team": team,
                "severity": severity,
            }
        except Exception as e:
            print(f"Function call parse error: {e}")

    # Fallback: parse content as JSON
    raw = message.content
    if raw:
        try:
            parsed = json.loads(raw.strip())
            label = parsed.get("label")
            reason = parsed.get("reason") or ""
            team = parsed.get("team") or LABEL_TO_TEAM.get(label)
            if label and label in BUG_LABELS:
                return {"label": label, "reason": reason.strip(), "team": team}
        except Exception:
            pass

        # Final fallback: regex search
        m = re.search(
            r"\b({})\b".format("|".join(re.escape(k) for k in BUG_LABELS.keys())), raw
        )
        if m:
            return {"label": m.group(1), "reason": raw}

    return {"label": "", "reason": "classification_failed", "team": None}


async def batch_classify(descriptions: List[str]):
    """
    Phân loại nhiều bug reports cùng lúc
    Returns: list of dicts với keys: label, reason, team, severity
    """
    results: List[Optional[dict]] = [None] * len(descriptions)

    # Heuristic pass
    remaining_indexes = []
    for i, desc in enumerate(descriptions):
        h = _quick_heuristic_for_text(desc)
        if h:
            results[i] = h
        else:
            remaining_indexes.append(i)

    if not remaining_indexes:
        return results

    # Build batch prompt
    input_list_text = "\n".join(
        [f"[{idx}]: {descriptions[idx]}" for idx in remaining_indexes]
    )
    example_text = "\n".join(
        [
            f"Bug report: \"{ex['description']}\"\nPhân loại: {ex['label']}"
            for ex in FEW_SHOT_EXAMPLES
        ]
    )

    batch_prompt = f"""
=== VAI TRÒ ===
Bạn là chuyên gia QA với 10+ năm kinh nghiệm, chuyên phân tích và phân loại bug hàng loạt với độ chính xác cao.

=== NHIỆM VỤ ===
Phân loại TẤT CẢ các báo cáo bug trong danh sách dưới đây.
Mỗi bug phải được gán ĐÚNG MỘT nhãn, kèm lý do, team, và severity.

=== NGỮ CẢNH ===
Các nhãn phân loại có sẵn:
{label_descriptions}

Các ví dụ minh họa:
{example_text}

=== LẬP LUẬN ===
Với mỗi bug:
1. Xác định từ khóa chính (keywords) trong mô tả.
2. So khớp với danh sách nhãn, chọn nhãn phù hợp nhất.
3. Ưu tiên nhãn cụ thể (VD: "Database" > "Backend" nếu liên quan query).
4. Đánh giá severity dựa trên tác động thực tế.

=== QUY TẮC ===
- PHẢI phân loại hết tất cả các bug (bao gồm cả index trong danh sách).
- KHÔNG bỏ sót bug nào.
- KHÔNG bịa ra nhãn mới ngoài danh sách.
- Lý do phải ngắn gọn (< 30 từ) và bằng tiếng Việt.

Danh sách báo cáo cần phân loại (format [index]: text):
{input_list_text}
    """

    batch_classify_function = {
        "name": "batch_classify_bugs",
        "description": "Phân loại nhiều bug reports cùng lúc",
        "parameters": {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "description": "Danh sách kết quả phân loại",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer", "description": "Chỉ số bug"},
                            "label": {
                                "type": "string",
                                "enum": list(BUG_LABELS.keys()),
                            },
                            "reason": {
                                "type": "string",
                                "description": "Lý do (< 30 từ)",
                            },
                            "team": {
                                "type": "string",
                                "enum": list(TEAM_GROUPS.keys()),
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["Low", "Medium", "High", "Critical"],
                            },
                        },
                        "required": ["index", "label", "reason"],
                    },
                }
            },
            "required": ["classifications"],
        },
    }

    call_kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a senior QA expert. Follow the structured prompt. Classify ALL bugs without omission. Output only valid JSON array.",
            },
            {"role": "user", "content": batch_prompt},
        ],
        "functions": [batch_classify_function],
        "function_call": {"name": "batch_classify_bugs"},
        "max_tokens": 4000,
    }
    if not model_name.startswith("gpt-5"):
        call_kwargs["temperature"] = 0.0

    response = await _call_model_with_retries(
        call_kwargs, retries=4, backoff_factor=0.6
    )

    # Extract function call result
    message = response.choices[0].message
    parsed_array = None

    if message.function_call:
        try:
            args = json.loads(message.function_call.arguments)
            parsed_array = args.get("classifications", [])
        except Exception as e:
            print(f"Function call parse error: {e}")

    # Fallback: parse content as JSON array
    if not parsed_array and message.content:
        raw = message.content.strip()
        try:
            parsed_array = json.loads(raw)
            if not isinstance(parsed_array, list):
                parsed_array = None
        except Exception:
            m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", raw)
            if m:
                try:
                    parsed_array = json.loads(m.group(1))
                except Exception:
                    parsed_array = None

    # Map results
    if parsed_array:
        for item in parsed_array:
            try:
                idx = int(item.get("index"))
                label = item.get("label")
                reason = item.get("reason") or ""
                team = item.get("team") or (
                    LABEL_TO_TEAM.get(label) if label in LABEL_TO_TEAM else None
                )
                severity = item.get("severity")
                if 0 <= idx < len(results):
                    results[idx] = {
                        "label": label if label in BUG_LABELS else label,
                        "reason": reason.strip(),
                        "team": team,
                        "severity": severity,
                    }
            except Exception:
                continue

    # Fallback individual classification for None entries
    for i in range(len(results)):
        if results[i] is None:
            try:
                results[i] = await classify_bug(descriptions[i])
            except Exception:
                results[i] = {
                    "label": "",
                    "reason": "classification_failed",
                    "team": None,
                }

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
