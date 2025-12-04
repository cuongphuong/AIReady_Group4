"""
GPT Service
Xử lý classification sử dụng OpenAI GPT models
"""

import os
import json
import re
import asyncio
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE_URL")
model_name = os.getenv("MODEL_NAME", "gpt-5")

# Khởi tạo OpenAI client
client = openai.OpenAI(base_url=base_url, api_key=api_key)


class GPTService:
    """Service để tương tác với GPT models"""
    
    def __init__(self):
        self.client = client
        self.model_name = model_name
        logger.debug(f"GPT Service initialized with model: {model_name}")
    
    async def _call_model_with_retries(
        self, call_kwargs: dict, retries: int = 3, backoff_factor: float = 0.5
    ):
        """Gọi LLM với retry logic và exponential backoff"""
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                resp = await asyncio.to_thread(
                    self.client.chat.completions.create, **call_kwargs
                )
                return resp
            except Exception as e:
                last_exc = e
                wait = backoff_factor * (2 ** (attempt - 1))
                logger.warning(f"⚠️ Attempt {attempt} failed, retrying in {wait}s...")
                await asyncio.sleep(wait)
        raise last_exc
    
    async def classify_bug(
        self, 
        description: str, 
        labels: List[str],
        label_descriptions: str,
        example_text: str,
        team_groups: List[str]
    ) -> Dict:
        logger.info("\n" + "="*80)
        logger.info("🤖 GPT CLASSIFY_BUG")
        logger.info(f"📝 Input: {description[:100]}..." if len(description) > 100 else f"📝 Input: {description}")
        
        # Build prompt
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
1. Đọc toàn bộ thông tin bug (có thể chứa nhiều trường: No, Summary, Description, Priority, Status, v.v.).
2. TỰ ĐỘNG XÁC ĐỊNH trường nào chứa nội dung mô tả bug chính (thường là Summary, Description, hoặc các trường tương tự).
3. BỎ QUA các thông tin không liên quan (như ID, Create date, Reporter, v.v.).
4. Tập trung vào nội dung mô tả lỗi để xác định từ khóa chính (keywords).
5. So sánh với các nhãn có sẵn, tìm nhãn khớp nhất về mặt ngữ nghĩa.
6. Nếu có nhiều nhãn phù hợp, ưu tiên nhãn cụ thể hơn (VD: "Backend" > "Functional").
7. Đánh giá tác động: Critical (hệ thống sập/bảo mật) > High (chức năng chính lỗi) > Medium (trải nghiệm kém) > Low (hiển thị sai nhỏ).

=== QUY TẮC ===
- KHÔNG bịa ra nhãn mới ngoài danh sách.
- Lý do phải ngắn gọn (< 30 từ) và bằng tiếng Việt.
- Phải chọn đúng team dựa trên nhãn phân loại.
- TỰ ĐỘNG lọc thông tin quan trọng từ dữ liệu đầu vào.

Thông tin bug cần phân loại (có thể chứa nhiều trường, hãy tự động lọc thông tin quan trọng):
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
                        "enum": labels,
                        "description": "Nhãn phân loại bug",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Lý do phân loại (ngắn gọn, < 30 từ, tiếng Việt)",
                    },
                    "team": {
                        "type": "string",
                        "enum": team_groups,
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
            "model": self.model_name,
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
        if not self.model_name.startswith("gpt-5"):
            call_kwargs["temperature"] = 0.0
        
        response = await self._call_model_with_retries(call_kwargs)
        
        # Extract function call result
        message = response.choices[0].message
        if message.function_call:
            try:
                args = json.loads(message.function_call.arguments)
                
                result = {
                    "label": args.get("label"),
                    "reason": (args.get("reason") or "").strip(),
                    "team": args.get("team"),
                    "severity": args.get("severity"),
                }
                logger.info(f"✅ {result['label']} - {result.get('team', 'N/A')}")
                return result
            except Exception as e:
                logger.error(f"❌ Function call parse error: {e}")
        
        # Fallback: parse content as JSON
        raw = message.content
        if raw:
            try:
                parsed = json.loads(raw.strip())
                label = parsed.get("label")
                reason = parsed.get("reason") or ""
                team = parsed.get("team")
                severity = parsed.get("severity")
                
                if label and label in labels:
                    result = {
                        "label": label,
                        "reason": reason.strip(),
                        "team": team,
                        "severity": severity
                    }
                    logger.info(f"✅ {label} (fallback)")
                    return result
            except Exception as e:
                logger.error(f"❌ JSON parse error: {e}")
            
            # Final fallback: regex search
            m = re.search(
                r"\b({})\b".format("|".join(re.escape(k) for k in labels)), raw
            )
            if m:
                result = {"label": m.group(1), "reason": raw, "team": None, "severity": None}
                logger.warning(f"⚠️ Regex fallback result: {result}")
                logger.info("="*80 + "\n")
                return result
        
        logger.error("❌ Classification failed")
        logger.info("="*80 + "\n")
        return {"label": "", "reason": "classification_failed", "team": None, "severity": None}
    
    async def batch_classify(
        self,
        descriptions: List[str],
        indexes: List[int],
        labels: List[str],
        label_descriptions: str,
        example_text: str,
        team_groups: List[str]
    ) -> Dict[int, Dict]:
        """
        Phân loại nhiều bug reports cùng lúc sử dụng GPT
        
        Args:
            descriptions: List các mô tả bug
            indexes: List các index tương ứng
            labels: Danh sách các nhãn có thể
            label_descriptions: Mô tả chi tiết các nhãn
            example_text: Các ví dụ few-shot
            team_groups: Danh sách các team
        
        Returns:
            Dict mapping index -> classification result
        """
        logger.info("\n" + "="*80)
        logger.info(f"🤖 GPT BATCH_CLASSIFY - Count: {len(descriptions)}")
        
        # Build batch prompt
        input_list_text = "\n".join(
            [f"[{idx}]: {descriptions[i]}" for i, idx in enumerate(indexes)]
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
1. Đọc toàn bộ thông tin (có thể chứa nhiều trường như No, Summary, Description, Priority, v.v.).
2. TỰ ĐỘNG XÁC ĐỊNH trường nào chứa nội dung mô tả bug chính.
3. BỎ QUA các thông tin không liên quan (ID, ngày tạo, người báo cáo, v.v.).
4. Xác định từ khóa chính (keywords) từ nội dung mô tả lỗi.
5. So khớp với danh sách nhãn, chọn nhãn phù hợp nhất.
6. Ưu tiên nhãn cụ thể (VD: "Database" > "Backend" nếu liên quan query).
7. Đánh giá severity dựa trên tác động thực tế.

=== QUY TẮC ===
- PHẢI phân loại hết tất cả các bug (bao gồm cả index trong danh sách).
- KHÔNG bỏ sót bug nào.
- KHÔNG bịa ra nhãn mới ngoài danh sách.
- Lý do phải ngắn gọn (< 30 từ) và bằng tiếng Việt.
- TỰ ĐỘNG lọc thông tin quan trọng từ dữ liệu đầu vào.

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
                                    "enum": labels,
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Lý do (< 30 từ)",
                                },
                                "team": {
                                    "type": "string",
                                    "enum": team_groups,
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
            "model": self.model_name,
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
        if not self.model_name.startswith("gpt-5"):
            call_kwargs["temperature"] = 0.0
        
        response = await self._call_model_with_retries(
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
                logger.error(f"❌ Function call parse error: {e}")
        
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
        results = {}
        if parsed_array:
            logger.info(f"📋 Parsed {len(parsed_array)} results from GPT")
            for item in parsed_array:
                try:
                    idx = int(item.get("index"))
                    label = item.get("label")
                    reason = item.get("reason") or ""
                    team = item.get("team")
                    severity = item.get("severity")
                    
                    results[idx] = {
                        "label": label if label in labels else label,
                        "reason": reason.strip(),
                        "team": team,
                        "severity": severity,
                    }
                except Exception as e:
                    logger.error(f"❌ Error parsing item: {e}")
                    continue
        
        logger.info(f"✅ Batch classification complete: {len(results)} results")
        logger.info("="*80 + "\n")
        return results


# Global instance (lazy loading)
_gpt_service = None

def get_gpt_service() -> GPTService:
    """Get singleton GPT service instance"""
    global _gpt_service
    if _gpt_service is None:
        _gpt_service = GPTService()
    return _gpt_service
