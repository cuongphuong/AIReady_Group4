import os
import json
import re
import asyncio
import time
from typing import List, Optional
from dotenv import load_dotenv
import openai

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE_URL")
model_name = os.getenv("MODEL_NAME", "gpt-5")

# 1. Khởi tạo client với API key
client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key
)

# 2. Danh sách nhãn bug - mở rộng để bao phủ nhiều loại hơn và thêm alias/keywords hỗ trợ
# Structure: key -> { desc: str, keywords: [str] }
BUG_LABELS = {
    "UI": {
        "desc": "Vấn đề về giao diện: bố cục, căn lề, màu sắc, kích thước, hiển thị sai phần tử",
        "keywords": ["giao diện", "căn lề", "màu", "layout", "hiển thị"],
        "examples": ["Nút bị lệch", "Text bị tràn ra ngoài"],
        "severity_hint": "Low"
    },
    "UX": {
        "desc": "Trải nghiệm người dùng: luồng thao tác khó hiểu, hành vi không trực quan, tương tác kém",
        "keywords": ["trải nghiệm", "ux", "khó hiểu", "flow", "ngắt quãng"],
        "examples": ["Flow đăng ký gây nhầm lẫn"],
        "severity_hint": "Medium"
    },
    "Accessibility": {
        "desc": "Khả năng truy cập: hỗ trợ screen-reader, tab order, contrast, ARIA",
        "keywords": ["accessibility", "a11y", "contrast", "aria", "screen-reader"],
        "examples": ["Trang không đọc được bởi screen reader"],
        "severity_hint": "Medium"
    },
    "Frontend": {
        "desc": "Lỗi phía frontend: rendering, state, bundle, JS errors",
        "keywords": ["frontend", "react", "vue", "render", "javascript", "js error"],
        "examples": ["Console lỗi TypeError trên trang chính"],
        "severity_hint": "Medium"
    },
    "Backend": {
        "desc": "Lỗi phía backend: exception, 5xx, API trả lỗi, timeout",
        "keywords": ["500", "5xx", "server error", "timeout", "exception"],
        "examples": ["API /orders trả 500 khi payload rỗng"],
        "severity_hint": "High"
    },
    "Database": {
        "desc": "Vấn đề database: query chậm, deadlock, migration lỗi, mất dữ liệu",
        "keywords": ["db", "database", "query", "deadlock", "migration", "index"],
        "examples": ["Query báo cáo mất 30s do thiếu index"],
        "severity_hint": "High"
    },
    "Network": {
        "desc": "Lỗi mạng: kết nối thất bại, DNS, latency, timeouts liên quan đến mạng",
        "keywords": ["timeout", "dns", "latency", "network", "kết nối"],
        "examples": ["Service B không reachable từ service A"],
        "severity_hint": "High"
    },
    "Performance": {
        "desc": "Hiệu suất: thời gian tải, độ trễ tương tác, chậm phản hồi",
        "keywords": ["chậm", "load", "timeout", "lag", "slow"],
        "examples": ["Trang dashboard mất nhiều giây để render"],
        "severity_hint": "Medium"
    },
    "Memory": {
        "desc": "Vấn đề bộ nhớ: leak, tăng tiêu thụ bộ nhớ hoặc crash do OOM",
        "keywords": ["memory", "leak", "oom", "out of memory", "bị đầy bộ nhớ"],
        "examples": ["Process tăng memory usage theo thời gian dẫn tới OOM"],
        "severity_hint": "High"
    },
    "Security": {
        "desc": "Bảo mật: lỗ hổng SQL injection, XSS, rò rỉ dữ liệu, phân quyền",
        "keywords": ["xss", "sql injection", "bảo mật", "rò rỉ", "auth", "vulnerability"],
        "examples": ["Endpoint leak thông tin người dùng"],
        "severity_hint": "Critical"
    },
    "Functional": {
        "desc": "Chức năng: logic sai, tính toán nhầm, validate không đúng, hành vi sai",
        "keywords": ["logic", "không đúng", "validate", "sai kết quả", "bug chức năng"],
        "examples": ["Tính tổng báo cáo sai 10%"],
        "severity_hint": "High"
    },
    "Data": {
        "desc": "Dữ liệu: input/CSV/xlsx sai định dạng, dữ liệu test không hợp lệ",
        "keywords": ["dữ liệu", "csv", "xlsx", "format", "invalid data", "encoding"],
        "examples": ["CSV import fail do header mismatch"],
        "severity_hint": "Medium"
    },
    "Integration": {
        "desc": "Tích hợp: lỗi gọi API bên thứ 3, kết nối service, đồng bộ dữ liệu",
        "keywords": ["api", "integration", "kết nối", "third-party", "webhook"],
        "examples": ["Webhook của payment provider trả lỗi 400"],
        "severity_hint": "High"
    },
    "Compatibility": {
        "desc": "Tương thích: trình duyệt, phiên bản OS, thiết bị di động khác nhau",
        "keywords": ["browser", "compatibility", "chrome", "firefox", "ios", "android"],
        "examples": ["Layout bị vỡ trên iOS 14"],
        "severity_hint": "Medium"
    },
    "Localization": {
        "desc": "Bản địa hóa: text/locale sai, format ngày giờ, ngôn ngữ hiển thị",
        "keywords": ["locale", "dịch", "translation", "ngôn ngữ", "định dạng"],
        "examples": ["Format ngày ở VN hiển thị MM/DD thay vì DD/MM"],
        "severity_hint": "Low"
    },
    "Crash": {
        "desc": "Ứng dụng/crash process: exception, app crash, stacktrace",
        "keywords": ["crash", "exception", "stacktrace", "fatal"],
        "examples": ["App crash khi mở modal X"],
        "severity_hint": "Critical"
    },
    "Observability": {
        "desc": "Quan sát & giám sát: logging, metrics, tracing thiếu hoặc sai",
        "keywords": ["logging", "metrics", "tracing", "monitoring", "alert"],
        "examples": ["Missing logs for payment failures"],
        "severity_hint": "Medium"
    },
    "Build": {
        "desc": "Quá trình build/deploy: lỗi CI, package, versioning",
        "keywords": ["ci", "build", "deploy", "pipeline", "artifact"],
        "examples": ["Build pipeline failed on step publish"],
        "severity_hint": "High"
    },
    "Privacy": {
        "desc": "Quyền riêng tư: rò rỉ PII, lưu trữ không đúng, consent",
        "keywords": ["privacy", "pii", "consent", "gdpr", "personal data"],
        "examples": ["User emails exposed in public API"],
        "severity_hint": "Critical"
    },
    "Config": {
        "desc": "Cấu hình: feature flags, env vars, cấu hình sai gây lỗi",
        "keywords": ["config", "env", "feature flag", "settings"],
        "examples": ["Missing env var causes service not to start"],
        "severity_hint": "High"
    }
}

# 6 nhóm team chính (mỗi nhóm tương ứng với 1 team trong dự án)
# Nhóm hóa các label chi tiết thành 6 nhóm tổng quát để phân công team
TEAM_GROUPS = {
    "Frontend Team": ["UI", "UX", "Accessibility", "Frontend", "Compatibility", "Localization"],
    "Backend Team": ["Backend", "Database", "Integration", "Functional"],
    "Data Team": ["Data", "Observability"],
    "Security & Privacy Team": ["Security", "Privacy"],
    "Platform/Performance Team": ["Performance", "Memory", "Crash"],
    "Infrastructure/DevOps Team": ["Build", "Network", "Config"]
}

# Inverse map: detailed label -> team name
LABEL_TO_TEAM = {}
for team, labels in TEAM_GROUPS.items():
    for l in labels:
        LABEL_TO_TEAM[l] = team

# 3. Ví dụ mẫu cho few-shot
FEW_SHOT_EXAMPLES = [
    # UI
    {
        "description": "Trên Chrome desktop, nút 'Gửi' trong form phản hồi bị dịch xuống dưới và bị che một phần bởi footer; theo thiết kế nó phải nằm trong cùng hàng với ô nhập.",
        "label": "UI"
    },
    # UX
    {
        "description": "Luồng đăng ký yêu cầu nhập quá nhiều thông tin ngay từ bước đầu, người dùng dễ bỏ dở; không có hướng dẫn rõ ràng cho trường bắt buộc.",
        "label": "UX"
    },
    # Accessibility
    {
        "description": "Các nhãn input thiếu thuộc tính aria-label; tab order nhảy không tuần tự, màn hình đọc (screen reader) không đọc được nhãn của checkbox.",
        "label": "Accessibility"
    },
    # Frontend
    {
        "description": "Console báo lỗi 'TypeError: Cannot read property x of undefined' khi mở trang báo cáo, component không render được dữ liệu trả về API.",
        "label": "Frontend"
    },
    # Backend
    {
        "description": "API /orders trả mã lỗi 500 khi gửi payload gồm order_items rỗng; stacktrace chỉ ra lỗi NullPointer trong service xử lý order.",
        "label": "Backend"
    },
    # Database
    {
        "description": "Query báo cáo sales trên production mất 45s, explain plan cho thấy full table scan do thiếu index trên cột created_at.",
        "label": "Database"
    },
    # Network
    {
        "description": "Service A không thể kết nối tới Service B (DNS lookup failed) trong giờ cao điểm, gây timeout liên tục.",
        "label": "Network"
    },
    # Performance
    {
        "description": "Trang dashboard khởi tạo chậm (TTFB > 2s) và chuyển tab trong SPA bị lag khi có >100 items hiển thị.",
        "label": "Performance"
    },
    # Memory
    {
        "description": "Sau vài ngày chạy, process backend tăng dần memory usage và cuối cùng OOM kill; không có dấu hiệu GC giải phóng đúng.",
        "label": "Memory"
    },
    # Security
    {
        "description": "Phát hiện header Authorization bị echo lại trong response body của một endpoint, có khả năng rò rỉ token cho client.",
        "label": "Security"
    },
    # Functional
    {
        "description": "Tính năng tính hoa hồng trả về kết quả sai: tổng không khớp với dữ liệu nguồn, vì lọc điều kiện áp dụng sai trong code.",
        "label": "Functional"
    },
    # Data
    {
        "description": "Người dùng upload file CSV, hệ thống không nhận format vì header khác tên; file không có validation nên chập dữ liệu vào DB.",
        "label": "Data"
    },
    # Integration
    {
        "description": "Webhook từ payment provider trả về HTTP 400 sau một thay đổi phiên bản API của họ, dẫn đến giao dịch không được xác nhận.",
        "label": "Integration"
    },
    # Compatibility
    {
        "description": "Layout và một số icon bị vỡ khi mở trên Safari iOS 13, nhưng hoạt bình thường trên các trình duyệt hiện đại.",
        "label": "Compatibility"
    },
    # Localization
    {
        "description": "Ngày hiển thị trên báo cáo ở môi trường VN đang theo định dạng MM/DD thay vì DD/MM, gây nhầm lẫn cho người dùng.",
        "label": "Localization"
    },
    # Crash
    {
        "description": "Ứng dụng mobile crash ngay sau khi nhấn nút 'Tải dữ liệu', stacktrace chỉ rõ NullReference trong module sync.",
        "label": "Crash"
    },
    # Observability
    {
        "description": "Thiếu logs cho luồng xử lý thanh toán; khi có thất bại không có trace id để điều tra, alert không được trigger.",
        "label": "Observability"
    },
    # Build
    {
        "description": "Pipeline CI bị lỗi step build khi cập nhật dependency X; artifact không được publish đến registry, blocking release.",
        "label": "Build"
    },
    # Privacy
    {
        "description": "Một endpoint công khai trả về danh sách user bao gồm email và phone number; rủi ro rò rỉ PII.",
        "label": "Privacy"
    },
    # Config
    {
        "description": "Sau deploy, service không khởi động do thiếu biến môi trường REQUIRED_API_KEY trong config, gây downtime.",
        "label": "Config"
    }
]

# 4. Hàm phân loại bug
def _label_line(label, v):
    kws = v.get('keywords') or []
    kw_text = f" (keywords: {', '.join(kws)})" if kws else ''
    return f"- {label}: {v.get('desc', '')}{kw_text}"


label_descriptions = "\n".join([
    _label_line(label, v) for label, v in BUG_LABELS.items()
])


# Helper: call the LLM in a thread with retries and exponential backoff
async def _call_model_with_retries(call_kwargs: dict, retries: int = 3, backoff_factor: float = 0.5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # client.chat.completions.create is blocking in this environment; run in thread
            resp = await asyncio.to_thread(client.chat.completions.create, **call_kwargs)
            return resp
        except Exception as e:
            last_exc = e
            wait = backoff_factor * (2 ** (attempt - 1))
            await asyncio.sleep(wait)
    # all retries exhausted
    raise last_exc


def _quick_heuristic_for_text(description: str):
    desc_lower = (description or '').lower()
    keyword_scores = {}
    keyword_matches = {}
    for label, v in BUG_LABELS.items():
        kws = v.get('keywords') or []
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
            top_scores = [s for s in keyword_scores.values() if s == keyword_scores[best_label]]
            if len(top_scores) == 1:
                team = LABEL_TO_TEAM.get(best_label)
                return {
                    'label': best_label,
                    'reason': f"Matched keywords: {', '.join(keyword_matches.get(best_label, []))} (heuristic)",
                    'team': team
                }
    return None


async def classify_bug(description: str):
    """Async classify a single bug description. Uses quick heuristic first, otherwise calls the LLM with function calling."""
    # heuristic quick path
    h = _quick_heuristic_for_text(description)
    if h:
        return h

    # Build prompt for single item classification
    example_text = "\n".join([
        f"Bug report: \"{ex['description']}\"\nPhân loại: {ex['label']}"
        for ex in FEW_SHOT_EXAMPLES
    ])

    prompt = f"""
Bạn là một trợ lý phân loại bug chuyên gia QA. Các nhãn có sẵn là:
{label_descriptions}

Ví dụ phân loại (few-shot):
{example_text}

Nhiệm vụ: Hãy phân loại báo cáo bug dưới đây vào MỘT trong các nhãn trên.

Báo cáo bug:
<<<
{description}
>>>
    """

    # Define function for structured output
    classify_function = {
        "name": "classify_bug_report",
        "description": "Phân loại bug report vào một trong các nhãn định sẵn",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": list(BUG_LABELS.keys()),
                    "description": "Nhãn phân loại bug (phải chính xác khớp một trong các nhãn có sẵn)"
                },
                "reason": {
                    "type": "string",
                    "description": "Lý do phân loại (một câu ngắn bằng tiếng Việt, không quá 30 từ)"
                },
                "team": {
                    "type": "string",
                    "enum": list(TEAM_GROUPS.keys()),
                    "description": "Team chịu trách nhiệm xử lý bug này"
                },
                "severity": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High", "Critical"],
                    "description": "Mức độ nghiêm trọng của bug"
                }
            },
            "required": ["label", "reason"]
        }
    }

    call_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an expert QA bug classifier. Analyze bug reports and classify them accurately."},
            {"role": "user", "content": prompt}
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
            label = args.get('label')
            reason = args.get('reason') or ''
            team = args.get('team') or LABEL_TO_TEAM.get(label)
            severity = args.get('severity')
            
            return {
                'label': label,
                'reason': reason.strip(),
                'team': team,
                'severity': severity
            }
        except Exception as e:
            print(f"Function call parse error: {e}")
    
    # Fallback: try to parse content as JSON (for models that don't support function calling)
    raw = message.content
    if raw:
        try:
            parsed = json.loads(raw.strip())
            label = parsed.get('label')
            reason = parsed.get('reason') or ''
            team = parsed.get('team') or LABEL_TO_TEAM.get(label)
            if label and label in BUG_LABELS:
                return {'label': label, 'reason': reason.strip(), 'team': team}
        except Exception:
            pass
        
        # Final fallback: regex search
        m = re.search(r"\b({})\b".format('|'.join(re.escape(k) for k in BUG_LABELS.keys())), raw)
        if m:
            return {'label': m.group(1), 'reason': raw}
    
    return {'label': '', 'reason': 'classification_failed', 'team': None}


async def batch_classify(descriptions: List[str]):
    """Classify a list of descriptions. Use heuristic per-item first; batch remaining via a single LLM call with function calling.
    Returns a list of dicts matching input order: {label, reason, team, severity}.
    """
    results: List[Optional[dict]] = [None] * len(descriptions)

    remaining_indexes = []
    for i, desc in enumerate(descriptions):
        h = _quick_heuristic_for_text(desc)
        if h:
            results[i] = h
        else:
            remaining_indexes.append(i)

    if not remaining_indexes:
        # all classified by heuristic
        return results

    # Build batch prompt
    input_list_text = "\n".join([f"[{idx}]: {descriptions[idx]}" for idx in remaining_indexes])
    example_text = "\n".join([
        f"Bug report: \"{ex['description']}\"\nPhân loại: {ex['label']}"
        for ex in FEW_SHOT_EXAMPLES
    ])

    batch_prompt = f"""
Bạn là một trợ lý phân loại bug chuyên gia QA. Các nhãn có sẵn là:
{label_descriptions}

Ví dụ phân loại (few-shot):
{example_text}

Nhiệm vụ: Phân loại các báo cáo bug được liệt kê dưới đây.

Các báo cáo cần phân loại (format [index]: text):
{input_list_text}
    """

    # Define function for batch classification
    batch_classify_function = {
        "name": "batch_classify_bugs",
        "description": "Phân loại nhiều bug reports cùng lúc",
        "parameters": {
            "type": "object",
            "properties": {
                "classifications": {
                    "type": "array",
                    "description": "Danh sách kết quả phân loại cho từng bug",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "Chỉ số của bug trong danh sách input"
                            },
                            "label": {
                                "type": "string",
                                "enum": list(BUG_LABELS.keys()),
                                "description": "Nhãn phân loại bug"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Lý do phân loại (câu ngắn bằng tiếng Việt, không quá 30 từ)"
                            },
                            "team": {
                                "type": "string",
                                "enum": list(TEAM_GROUPS.keys()),
                                "description": "Team chịu trách nhiệm"
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["Low", "Medium", "High", "Critical"],
                                "description": "Mức độ nghiêm trọng"
                            }
                        },
                        "required": ["index", "label", "reason"]
                    }
                }
            },
            "required": ["classifications"]
        }
    }

    call_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are an expert QA bug classifier. Analyze multiple bug reports and classify them accurately."},
            {"role": "user", "content": batch_prompt}
        ],
        "functions": [batch_classify_function],
        "function_call": {"name": "batch_classify_bugs"},
        "max_tokens": 4000,
    }
    if not model_name.startswith("gpt-5"):
        call_kwargs["temperature"] = 0.0

    response = await _call_model_with_retries(call_kwargs, retries=4, backoff_factor=0.6)

    # Extract function call result
    message = response.choices[0].message
    parsed_array = None
    
    if message.function_call:
        try:
            args = json.loads(message.function_call.arguments)
            parsed_array = args.get('classifications', [])
        except Exception as e:
            print(f"Function call parse error: {e}")
    
    # Fallback: try to parse content as JSON array
    if not parsed_array and message.content:
        raw = message.content.strip()
        try:
            parsed_array = json.loads(raw)
            if not isinstance(parsed_array, list):
                parsed_array = None
        except Exception:
            # try to extract a JSON array substring
            m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", raw)
            if m:
                try:
                    parsed_array = json.loads(m.group(1))
                except Exception:
                    parsed_array = None

    # Map parsed results back to results array
    if parsed_array:
        for item in parsed_array:
            try:
                idx = int(item.get('index'))
                label = item.get('label')
                reason = item.get('reason') or ''
                team = item.get('team') or (LABEL_TO_TEAM.get(label) if label in LABEL_TO_TEAM else None)
                severity = item.get('severity')
                if 0 <= idx < len(results):
                    results[idx] = {
                        'label': label if label in BUG_LABELS else label,
                        'reason': reason.strip(),
                        'team': team,
                        'severity': severity
                    }
            except Exception:
                continue

    # For any remaining None entries, fallback to individual classification
    for i in range(len(results)):
        if results[i] is None:
            try:
                results[i] = await classify_bug(descriptions[i])
            except Exception:
                results[i] = {'label': '', 'reason': 'classification_failed', 'team': None}

    return results

# 5. Lấy input từ console
if __name__ == "__main__":
    # 5. Lấy input từ console (chỉ chạy khi file được gọi trực tiếp)
    bug_report = input("Nhập nội dung bug report: ")

    # 6. Phân loại và hiển thị kết quả
    # run the async classifier in an event loop
    try:
        res = asyncio.run(classify_bug(bug_report))
    except Exception as e:
        print(f"Classification error: {e}")
        res = None

    if isinstance(res, dict):
        print(f"\nBug report: {bug_report}\nPhân loại: {res.get('label')}\nLý do: {res.get('reason')}")
    else:
        print(f"\nBug report: {bug_report}\nPhân loại: {res}")
    input('.')