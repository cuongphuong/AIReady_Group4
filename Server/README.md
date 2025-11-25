# BugClassifier Server (Python)

Thư mục này chứa một API REST nhẹ được xây dựng bằng Python để bao bọc logic phân loại bug hiện có.

## Bắt Đầu Nhanh (Windows PowerShell)

```pwsh
cd D:\AIReady_Group4\Server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Khởi động API:

Nếu đã `cd` vào thư mục `Server`, chạy:
```pwsh
uvicorn api:app --reload --port 8000
```

Hoặc từ thư mục gốc (một cấp trên `Server`), chạy:
```pwsh
uvicorn Server.api:app --reload --port 8000
```

## Endpoints

- `GET /health` — kiểm tra sức khỏe đơn giản
- `POST /classify` — JSON body `{ "text": "..." }` trả về `{ "results": [...] }`

## Ghi Chú

- Server sử dụng hàm `bug_classifier.classify_bug` hiện có và yêu cầu `OPENAI_API_KEY` (và tùy chọn `OPENAI_API_BASE_URL`) trong các biến môi trường. Sử dụng file `.env` hoặc đặt các biến môi trường trước khi chạy.
- CORS được bật cho `http://localhost:5173` để cho phép frontend dev server gọi API.

---

## Các Kỹ Thuật Tối Ưu Hóa

| # | Kỹ Thuật | Vị Trí | Mô Tả | Lợi Ích |
|---|----------|--------|--------|---------|
| 1 | **Function Definition** | `bug_classifier.py` lines 340-360 | Model trả về JSON strict với các field định nghĩa: `label`, `reason`, `team`, `severity`, `tags` | Phân tích xác định, không ambiguous, tích hợp dễ dàng |
| 2 | **Batch Processing** | `bug_classifier.py` lines 400-480 (hàm `batch_classify`) | Phân loại nhiều bug trong **một lệnh gọi LLM** bằng mảng JSON có index | Giảm 80-90% lệnh gọi API, tiết kiệm chi phí và latency |
| 3 | **Async/Await** | `bug_classifier.py` lines 285-395 | Các hàm async: `classify_bug()`, `batch_classify()`, `_call_model_with_retries()` | I/O không blocking, xử lý yêu cầu đồng thời, tận dụng tài nguyên tốt |
| 4 | **Retry & Backoff** | `bug_classifier.py` lines 287-300 (hàm `_call_model_with_retries`) | Retry tự động với exponential backoff: 0.5s → 1s → 2s → fail | Xử lý lỗi API tạm thời, tăng độ tin cậy |
| 5 | **Heuristic Pre-filtering** | `bug_classifier.py` lines 303-325 (hàm `_quick_heuristic_for_text`) | Matching từ khóa nhanh để phân loại bug rõ ràng **mà không cần LLM** | 30-50% bug matched → chi phí LLM = 0 |
| 6 | **Type-Safe API** | `api.py` (models Pydantic) | ClassifyRequest, ClassifyItem, ClassifyResponse | Auto-validation, OpenAPI docs, IDE type hints |
| 7 | **Fallback Logic** | `bug_classifier.py` lines 380-395 | Thứ tự: JSON parse → regex extract → fallback mặc định | Xử lý lỗi mạnh mẽ cho phản hồi LLM không hoàn hảo |
| 8 | **Thread Pool** | `bug_classifier.py` line 291 (`asyncio.to_thread`) | Chạy blocking `client.chat.completions.create()` trong thread pool | Giữ FastAPI async responsive khi xử lý LLM dài |

---

## Tác Động Hiệu Suất

| Kịch Bản | Không Tối Ưu | Có Tối Ưu | Cải Thiện |
|----------|--------------|-----------|---------|
| 100 bugs (cache lạnh) | 100 lệnh gọi LLM | ~50-70 lệnh (batch + heuristic) | **40-50% giảm** |
| 100 bugs (cache nóng) | 100 x 3s = 5 phút | 10 lệnh x 3s + 90 heuristic = ~30s | **90% nhanh hơn** |
| Lỗi API tạm thời | Thất bại | Auto-retry với backoff | **Rất bền vững** |
| Yêu cầu đồng thời | Chặn/chậm | Async xử lý tất cả đồng thời | **Không blocking** |

---

## Cấu Trúc Thư Mục

```
Server/
├── bug_classifier.py       # Logic phân loại cốt lõi (heuristic, async, batch, retry)
├── api.py                  # FastAPI wrapper với endpoints /health và /classify
├── requirements.txt        # Dependencies (fastapi, uvicorn, openai, python-dotenv, pydantic)
├── .env                    # Biến môi trường (OPENAI_API_KEY, OPENAI_API_BASE_URL)
├── start_server.bat        # Script batch Windows để khởi động server
├── run_server.ps1          # Script PowerShell Windows để khởi động server
├── test_async_classifier.py   # Unit tests cho heuristic và batch functions
└── README.md               # File này
```

---

## Kiểm Thử

Chạy unit tests:
```pwsh
cd D:\AIReady_Group4\Server
python test_async_classifier.py
```
