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

## Khởi Tạo Database

Trước khi sử dụng lần đầu, khởi tạo database:

```pwsh
python init_database.py
```

Database sẽ tự động tạo khi start server nếu chưa tồn tại.

## Endpoints

### Core API
- `GET /health` — kiểm tra sức khỏe đơn giản
- `POST /classify` — JSON body `{ "text": "..." }` trả về `{ "results": [...] }`
- `POST /upload?session_id={id}` — Upload file CSV/XLSX để phân loại (lưu vào DB + disk)
- `POST /download-excel` — Tải xuống kết quả dưới dạng Excel

### Chat Management
- `POST /chat/sessions` — Tạo chat session mới
- `GET /chat/sessions` — Lấy tất cả chat sessions
- `PUT /chat/sessions/{id}` — Cập nhật tiêu đề chat
- `DELETE /chat/sessions/{id}` — Xóa chat session (cascade delete messages + uploads)
- `POST /chat/sessions/{id}/messages` — Thêm tin nhắn vào chat
- `GET /chat/sessions/{id}/messages` — Lấy tin nhắn của session

### File & Statistics
- `GET /uploads?session_id={id}` — Danh sách file uploads
- `GET /statistics` — Thống kê tổng quan (sessions, messages, uploads, bugs classified)

## Ghi Chú

- Server sử dụng hàm `bug_classifier.classify_bug` và `bug_classifier.batch_classify` với **OpenAI Function Calling** để đảm bảo structured output.
- Yêu cầu các biến môi trường trong file `.env`:
  - `OPENAI_API_KEY` — API key của OpenAI (bắt buộc)
  - `OPENAI_API_BASE_URL` — Base URL tùy chỉnh (tùy chọn)
  - `MODEL_NAME` — Tên model (mặc định: `gpt-5`)
- CORS được bật cho `http://localhost:5173` để cho phép frontend dev server gọi API.

---

## Các Kỹ Thuật Tối Ưu Hóa

| # | Kỹ Thuật | Vị Trí | Mô Tả | Lợi Ích |
|---|----------|--------|--------|---------|
| 1 | **Function Calling (OpenAI)** | `bug_classifier.py` lines 340-380, 430-490 | Sử dụng OpenAI Function Calling API với schema định nghĩa cho `label`, `reason`, `team`, `severity`. Model gọi function `classify_bug_report` hoặc `batch_classify_bugs` với arguments đã validate | **Type-safe structured output**, tự động validate theo enum/schema, giảm hallucination, không cần parse JSON thủ công |
| 2 | **Batch Processing** | `bug_classifier.py` lines 425-540 (hàm `batch_classify`) | Phân loại nhiều bug trong **một lệnh gọi LLM** bằng function `batch_classify_bugs` trả về mảng classifications với index | Giảm 80-90% lệnh gọi API, tiết kiệm chi phí và latency, xử lý hàng loạt hiệu quả |
| 3 | **Async/Await** | `bug_classifier.py` lines 287-540 | Các hàm async: `classify_bug()`, `batch_classify()`, `_call_model_with_retries()` | I/O không blocking, xử lý yêu cầu đồng thời, tận dụng tài nguyên tốt, throughput cao |
| 4 | **Retry & Backoff** | `bug_classifier.py` lines 289-302 (hàm `_call_model_with_retries`) | Retry tự động với exponential backoff: 0.5s → 1s → 2s → fail (3 lần retry) | Xử lý lỗi API tạm thời (rate limit, timeout), tăng độ tin cậy hệ thống |
| 5 | **Heuristic Pre-filtering** | `bug_classifier.py` lines 305-327 (hàm `_quick_heuristic_for_text`) | Matching từ khóa nhanh để phân loại bug rõ ràng **mà không cần LLM** (keyword scoring) | 30-50% bug matched → chi phí LLM = 0, latency gần như tức thì |
| 6 | **Type-Safe API** | `api.py` (models Pydantic) | ClassifyRequest, ClassifyItem, ClassifyResponse, UploadResponse, DownloadExcelRequest | Auto-validation, OpenAPI docs tự động, IDE type hints, error handling |
| 7 | **Fallback Logic** | `bug_classifier.py` lines 375-395, 515-530 | Thứ tự: Function call → JSON parse từ content → regex extract → fallback mặc định | Xử lý mạnh mẽ khi model không hỗ trợ function calling hoặc phản hồi không hoàn hảo |
| 8 | **Thread Pool** | `bug_classifier.py` line 293 (`asyncio.to_thread`) | Chạy blocking `client.chat.completions.create()` trong thread pool | Giữ FastAPI async event loop responsive khi xử lý LLM dài, tránh blocking I/O |
| 9 | **Environment Config** | `.env` file + `bug_classifier.py` lines 11-13 | Model name, API key, base URL đọc từ biến môi trường (`MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`) | Dễ dàng thay đổi config mà không cần sửa code, hỗ trợ nhiều môi trường (dev/prod) |

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
├── api.py                  # FastAPI wrapper với tất cả endpoints (chat, upload, stats)
├── database.py             # Database operations (SQLite CRUD)
├── file_storage.py         # File upload storage management
├── init_database.py        # Script khởi tạo database
├── bugclassifier.db        # SQLite database file (tự động tạo)
├── uploads/                # Thư mục lưu file uploads (tự động tạo)
├── requirements.txt        # Dependencies (fastapi, uvicorn, openai, python-dotenv, pydantic)
├── .env                    # Biến môi trường (OPENAI_API_KEY, OPENAI_API_BASE_URL, MODEL_NAME)
├── DATABASE.md             # Documentation chi tiết database schema & API
├── start_server.bat        # Script batch Windows để khởi động server
├── run_server.ps1          # Script PowerShell Windows để khởi động server
├── test_async_classifier.py   # Unit tests cho heuristic và batch functions
└── README.md               # File này
```

---

## Database & File Storage

Hệ thống sử dụng **SQLite** để lưu trữ:
- ✅ Lịch sử chat sessions và messages
- ✅ File uploads metadata (CSV/XLSX)
- ✅ Kết quả phân loại bug chi tiết
- ✅ File uploads được lưu vào thư mục `uploads/`

### Database Schema (4 Tables)

1. **chat_sessions** - Thông tin chat sessions (id, session_id, title, timestamps)
2. **chat_messages** - Tin nhắn trong chat (role: user/assistant, content)
3. **file_uploads** - Metadata file uploads (filename, path, size, row counts)
4. **classification_results** - Kết quả phân loại (bug_text, label, reason, team, severity)

### Features

- **Auto-backup**: Database được lưu trên disk (SQLite)
- **Cascade delete**: Xóa session → tự động xóa messages + uploads + results
- **Indexes**: Tối ưu query performance
- **Graceful degradation**: Nếu DB fail, API vẫn hoạt động (không lưu)
- **File naming**: Timestamp prefix để tránh trùng lặp (`YYYYMMDD_HHMMSS_filename`)
- **Statistics**: Top labels, total counts, etc.

### Usage Examples

**Tạo chat session:**
```bash
curl -X POST http://localhost:8000/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "uuid-here", "title": "New Chat"}'
```

**Upload file với session:**
```bash
curl -X POST "http://localhost:8000/upload?session_id=uuid-here" \
  -F "file=@bugs.csv"
```

**Lấy thống kê:**
```bash
curl http://localhost:8000/statistics
```

Xem chi tiết: [DATABASE.md](DATABASE.md)

---

## Kiểm Thử

Chạy unit tests:
```pwsh
cd D:\AIReady_Group4\Server
python test_async_classifier.py
```
