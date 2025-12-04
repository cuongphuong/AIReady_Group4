# Chatbot Phân Loại Bug Report

## 1. Giới thiệu về chủ đề

### Chủ đề: Chatbot phân loại bug report bằng AI

**Input:**  
- Người dùng nhập nội dung bug report: Thông tin mô tả của bug, có thể bổ sung nguyên nhân gây ra bug và action fix bug để Chatbot có thể thông tin phân loại

**Output:**  
- Hệ thống tự động phân loại bug vào một trong các nhóm mà hệ thống đã define
  ##### VD: 
    - `UI`: Kiểm tra về mặt hiển thị, căn lề, màu sắc của item, ...
    - `Performance`: Hiệu xuất của ứng dụng, thời gian xử lý, thời gian load ứng dụng, ...
    - `Security`: Bảo mật dữ liệu, SQL injection, phân quyền truy cập, ...
    - `Functional`: Logic xử lý không đúng, điều kiện xử lý không đúng, đọc ghi dữ liệu sai điều kiện, call API không đúng, validate sai item, ...
    - `Data`: Dữ liệu test không hợp lệ, dữ liệu input vào màn hình không hợp lệ, file data upload không đúng, ...

**Ứng dụng thực tế:**  
- **Hỗ trợ công việc kiểm thử phần mềm (QA):** Giúp đội QA, Dev, Support dễ dàng xác định loại lỗi để phân công xử lý nhanh hơn.
- **Tiết kiệm thời gian:** Loại bỏ thao tác thủ công, tự động hóa bước phân loại giúp tăng tốc quy trình report bug.
- **Tăng độ chính xác:** Hạn chế sai sót khi phân loại nhờ AI nắm được ngữ nghĩa và ngữ cảnh, tránh các yếu tố chủ quan của con người.
- **Về học tập:** Hiểu cách AI xử lý ngôn ngữ tự nhiên (NLP) và ứng dụng vào công việc thực tế.

---

## 2. Công Nghệ & Mô Hình Sử Dụng

### Công nghệ sử dụng:
- **Ngôn ngữ lập trình:** Python console (Scope workshop1, sẽ bổ sung UI và xử lý file trong các bài sau)
- **Thư viện AI:** OpenAI API (sử dụng mô hình GPT-5 do có khả năng tổng hợp và phân loại dữ liệu tốt nhất)
- **Giao tiếp:** Chatbot hoạt động theo dạng hội thoại giữa 2 role:
  - **System:** Định nghĩa bối cảnh - đóng vai trò chuyên gia phân loại bug, thiết lập luật, tiêu chuẩn phân loại.
  - **User:** Người dùng đầu cuối, nhập dữ liệu bug report qua dòng lệnh (console).

### Mô tả luồng hoạt động:
1. **System**: Đưa ra khung tiêu chuẩn, các nhóm phân loại bug, đảm bảo chatbot hiểu đúng nhiệm vụ (phân loại bug).
2. **User**: Nhập nội dung bug report qua terminal/console.
3. **Chatbot**: Dựa vào đầu vào và tiêu chuẩn của system, AI chọn ra nhãn phù hợp nhất cho bug report.

## 3. ChromaDB Integration - Semantic Search & Vector Database

### 🏗️ Kiến Trúc Phân Loại 3-Tier

Hệ thống sử dụng ChromaDB để tăng độ chính xác và tốc độ phân loại:

```
Input Bug Report
    ↓
┌───────────────────┐
│ 1. Keyword        │  ← Nhanh nhất (regex matching)
│    Heuristic      │
└───────────────────┘
    ↓ (not matched)
┌───────────────────┐
│ 2. ChromaDB       │  ← Semantic search (85% similarity)
│    Semantic Search│     Tìm bugs tương tự đã phân loại
└───────────────────┘
    ↓ (not matched)
┌───────────────────┐
│ 3. Dynamic        │  ← Lấy examples phù hợp từ ChromaDB
│    Few-Shot       │
└───────────────────┘
    ↓
┌───────────────────┐
│ 4. LLM            │  ← GPT-5 / Llama với context tốt hơn
│    Classification │
└───────────────────┘
    ↓
┌───────────────────┐
│ 5. Save to        │  ← Lưu kết quả để học từ dữ liệu
│    ChromaDB       │
└───────────────────┘
```
---

## 5. Hướng Dẫn Cài Đặt, Khởi Động Hệ Thống

### Bước 1: Chuẩn bị môi trường

- **Cài đặt Python**  
  Đảm bảo máy đã cài Python >= 3.8

- **Clone repository**
  ```bash
  git clone https://github.com/cuongphuong/AIReady_Group4.git
  cd AIReady_Group4
  ```

- **Cài đặt dependencies Backend**  
  ```bash
  cd Server
  pip install -r requirements.txt
  ```

- **Cài đặt dependencies Frontend**  
  ```bash
  cd Web
  npm install
  ```

- **Cấu hình API Keys**  
  Tạo file `.env` trong thư mục `Server/`:
  ```bash
  # GPT-4o-mini configuration
  OPENAI_API_KEY=your_openai_key_here
  OPENAI_API_BASE_URL=your_openai_url_here
  MODEL_NAME=GPT-4o-mini

  # Embedding model configuration (ChromaDB)
  DB_OPENAI_API_KEY=your_embedding_key_here
  DB_OPENAI_API_BASE_URL=your_embedding_url_here
  DB_MODEL_NAME=text-embedding-3-small

  # JIRA configuration (optional)
  JIRA_TOKEN=your_jira_token
  JIRA_BASE_URL=https://your-domain.atlassian.net
  JIRA_EMAIL=your_email@example.com
  ```

### Bước 2: Khởi tạo ChromaDB Vector Store

```bash
cd Server
python -c "from models.vector_store import init_vector_store; init_vector_store()"
```

### Bước 3: Khởi động Backend Server

```bash
cd Server
uvicorn api:app --reload --port 8000
```

Server sẽ chạy tại: `http://localhost:8000`

### Bước 4: Khởi động Frontend

```bash
cd Web
npm run dev
```

Web app sẽ chạy tại: `http://localhost:5173`

### Bước 5: Sử dụng ứng dụng

1. Mở trình duyệt và truy cập `http://localhost:5173`
2. Nhập bug report vào ô chat hoặc upload file Excel
3. Chọn model AI (GPT-5 hoặc Llama 3.1)
4. Xem kết quả phân loại với label, reason, team, severity
5. Download kết quả dạng Excel bất cứ lúc nào

**Ví dụ:**
```
Input: "Khi bấm nút Submit không hiện thông báo xác nhận"
Output: 
  - Label: UI
  - Reason: Missing confirmation dialog
  - Team: Frontend Team
  - Severity: Medium
```

---

## 6. Cấu Trúc Project

```
AIReady_Group4/
├── Server/                      # Backend FastAPI
│   ├── api.py                  # REST API endpoints
│   ├── services/
│   │   ├── classifier.py       # 3-tier classification logic
│   │   ├── gpt_service.py      # GPT-5 integration
│   │   ├── llama_service.py    # Llama 3.1 integration
│   │   └── chroma_service.py   # ChromaDB vector database
│   ├── models/
│   │   ├── database.py         # SQLite operations
│   │   └── vector_store.py     # ChromaDB operations
│   ├── config/
│   │   ├── bug_labels.py       # 20 label definitions
│   │   └── examples.py         # Few-shot examples
│   ├── chroma_db/              # Vector database storage
│   ├── gguf/                   # Llama GGUF models
│   └── requirements.txt
│
├── Web/                         # Frontend React
│   ├── src/
│   │   ├── App.jsx             # Main app component
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx  # Chat interface
│   │   │   ├── Sidebar.jsx     # History sidebar
│   │   │   └── Message.jsx     # Message component
│   │   └── styles.css
│   └── package.json
│
├── Docs/                        # Documentation
└── README.md                    # This file
```

---

## 7. Tech Stack

**Backend:**
- Python 3.8+
- FastAPI - REST API framework
- ChromaDB - Vector database
- SQLite - Relational database
- OpenAI API - GPT-5
- Llama 3.1 8B - Local LLM
- sentence-transformers - Local embeddings

**Frontend:**
- React 18
- Vite - Build tool
- Tailwind CSS - Styling

**AI Models:**
- GPT-5 (OpenAI API)
- Llama 3.1 8B Instruct (GGUF quantized)
- text-embedding-3-small (OpenAI embeddings)

---

## 8. Tài Liệu Tham Khảo

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Llama Models](https://github.com/meta-llama/llama)

---

**Last Updated:** 2025-12-04  
**Version:** 2.0.0  
**Contributors:** AIReady Group 4
---