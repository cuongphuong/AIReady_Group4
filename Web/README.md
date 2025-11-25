# BugClassifier Chat (Web)

BugClassifier Chat là một scaffold nhỏ sử dụng Vite + React cho giao diện chat chuyên biệt để phân loại báo cáo bug. Nó nằm trong thư mục `Web`.

## Bắt Đầu Nhanh (Windows PowerShell)

```powershell
cd d:\AIReady_Group4\Web
npm install
npm run dev
```

Mở `http://localhost:5173` trong trình duyệt.

## Tính Năng Chính

- **`src/components/ChatWindow.jsx`** — danh sách tin nhắn + composer
- **`src/components/Message.jsx`** — bubble tin nhắn
- **`src/components/Sidebar.jsx`** — quản lý hội thoại (create, select, rename, delete)
- **`src/styles.css`** — styling căn bản + theme variables (light/dark)
- **Conversation Persistence** — lưu hội thoại vào localStorage
- **File Upload** — hỗ trợ CSV/XLSX preview và xử lý
- **CSV Export** — xuất kết quả phân loại thành CSV
- **Theme Switcher** — chuyển đổi giữa chế độ sáng/tối

## Kiến Trúc

```
Web/
├── src/
│   ├── components/
│   │   ├── App.jsx            # Component chính
│   │   ├── Sidebar.jsx        # Quản lý hội thoại + modal delete
│   │   ├── ChatWindow.jsx     # Messages + composer + file upload
│   │   └── Message.jsx        # Bubble tin nhắn + typing indicator
│   ├── styles.css             # Styling toàn cục + theme variables
│   └── main.jsx               # Entry point
├── public/
├── index.html
├── package.json
├── vite.config.js
└── README.md                  # File này
```

## Kết Nối Backend

Frontend gửi POST request đến `http://localhost:8000/classify`:

```javascript
const response = await fetch("http://localhost:8000/classify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Nút bị lệch vị trí giao diện" })
});

// Response format:
// {
//   "results": [
//     { "text": "Nút bị lệch...", "label": "UI", "raw": "...", "team": "Frontend Team" }
//   ]
// }
```

## Bước Tiếp Theo

- ✅ Kết nối backend API (đã hoàn thành)
- ✅ Persistent hội thoại (đã hoàn thành)
- ✅ Upload file & preview (đã hoàn thành)
- ✅ Theme switcher (đã hoàn thành)
- ✅ Modal delete confirmation (đã hoàn thành)
- 🔄 Message streaming (streaming responses từ backend)
- 🔄 Authentication (đăng nhập người dùng)
- 🔄 Real-time collaboration (chia sẻ hội thoại)

## Biến Môi Trường

Hiện tại backend được hardcode tại `http://localhost:8000`. Để sử dụng với server khác, hãy cập nhật URL trong `ChatWindow.jsx` hoặc tạo file `.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```
