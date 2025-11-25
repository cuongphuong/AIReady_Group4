# BugClassifier Chat — Integration & Instructions

Tài liệu này mô tả nghiệp vụ, API contract và ví dụ tích hợp frontend/backend cho hệ thống Chatbot phân loại bug report (dựa trên nội dung `README.md` repository).

**Mục tiêu ngắn gọn**
- Người dùng gửi bug report (text) hoặc upload file (CSV / Excel) chứa nhiều báo cáo.
- Hệ thống trả về nhãn phân loại (UI, Performance, Security, Functional, Data, ...) và (tùy chọn) gợi ý khắc phục hoặc action.

## 1. Nghiệp vụ & nhãn phân loại

- Mỗi bug report được phân vào một trong các nhãn chính:
  - `UI` — vấn đề giao diện, hiển thị, bố cục, màu sắc
  - `Performance` — chậm, timeout, memory/cpu
  - `Security` — bảo mật, injection, phân quyền
  - `Functional` — logic, xử lý sai, hành vi không đúng
  - `Data` — dữ liệu test/đầu vào sai, file upload không đúng định dạng

## 2. Luồng hoạt động đề xuất

1. Frontend: Giao diện chat (React). Người dùng có thể nhập text hoặc upload file CSV/XLSX. Frontend gửi request tới backend.
2. Backend: nhận payload (text hoặc file), chuẩn hoá nội dung (nối văn bản, trích preview từ bảng), gọi mô hình AI (OpenAI) với `system` prompt mô tả nhiệm vụ phân loại.
3. Backend trả về JSON chứa nhãn (label), confidence (nếu có), và optional `suggestion` (text hướng xử lý).
4. Frontend hiển thị kết quả trong chat và lưu lịch sử cuộc hội thoại.

## 3. API Contract (REST)

- POST `/api/classify`
  - Content-Type: `multipart/form-data` hoặc `application/json`
  - Body (JSON):
    - `type`: "text" | "file"
    - `text`: (nếu `type` === "text") nội dung bug report
    - `file`: (multipart file) CSV / XLSX (nếu `type` === "file")
    - `conversationId` (optional): id cuộc hội thoại để gắn kết
  - Response (JSON):
    - `label`: string (ví dụ: "UI")
    - `confidence`: number (0-1) — nếu backend tính được
    - `suggestion`: string — hướng khắc phục/notes
    - `sourcePreview`: string — preview trích từ file (nếu upload)

Example response:

```json
{
  "label": "UI",
  "confidence": 0.92,
  "suggestion": "Kiểm tra component X CSS, validator, và hành vi responsive",
  "sourcePreview": "Row1,...\nRow2,..."
}
```

## 4. Ví dụ Backend (Python Flask + OpenAI)

Gợi ý triển khai nhanh bằng Flask — xử lý text và file, dùng `pandas` để đọc Excel/CSV, gọi OpenAI.

```python
# server/app.py
from flask import Flask, request, jsonify
import os
import openai
import pandas as pd

openai.api_key = os.getenv('OPENAI_API_KEY')
app = Flask(__name__)

SYSTEM_PROMPT = '''You are an expert bug triage assistant. Given a bug report text or file content, choose the best one-label classification from: UI, Performance, Security, Functional, Data. Provide a short suggestion for fixing. Output JSON only with keys: label, suggestion.'''

def classify_text(text):
    prompt = SYSTEM_PROMPT + "\n\nUser report:\n" + text
    resp = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        max_tokens=300,
        temperature=0.0
    )
    content = resp['choices'][0]['message']['content']
    # parse result (expect JSON or simple text). In production, robust-parse here.
    return content

@app.route('/api/classify', methods=['POST'])
def classify():
    if 'file' in request.files:
        f = request.files['file']
        try:
            df = pd.read_csv(f) if f.filename.lower().endswith('.csv') else pd.read_excel(f)
            preview = df.head(5).to_csv(index=False)
        except Exception as e:
            return jsonify({'error': 'file parse error', 'detail': str(e)}), 400
        text = preview
    else:
        payload = request.get_json() or {}
        text = payload.get('text', '')

    if not text:
        return jsonify({'error': 'empty input'}), 400

    result = classify_text(text)
    return jsonify({'raw': result})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
```

Notes:
- Ở ví dụ trên, `classify_text` gọi OpenAI. Bạn nên parse kết quả trả về sang JSON (hoặc yêu cầu model trả JSON). Đặt `temperature=0` để kết quả ổn định.
- Thêm rate-limit, auth và validate file size trước khi deploy.

## 5. Ví dụ Backend (Node.js / Express)

```js
// server/index.js (sketch)
import express from 'express'
import multer from 'multer'
import OpenAI from 'openai'
import csv from 'csv-parser'
import fs from 'fs'

const app = express()
const upload = multer({ dest: 'uploads/' })
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

app.post('/api/classify', upload.single('file'), async (req, res) => {
  try {
    let text = ''
    if (req.file) {
      // parse CSV or Excel — here we show CSV
      const rows = []
      fs.createReadStream(req.file.path).pipe(csv()).on('data', (data) => rows.push(data)).on('end', async () => {
        text = JSON.stringify(rows.slice(0, 5))
        const response = await client.chat.completions.create({ model: 'gpt-4o-mini', messages: [{role:'system',content: '...'}, {role:'user', content: text}] })
        res.json({ raw: response })
      })
    } else {
      text = req.body.text || ''
      const response = await client.chat.completions.create({ model: 'gpt-4o-mini', messages: [{role:'system',content: '...'}, {role:'user', content: text}] })
      res.json({ raw: response })
    }
  } catch (err) {
    console.error(err)
    res.status(500).json({ error: err.message })
  }
})

app.listen(8000)
```

## 6. Frontend integration (React)

- Use existing `ChatWindow` to upload files: it already uses `papaparse` and `xlsx` to create a preview message client-side. To actually classify on server, send `fetch` to the backend endpoint.

Example `fetch` call (send text):

```js
async function classifyText(text) {
  const res = await fetch('/api/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type: 'text', text })
  })
  return res.json()
}
```

Example to upload file (multipart):

```js
async function uploadFile(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch('/api/classify', { method: 'POST', body: fd })
  return res.json()
}
```

Integration notes:
- Khi nhận được response, append một message assistant chứa `label` + `suggestion` vào conversation.
- Lưu `conversationId` để theo dõi lịch sử.

## 7. System prompt mẫu (kiểu JSON response)

Sử dụng một `system` prompt rõ ràng giúp model trả về định dạng dễ parse. Ví dụ:

```
You are an expert bug triage assistant. Given a bug report or table rows, output a JSON object with keys: label, confidence, suggestion. label must be one of: UI, Performance, Security, Functional, Data. confidence is a number between 0 and 1. suggestion is short actionable advice.

Respond with JSON only, no extra text.
```

Ví dụ user message (đưa cho model):

```
User report: Khi bấm nút Submit không hiện thông báo xác nhận. Các bước: ...
```

## 8. Chạy & triển khai

- Frontend:
  - cd `Web`
  - `npm install`
  - `npm run dev` (dev) hoặc `npm run build` + serve `dist` cho production
- Backend:
  - Python: `pip install flask openai pandas openpyxl` rồi `python app.py`
  - Node: `npm install express multer openai csv-parser xlsx` rồi `node index.js`

## 9. Bảo mật & vận hành

- Không lưu API keys trong frontend; backend phải gọi OpenAI.
- Thêm giới hạn kích thước file (ví dụ 5MB) và kiểm tra mime-type.
- Rate-limit và authentication cho endpoint `/api/classify`.

## 10. Nâng cao (gợi ý)



Nếu bạn muốn, tôi có thể:

Also included in this repo:

- `src/assets/logo.svg` — primary badge logo (gradient chat-bubble + bug + checkmark). Good for landing pages and header branding.
- `src/assets/logo-rounded.svg` — compact circular icon for avatars, favicons and app icons.

Logo usage notes:

- Export sizes: 512×512, 256×256, 128×128 for icons; 1200×630 (social preview) for badge.
- Color palette: gradient from `#7b61ff` → `#2b6ef6` (same as the UI accent palette).
- File formats: keep SVG for web; export PNG/WebP for older clients; provide favicon.ico if needed.

If you'd like different concepts (outline, flat, monochrome), tell me which style and I'll add alternates.
Hãy chọn bước tiếp theo bạn muốn tôi thực hiện.
