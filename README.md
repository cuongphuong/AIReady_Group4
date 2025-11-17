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

## 3. Định hướng và mở rộng
- Có UI dạng chatbot để dễ dàng sử dụng
- Khả năng xử lý hàng loạt bằng cách input file
- Đối chiếu dữ liệu đã phân tích với các nhóm đã phân loại trước đó tránh sảy ra sai sót
- Đưa ra hướng dẩn về cách khắc phục và action tránh lặp lại
---

## 4. Hướng Dẫn Cài Đặt, Khởi Động Console

### Bước 1: Chuẩn bị môi trường

- **Cài đặt Python**  
  Đảm bảo máy đã cài Python >= 3.8

- **Tải source code**
  Clone file source code `bug_classifier.py `

- **Cài đặt thư viện OpenAI,  `python-dotenv`.**  
  Mở terminal/cmd và chạy lệnh:
  ```bash
  pip install openai python-dotenv
  ```

- **Đăng ký API Key OpenAI**  
  Tạo tài khoản tại [OpenAI](https://platform.openai.com/) & lấy key, lưu vào biến môi trường, **.env**
  ```bash
  OPENAI_API_KEY=your_openai_key_here
  ```

### Bước 2: Khởi chạy ứng dụng

- Mở terminal/cmd, di chuyển đến thư mục chứa file **bug_classifier.py**.
- Chạy lệnh:
  ```bash
  python bug_classifier.py
  ```
- Nhập nội dung bug report khi được yêu cầu, ví dụ:
  ```
  Nhập nội dung bug report: Khi bấm nút Submit không hiện thông báo xác nhận.
  ```

- **Kết quả nhận được:**
  ```
  Bug report: Khi bấm nút Submit không hiện thông báo xác nhận.
  Phân loại: UI
  ```
---