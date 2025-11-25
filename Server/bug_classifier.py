import os
from openai import OpenAI
from dotenv import load_dotenv
import openai

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE_URL")

# 1. Khởi tạo client với API key
client = openai.OpenAI(
  base_url=base_url,
  api_key=api_key
)

# 2. Danh sách nhãn bug
BUG_LABELS = {
  "UI": "Kiểm tra về mặt hiển thị, căn lề, màu sắc của item",
  "Performance": "Hiệu suất của ứng dụng, thời gian xử lý, thời gian load ứng dụng",
  "Security": "Bảo mật dữ liệu, SQL injection, phân quyền truy cập",
  "Functional": "Logic xử lý không đúng, điều kiện xử lý không đúng, đọc ghi dữ liệu sai điều kiện, call API không đúng, validate sai item",
  "Data": "Dữ liệu test không hợp lệ, dữ liệu input vào màn hình không hợp lệ, file data upload không đúng"
}

# 3. Ví dụ mẫu cho few-shot
FEW_SHOT_EXAMPLES = [
    # UI
    {
        "description": "Nút đăng nhập bị lệch sang phải so với form, màu nền không đúng thiết kế.",
        "label": "UI"
    },
    # Performance
    {
        "description": "Trang dashboard mất 10 giây để load dữ liệu, thao tác chuyển tab bị giật.",
        "label": "Performance"
    },
    # Security
    {
        "description": "Người dùng không có quyền vẫn truy cập được màn hình quản trị.",
        "label": "Security"
    },
    # Functional
    {
        "description": "Khi nhập số âm vào trường giá trị, hệ thống vẫn cho phép lưu dữ liệu.",
        "label": "Functional"
    },
    # Data
    {
        "description": "File CSV upload bị lỗi định dạng, hệ thống không cảnh báo.",
        "label": "Data"
    }
]

# 4. Hàm phân loại bug
def classify_bug(description):
  # Tạo mô tả tổng hợp các nhãn
  label_descriptions = "\n".join(
      [f"- {label}: {desc}" for label, desc in BUG_LABELS.items()]
  )

  # Tạo ví dụ mẫu cho prompt
  example_text = "\n".join([
      f"Bug report: \"{ex['description']}\"\nPhân loại: {ex['label']}"
      for ex in FEW_SHOT_EXAMPLES
  ])

  prompt = f"""
  Bạn là một chuyên gia QA. Dưới đây là mô tả các nhóm bug:
  {label_descriptions}

  Các ví dụ phân loại bug report:
  {example_text}

  Hãy phân loại bug dưới đây vào một trong các nhóm trên.
  Bug report:
  \"\"\"{description}\"\"\"
  Trả về chỉ một nhãn phù hợp nhất (UI, Performance, Security, Functional hoặc Data).
  """
  response = client.chat.completions.create(
      model="gpt-5",
      messages=[
          {"role": "system", "content": "Bạn là một chuyên gia phân loại bug."},
          {"role": "user", "content": prompt}
      ],
      temperature=1
  )
  return response.choices[0].message.content.strip()

# 5. Lấy input từ console
bug_report = input("Nhập nội dung bug report: ")

# 6. Phân loại và hiển thị kết quả
label = classify_bug(bug_report)
print(f"\nBug report: {bug_report}\nPhân loại: {label}")
input(".")