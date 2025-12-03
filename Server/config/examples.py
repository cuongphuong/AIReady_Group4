"""
Few-shot examples for bug classification
Ví dụ mẫu để cải thiện độ chính xác của model
"""

FEW_SHOT_EXAMPLES = [
    {
        "description": "Trên Chrome desktop, nút 'Gửi' trong form phản hồi bị dịch xuống dưới và bị che một phần bởi footer; theo thiết kế nó phải nằm trong cùng hàng với ô nhập.",
        "label": "UI"
    },
    {
        "description": "Luồng đăng ký yêu cầu nhập quá nhiều thông tin ngay từ bước đầu, người dùng dễ bỏ dở; không có hướng dẫn rõ ràng cho trường bắt buộc.",
        "label": "UX"
    },
    {
        "description": "Các nhãn input thiếu thuộc tính aria-label; tab order nhảy không tuần tự, màn hình đọc (screen reader) không đọc được nhãn của checkbox.",
        "label": "Accessibility"
    },
    {
        "description": "Console báo lỗi 'TypeError: Cannot read property x of undefined' khi mở trang báo cáo, component không render được dữ liệu trả về API.",
        "label": "Frontend"
    },
    {
        "description": "API /orders trả mã lỗi 500 khi gửi payload gồm order_items rỗng; stacktrace chỉ ra lỗi NullPointer trong service xử lý order.",
        "label": "Backend"
    },
    {
        "description": "Query báo cáo sales trên production mất 45s, explain plan cho thấy full table scan do thiếu index trên cột created_at.",
        "label": "Database"
    },
    {
        "description": "Service A không thể kết nối tới Service B (DNS lookup failed) trong giờ cao điểm, gây timeout liên tục.",
        "label": "Network"
    },
    {
        "description": "Trang dashboard khởi tạo chậm (TTFB > 2s) và chuyển tab trong SPA bị lag khi có >100 items hiển thị.",
        "label": "Performance"
    },
    {
        "description": "Sau vài ngày chạy, process backend tăng dần memory usage và cuối cùng OOM kill; không có dấu hiệu GC giải phóng đúng.",
        "label": "Memory"
    },
    {
        "description": "Phát hiện header Authorization bị echo lại trong response body của một endpoint, có khả năng rò rỉ token cho client.",
        "label": "Security"
    },
    {
        "description": "Tính năng tính hoa hồng trả về kết quả sai: tổng không khớp với dữ liệu nguồn, vì lọc điều kiện áp dụng sai trong code.",
        "label": "Functional"
    },
    {
        "description": "Người dùng upload file CSV, hệ thống không nhận format vì header khác tên; file không có validation nên chập dữ liệu vào DB.",
        "label": "Data"
    },
    {
        "description": "Webhook từ payment provider trả về HTTP 400 sau một thay đổi phiên bản API của họ, dẫn đến giao dịch không được xác nhận.",
        "label": "Integration"
    },
    {
        "description": "Layout và một số icon bị vỡ khi mở trên Safari iOS 13, nhưng hoạt bình thường trên các trình duyệt hiện đại.",
        "label": "Compatibility"
    },
    {
        "description": "Ngày hiển thị trên báo cáo ở môi trường VN đang theo định dạng MM/DD thay vì DD/MM, gây nhầm lẫn cho người dùng.",
        "label": "Localization"
    },
    {
        "description": "Ứng dụng mobile crash ngay sau khi nhấn nút 'Tải dữ liệu', stacktrace chỉ rõ NullReference trong module sync.",
        "label": "Crash"
    },
    {
        "description": "Thiếu logs cho luồng xử lý thanh toán; khi có thất bại không có trace id để điều tra, alert không được trigger.",
        "label": "Observability"
    },
    {
        "description": "Pipeline CI bị lỗi step build khi cập nhật dependency X; artifact không được publish đến registry, blocking release.",
        "label": "Build"
    },
    {
        "description": "Một endpoint công khai trả về danh sách user bao gồm email và phone number; rủi ro rò rỉ PII.",
        "label": "Privacy"
    },
    {
        "description": "Sau deploy, service không khởi động do thiếu biến môi trường REQUIRED_API_KEY trong config, gây downtime.",
        "label": "Config"
    }
]
