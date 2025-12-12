"""
Bug classification labels configuration
Định nghĩa các nhãn bug, keywords và severity hints
"""

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

# Team groups mapping (3 team, mỗi team 2 label)
TEAM_GROUPS = {
    "Frontend": [
        "UI", "UX", "Accessibility", "Frontend", "Compatibility", "Localization"
    ],
    "Backend": [
        "Backend", "Database", "Integration", "Functional", "Data", "Security", "Privacy"
    ],
    "Infra": [
        "Network", "Performance", "Memory", "Crash", "Observability", "Build", "Config"
    ]
}

# Build inverse mapping: label -> team
LABEL_TO_TEAM = {}
for team, labels in TEAM_GROUPS.items():
    for label in labels:
        LABEL_TO_TEAM[label] = team

# Build inverse mapping: label -> team
LABEL_TO_TEAM = {}
for team, labels in TEAM_GROUPS.items():
    for label in labels:
        LABEL_TO_TEAM[label] = team
