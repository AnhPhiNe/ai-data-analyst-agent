from dataclasses import dataclass
from enum import StrEnum
import re
from backend.agent.column_resolver import normalize_text


class GuardrailCategory(StrEnum):
    ALLOWED = "allowed"
    CODE_EXECUTION = "code_execution"
    FILE_SYSTEM = "file_system"
    SECRETS = "secrets"
    INTERNET = "internet"
    DESTRUCTIVE_ACTION = "destructive_action"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class GuardrailResult:
    is_allowed: bool
    category: GuardrailCategory
    message: str


BLOCKED_PATTERNS: list[tuple[GuardrailCategory, tuple[str, ...], str]] = [
    (
        GuardrailCategory.CODE_EXECUTION,
        (
            "chay code",
            "chạy code",
            "execute code",
            "run python",
            "python code",
            "eval(",
            "exec(",
            "subprocess",
            "os.system",
            "shell command",
            "lenh shell",
            "lệnh shell",
        ),
        "Mình không thể chạy code hoặc lệnh tùy ý. Mình chỉ phân tích dữ liệu qua các tool pandas đã whitelist.",
    ),
    (
        GuardrailCategory.FILE_SYSTEM,
        (
            "doc file he thong",
            "đọc file hệ thống",
            "read system file",
            "/etc/passwd",
            "c:\\",
            "windows\\system32",
            "mo file",
            "mở file",
            "read file",
        ),
        "Mình không thể đọc file hệ thống. Hãy upload CSV/XLSX để mình phân tích trong phiên hiện tại.",
    ),
    (
        GuardrailCategory.SECRETS,
        (
            "api key",
            "apikey",
            "secret",
            "password",
            "mat khau",
            "mật khẩu",
            ".env",
            "environment variable",
            "bien moi truong",
            "biến môi trường",
        ),
        "Mình không thể truy cập hoặc tiết lộ API key, secret, mật khẩu hay biến môi trường.",
    ),
    (
        GuardrailCategory.INTERNET,
        (
            "internet",
            "web search",
            "search web",
            "google",
            "truy cap web",
            "truy cập web",
            "goi api ngoai",
            "gọi api ngoài",
            "external api",
        ),
        "MVP này không truy cập internet hoặc API bên ngoài; mình chỉ phân tích dataset đã upload.",
    ),
    (
        GuardrailCategory.DESTRUCTIVE_ACTION,
        (
            "xoa file",
            "xóa file",
            "delete file",
            "remove file",
            "sua file",
            "sửa file",
            "overwrite",
            "drop table",
            "format disk",
        ),
        "Mình không thể xóa hoặc sửa file/hệ thống. Phạm vi hiện tại chỉ là phân tích dataset trong memory.",
    ),
    (
        GuardrailCategory.OUT_OF_SCOPE,
        (
            "thoi tiet",
            "thời tiết",
            "gia vang",
            "giá vàng",
            "bitcoin",
            "tin tuc",
            "tin tức",
            "lich bong da",
            "lịch bóng đá",
        ),
        "Câu hỏi này nằm ngoài phạm vi dataset đã upload. Hãy hỏi về dữ liệu dạng bảng trong phiên hiện tại.",
    ),
]


def check_guardrails(question: str) -> GuardrailResult:
    # 1. Raw signature check for special strings (eval(, exec(, path prefixes, etc.)
    raw_lower = question.lower()
    for category, patterns, message in BLOCKED_PATTERNS:
        for pattern in patterns:
            # If pattern contains special characters, check raw substring
            if any(char in pattern for char in ("(", "/", "\\", ".")):
                if pattern in raw_lower:
                    return GuardrailResult(
                        is_allowed=False, category=category, message=message
                    )

    # 2. Normalized diacritics-safe whole-word boundary check
    normalized = normalize_text(question)
    if not normalized:
        return GuardrailResult(
            is_allowed=False,
            category=GuardrailCategory.OUT_OF_SCOPE,
            message="Vui lòng nhập một câu hỏi về dataset đã upload.",
        )

    for category, patterns, message in BLOCKED_PATTERNS:
        for pattern in patterns:
            norm_pattern = normalize_text(pattern)
            if not norm_pattern:
                continue
            if re.search(rf"(?<!\w){re.escape(norm_pattern)}(?!\w)", normalized):
                return GuardrailResult(
                    is_allowed=False, category=category, message=message
                )

    return GuardrailResult(
        is_allowed=True,
        category=GuardrailCategory.ALLOWED,
        message="Request is allowed.",
    )
