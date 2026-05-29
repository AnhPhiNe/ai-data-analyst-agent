from dataclasses import dataclass
import re


SQL_TABLE_NAME = "dataset"
DEFAULT_SQL_LIMIT = 100
MAX_SQL_LIMIT = 100

BLOCKED_SQL_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "COPY",
    "ATTACH",
    "DETACH",
    "INSTALL",
    "LOAD",
    "EXPORT",
    "IMPORT",
    "PRAGMA",
    "SET",
    "CALL",
}

BLOCKED_SQL_PATTERNS = (
    "read_csv",
    "read_parquet",
    "read_json",
    "read_excel",
    "httpfs",
    "s3",
    "file:",
    "glob",
)


@dataclass(frozen=True)
class ValidatedSql:
    sql: str
    limit: int
    executable_sql: str


def validate_read_only_sql(
    sql: object, limit: object = DEFAULT_SQL_LIMIT
) -> ValidatedSql:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("'sql' must be a non-empty string.")

    normalized_sql = _strip_single_trailing_semicolon(sql.strip())
    if ";" in normalized_sql:
        raise ValueError("Only one SQL statement is allowed.")

    normalized_for_checks = _collapse_whitespace(normalized_sql)
    upper_sql = normalized_for_checks.upper()
    blocked_keyword = _find_blocked_keyword(upper_sql)
    if blocked_keyword is not None:
        raise ValueError(f"SQL keyword '{blocked_keyword}' is not allowed.")
    if not (upper_sql.startswith("SELECT ") or upper_sql.startswith("WITH ")):
        raise ValueError("Only SELECT or WITH ... SELECT queries are allowed.")

    lower_sql = normalized_for_checks.lower()
    for pattern in BLOCKED_SQL_PATTERNS:
        if pattern in lower_sql:
            raise ValueError(f"SQL pattern '{pattern}' is not allowed.")

    _validate_table_references(normalized_for_checks)
    bounded_limit = _bounded_limit(limit)
    return ValidatedSql(
        sql=normalized_sql,
        limit=bounded_limit,
        executable_sql=(
            f"SELECT * FROM ({normalized_sql}) AS result LIMIT {bounded_limit}"
        ),
    )


def _strip_single_trailing_semicolon(sql: str) -> str:
    stripped = sql.rstrip()
    if stripped.endswith(";"):
        return stripped[:-1].rstrip()
    return stripped


def _collapse_whitespace(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def _find_blocked_keyword(upper_sql: str) -> str | None:
    for keyword in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", upper_sql):
            return keyword
    return None


def _validate_table_references(sql: str) -> None:
    cte_names = {
        match.group(1).lower()
        for match in re.finditer(r"\bWITH\s+([A-Za-z_][\w]*)\s+AS\b", sql, re.I)
    }
    cte_names.update(
        match.group(1).lower()
        for match in re.finditer(r",\s*([A-Za-z_][\w]*)\s+AS\s*\(", sql, re.I)
    )
    table_refs = [
        match.group(2).strip('"').strip("`").lower()
        for match in re.finditer(
            r"\b(FROM|JOIN)\s+([A-Za-z_][\w]*|\"[^\"]+\"|`[^`]+`)",
            sql,
            re.I,
        )
    ]
    if not table_refs:
        raise ValueError("SQL must query the dataset table.")
    for table_name in table_refs:
        if table_name != SQL_TABLE_NAME and table_name not in cte_names:
            raise ValueError("SQL can only query the dataset table.")


def _bounded_limit(limit: object) -> int:
    if limit is None:
        return DEFAULT_SQL_LIMIT
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("'limit' must be an integer.")
    if limit < 1 or limit > MAX_SQL_LIMIT:
        raise ValueError(f"'limit' must be between 1 and {MAX_SQL_LIMIT}.")
    return limit
