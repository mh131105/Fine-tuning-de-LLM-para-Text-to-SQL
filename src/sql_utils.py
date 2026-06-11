from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DESTRUCTIVE_TOKENS = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "UPDATE",
    "VACUUM",
}


@dataclass(frozen=True)
class QueryResult:
    success: bool
    rows: list[tuple[Any, ...]]
    error_type: str | None = None
    error_message: str | None = None
    sql: str | None = None


def _without_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql.strip()


def is_safe_select_query(sql: str) -> bool:
    cleaned = _without_comments(sql)
    if not cleaned:
        return False
    statements = [part.strip() for part in cleaned.split(";") if part.strip()]
    if len(statements) != 1:
        return False
    first = statements[0].lstrip(" \n\t(").upper()
    if not (first.startswith("SELECT") or first.startswith("WITH")):
        return False
    tokens = set(re.findall(r"\b[A-Z_]+\b", cleaned.upper()))
    return DESTRUCTIVE_TOKENS.isdisjoint(tokens)


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql or "", flags=re.IGNORECASE))


def execute_sql(db_path: str | Path, sql: str, timeout_seconds: int = 5) -> QueryResult:
    if not is_safe_select_query(sql):
        return QueryResult(False, [], "unsafe_sql", "Only read-only SELECT/WITH queries are allowed", sql)

    path = Path(db_path)
    if not path.exists():
        return QueryResult(False, [], "sqlite_connection_error", f"SQLite database not found: {path}", sql)

    started = time.monotonic()
    connection: sqlite3.Connection | None = None
    try:
        uri = f"file:{path.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
        connection.text_factory = bytes
        connection.execute("PRAGMA query_only = ON")

        def progress_handler() -> int:
            return 1 if time.monotonic() - started > timeout_seconds else 0

        connection.set_progress_handler(progress_handler, 1000)
        cursor = connection.execute(sql)
        rows = cursor.fetchall()
        return QueryResult(True, rows, sql=sql)
    except sqlite3.OperationalError as exc:
        message = str(exc)
        error_type = "timeout" if "interrupted" in message.lower() else "execution_error"
        return QueryResult(False, [], error_type, message, sql)
    except sqlite3.Error as exc:
        return QueryResult(False, [], "execution_error", str(exc), sql)
    finally:
        if connection is not None:
            connection.close()


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_row(row: tuple[Any, ...] | list[Any] | Any) -> tuple[Any, ...]:
    if not isinstance(row, (tuple, list)):
        row = (row,)
    return tuple(_normalize_cell(value) for value in row)


def normalize_result(rows: list[tuple[Any, ...]], preserve_order: bool) -> list[tuple[Any, ...]]:
    normalized = [_normalize_row(row) for row in rows]
    if preserve_order:
        return normalized
    return sorted(normalized, key=repr)


def compare_sql_results(
    predicted: QueryResult,
    gold: QueryResult,
    preserve_order: bool | None = None,
) -> bool:
    if not predicted.success or not gold.success:
        return False
    if preserve_order is None:
        preserve_order = has_order_by(predicted.sql or "") or has_order_by(gold.sql or "")
    return normalize_result(predicted.rows, preserve_order) == normalize_result(gold.rows, preserve_order)
