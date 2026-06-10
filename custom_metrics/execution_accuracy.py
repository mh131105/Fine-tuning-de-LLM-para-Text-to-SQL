from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    from deepeval.metrics import BaseMetric
except Exception:

    class BaseMetric:  # type: ignore[no-redef]
        threshold: float
        score: float | None
        reason: str | None


from src.tp2.inference import extract_sql
from src.tp2.sql_utils import compare_sql_results, execute_sql, has_order_by, is_safe_select_query


class ExecutionAccuracy(BaseMetric):
    """DeepEval-compatible execution accuracy for Spider Text-to-SQL."""

    def __init__(self, spider_db_dir: str | Path | None = None, threshold: float = 1.0, timeout_seconds: int = 5):
        self.spider_db_dir = Path(spider_db_dir) if spider_db_dir else None
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self.score: float | None = None
        self.reason: str | None = None
        self.error_type: str | None = None
        self.error_message: str | None = None
        self.predicted_sql: str | None = None
        self.latency_seconds = 0.0

    @property
    def __name__(self) -> str:
        return "ExecutionAccuracy"

    def _metadata(self, test_case: Any) -> dict[str, Any]:
        for attr in ("additional_metadata", "metadata"):
            value = getattr(test_case, attr, None)
            if isinstance(value, dict):
                return value
        return {}

    def _db_path(self, metadata: dict[str, Any]) -> Path | None:
        if metadata.get("db_path"):
            return Path(metadata["db_path"])
        db_id = metadata.get("db_id")
        if db_id and self.spider_db_dir:
            return self.spider_db_dir / db_id / f"{db_id}.sqlite"
        return None

    def _finish(self, score: float, reason: str, error_type: str | None = None, error_message: str | None = None) -> float:
        self.score = float(score)
        self.reason = reason
        self.error_type = error_type
        self.error_message = error_message
        return self.score

    def measure(self, test_case: Any) -> float:
        started = time.perf_counter()
        self.error_type = None
        self.error_message = None
        self.predicted_sql = None

        metadata = self._metadata(test_case)
        db_path = self._db_path(metadata)
        if db_path is None:
            self.latency_seconds = time.perf_counter() - started
            return self._finish(0.0, "DB_PATH_MISSING", "sqlite_connection_error", "No db_path or db_id provided")

        raw_prediction = str(getattr(test_case, "actual_output", "") or "")
        predicted_sql = extract_sql(raw_prediction)
        self.predicted_sql = predicted_sql
        if not predicted_sql:
            self.latency_seconds = time.perf_counter() - started
            return self._finish(0.0, "SQL_EXTRACTION_ERROR", "sql_extraction_error", "No SQL found in model output")
        if not is_safe_select_query(predicted_sql):
            self.latency_seconds = time.perf_counter() - started
            return self._finish(0.0, "UNSAFE_SQL", "unsafe_sql", "Predicted SQL is not a safe SELECT/WITH query")

        gold_sql = str(getattr(test_case, "expected_output", "") or "").strip()
        predicted = execute_sql(db_path, predicted_sql, timeout_seconds=self.timeout_seconds)
        gold = execute_sql(db_path, gold_sql, timeout_seconds=self.timeout_seconds)
        self.latency_seconds = time.perf_counter() - started

        if not predicted.success:
            return self._finish(0.0, "SQL_EXECUTION_ERROR", predicted.error_type, predicted.error_message)
        if not gold.success:
            return self._finish(0.0, "GOLD_SQL_ERROR", "gold_execution_error", gold.error_message)

        preserve_order = has_order_by(predicted_sql) or has_order_by(gold_sql)
        if compare_sql_results(predicted, gold, preserve_order=preserve_order):
            return self._finish(1.0, "SUCCESS")
        return self._finish(0.0, "RESULT_MISMATCH", "result_mismatch", "Result sets differ")

    async def a_measure(self, test_case: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold


ExecutionAccuracyMetric = ExecutionAccuracy
