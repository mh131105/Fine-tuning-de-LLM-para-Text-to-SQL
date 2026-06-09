from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sql_utils import execute_sql_readonly, compare_sql_results

class ExecutionAccuracyMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = None
        self.reason = None
        self.latency_seconds = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        import time
        start_time = time.time()
        
        predicted_sql = test_case.actual_output
        gold_sql = test_case.expected_output
        db_path = test_case.metadata.get("db_path") if hasattr(test_case, "metadata") and test_case.metadata else test_case.additional_metadata.get("db_path") if hasattr(test_case, "additional_metadata") else None
        
        if not db_path or not os.path.exists(db_path):
            self.score = 0.0
            self.reason = "DB_NOT_FOUND"
            return self.score
            
        status_pred, res_pred = execute_sql_readonly(db_path, predicted_sql)
        status_gold, res_gold = execute_sql_readonly(db_path, gold_sql)
        
        self.latency_seconds = time.time() - start_time
        
        if status_pred == "error":
            self.score = 0.0
            err_str = res_pred.lower()
            if "syntax" in err_str or "incomplete" in err_str:
                self.reason = "SQL_SYNTAX_ERROR"
            elif "timeout" in err_str:
                self.reason = "SQL_TIMEOUT"
            else:
                self.reason = "SQL_EXECUTION_ERROR"
            return self.score
            
        if status_gold == "error":
            # Gold failed?
            self.score = 0.0
            self.reason = "GOLD_SQL_ERROR"
            return self.score
            
        match = compare_sql_results(res_pred, res_gold, predicted_sql, gold_sql)
        
        if match:
            self.score = 1.0
            self.reason = "SUCCESS"
        else:
            self.score = 0.0
            self.reason = "RESULT_MISMATCH"
            
        return self.score

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold

    @property
    def __name__(self):
        return "Execution Accuracy"
