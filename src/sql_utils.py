import re
import sqlite3
import multiprocessing

def extract_sql(raw_output: str) -> str:
    # Remove markdown formatting
    if "```sql" in raw_output:
        parts = raw_output.split("```sql")
        if len(parts) > 1:
            sql_part = parts[1].split("```")[0]
            return sql_part.strip()
    if "```" in raw_output:
        parts = raw_output.split("```")
        if len(parts) > 1:
            sql_part = parts[1]
            return sql_part.strip()
            
    # Fallback to prefix removal
    lines = raw_output.split('\n')
    for line in lines:
        upper = line.upper()
        if upper.startswith("SQL:"):
            return line[4:].strip()
        if upper.startswith("SELECT"):
            return line.strip()
            
    # Try to find SELECT ...
    idx = raw_output.upper().find("SELECT")
    if idx != -1:
        return raw_output[idx:].strip()
        
    return raw_output.strip()

def _execute_sql(db_path, sql, result_queue):
    try:
        uri_path = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri_path, uri=True)
        conn.execute("PRAGMA query_only = ON;")
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        result_queue.put(("success", res))
    except Exception as e:
        result_queue.put(("error", str(e)))

def execute_sql_readonly(db_path: str, sql: str, timeout: int = 10):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_execute_sql, args=(db_path, sql, queue))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return "error", "SQL_TIMEOUT"
        
    if not queue.empty():
        return queue.get()
    return "error", "UNKNOWN_ERROR"

def compare_sql_results(res1, res2, sql1: str, sql2: str) -> bool:
    if not isinstance(res1, list) or not isinstance(res2, list):
        return res1 == res2
        
    def normalize(val):
        if isinstance(val, tuple) and len(val) == 1:
            val = val[0]
        if isinstance(val, float):
            return round(val, 3)
        if isinstance(val, str):
            return val.lower().strip()
        return val

    norm_res1 = [normalize(row) for row in res1]
    norm_res2 = [normalize(row) for row in res2]
    
    has_order_by1 = "ORDER BY" in sql1.upper()
    has_order_by2 = "ORDER BY" in sql2.upper()
    
    if has_order_by1 or has_order_by2:
        return norm_res1 == norm_res2
    else:
        # Multiset comparison without caring about order
        # Sort both lists
        try:
            return sorted(norm_res1) == sorted(norm_res2)
        except Exception: # In case of unorderable types
            return set(tuple(x) if isinstance(x, list) else x for x in norm_res1) == set(tuple(x) if isinstance(x, list) else x for x in norm_res2)
