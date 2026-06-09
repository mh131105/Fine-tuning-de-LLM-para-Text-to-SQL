import json
import os

def load_tables_json(tables_json_path: str) -> dict:
    if not os.path.exists(tables_json_path):
        return {}
        
    with open(tables_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # create a lookup dictionary by db_id
    schema_map = {}
    for db in data:
        db_id = db['db_id']
        table_names = db['table_names_original']
        column_names = db['column_names_original']
        column_types = db['column_types']
        primary_keys = db['primary_keys']
        foreign_keys = db['foreign_keys']
        
        tables = {}
        for i, t_name in enumerate(table_names):
            tables[i] = {'name': t_name, 'columns': [], 'primary_keys': [], 'foreign_keys': []}
            
        for col_idx, (tab_idx, col_name) in enumerate(column_names):
            if tab_idx == -1: # * column
                continue
            col_type = column_types[col_idx]
            is_pk = col_idx in primary_keys
            tables[tab_idx]['columns'].append({
                'id': col_idx,
                'name': col_name,
                'type': col_type,
                'is_pk': is_pk
            })
            if is_pk:
                tables[tab_idx]['primary_keys'].append(col_name)
                
        for fk_col, pk_col in foreign_keys:
            fk_col_name = column_names[fk_col][1]
            fk_tab_idx = column_names[fk_col][0]
            pk_col_name = column_names[pk_col][1]
            pk_tab_idx = column_names[pk_col][0]
            if fk_tab_idx != -1 and pk_tab_idx != -1:
                tables[fk_tab_idx]['foreign_keys'].append({
                    'column': fk_col_name,
                    'references_table': table_names[pk_tab_idx],
                    'references_column': pk_col_name
                })
                
        schema_map[db_id] = tables
    return schema_map

def serialize_schema(schema_map: dict, db_id: str) -> str:
    if db_id not in schema_map:
        return ""
    tables = schema_map[db_id]
    schema_lines = []
    
    # Sort for deterministic output
    for tab_idx in sorted(tables.keys()):
        t = tables[tab_idx]
        col_strs = []
        for c in t['columns']:
            pk_str = " (PRIMARY KEY)" if c['is_pk'] else ""
            col_strs.append(f"{c['name']} {c['type'].upper()}{pk_str}")
        
        # Foreign keys
        fk_strs = []
        for fk in t['foreign_keys']:
            fk_strs.append(f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['references_table']}({fk['references_column']})")
            
        schema_lines.append(f"CREATE TABLE {t['name']} (")
        all_cols = col_strs + fk_strs
        schema_lines.append("  " + ",\n  ".join(all_cols))
        schema_lines.append(");")
        
    return "\n".join(schema_lines)
