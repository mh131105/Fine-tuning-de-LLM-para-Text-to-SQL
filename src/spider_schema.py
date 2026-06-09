def serialize_schema(schema_dict: dict) -> str:
    """Serializa o schema da nova arquitetura de dados (data.py) para string DDL (CREATE TABLE)."""
    lines = []
    
    for t in schema_dict.get('tables', []):
        cols = []
        for c in t.get('columns', []):
            pk_str = " (PRIMARY KEY)" if c.get('is_primary_key') else ""
            c_type = c.get('type') or "text"
            cols.append(f"{c.get('name')} {str(c_type).upper()}{pk_str}")
            
        fk_lines = []
        # Localize as FKs desta tabela
        for fk in schema_dict.get('foreign_keys', []):
            if fk.get('source_table') == t.get('name'):
                fk_lines.append(f"FOREIGN KEY ({fk.get('source_column')}) REFERENCES {fk.get('target_table')}({fk.get('target_column')})")
                
        all_cols = cols + fk_lines
        lines.append(f"CREATE TABLE {t.get('name')} (\n  " + ",\n  ".join(all_cols) + "\n);")
        
    return "\n".join(lines)
