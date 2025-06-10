"""
Schema Parser Module - Parse DDL files and database metadata
Integrates with existing OptimizedDataGenerator
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, Name, Punctuation, Literal
import sql_metadata


class SchemaParser:
    """Parse DDL files and extract table metadata for data generation"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.tables = {}
        self.foreign_keys = []
        
    def parse_ddl_file(self, file_path: str) -> Dict[str, Any]:
        """Parse DDL file and return table metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ddl_content = f.read()
            
            return self.parse_ddl_string(ddl_content)
        
        except Exception as e:
            self.logger.error(f"Error parsing DDL file {file_path}: {e}")
            raise
    
    def parse_ddl_string(self, ddl_content: str) -> Dict[str, Any]:
        """Parse DDL string and extract table metadata"""
        try:
            # Parse with sqlparse
            parsed = sqlparse.parse(ddl_content)
            
            # Extract table information
            for statement in parsed:
                if statement.get_type() == 'CREATE':
                    self._parse_create_statement(statement)
            
            # Build final configuration
            config = {
                "locale": "en_GB",
                "output_format": "csv",
                "row_count": 1000,
                "tables": list(self.tables.values())
            }
            
            # Add foreign key relationships
            self._process_foreign_keys()
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error parsing DDL content: {e}")
            raise
    
    def _parse_create_statement(self, statement: Statement):
        """Parse CREATE TABLE statement"""
        tokens = list(statement.flatten())
        
        # Find table name
        table_name = None
        in_create_table = False
        
        for i, token in enumerate(tokens):
            if token.ttype is Keyword and token.value.upper() == 'CREATE':
                in_create_table = True
            elif in_create_table and token.ttype is Keyword and token.value.upper() == 'TABLE':
                # Next non-whitespace token should be table name
                for j in range(i + 1, len(tokens)):
                    if tokens[j].ttype is not Token.Text.Whitespace:
                        table_name = tokens[j].value.strip('`"[]')
                        break
                break
        
        if not table_name:
            return
        
        # Parse table definition
        table_def = self._extract_table_definition(statement)
        columns = self._parse_columns(table_def)
        
        # Build table metadata
        table_metadata = {
            "table_name": table_name,
            "columns": columns,
            "primary_key": [],
            "foreign_keys": [],
            "constraints": []
        }
        
        # Extract constraints
        self._extract_constraints(table_def, table_metadata)
        
        self.tables[table_name] = table_metadata
        self.logger.info(f"Parsed table: {table_name} with {len(columns)} columns")
    
    def _extract_table_definition(self, statement: Statement) -> str:
        """Extract table definition from CREATE statement"""
        sql = str(statement)
        
        # Find content between parentheses
        start = sql.find('(')
        end = sql.rfind(')')
        
        if start != -1 and end != -1:
            return sql[start + 1:end]
        
        return sql
    
    def _parse_columns(self, table_def: str) -> List[Dict[str, Any]]:
        """Parse column definitions from table definition"""
        columns = []
        
        # Split by commas, but be careful of nested parentheses
        column_parts = self._smart_split(table_def, ',')
        
        for part in column_parts:
            part = part.strip()
            if not part or part.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'INDEX', 'KEY')):
                continue
            
            column = self._parse_single_column(part)
            if column:
                columns.append(column)
        
        return columns
    
    def _parse_single_column(self, column_def: str) -> Optional[Dict[str, Any]]:
        """Parse single column definition"""
        parts = column_def.strip().split()
        if len(parts) < 2:
            return None
        
        column_name = parts[0].strip('`"[]')
        data_type = parts[1].upper()
        
        # Map SQL types to generator types
        generator_type = self._map_sql_type_to_generator_type(data_type)
        
        # Parse constraints
        constraints = []
        nullable = True
        default_value = None
        
        for i, part in enumerate(parts[2:], 2):
            part_upper = part.upper()
            
            if part_upper == 'NOT':
                if i + 1 < len(parts) and parts[i + 1].upper() == 'NULL':
                    nullable = False
            elif part_upper == 'PRIMARY':
                if i + 1 < len(parts) and parts[i + 1].upper() == 'KEY':
                    constraints.append('PK')
            elif part_upper == 'UNIQUE':
                constraints.append('unique')
            elif part_upper == 'AUTO_INCREMENT':
                constraints.append('auto_increment')
            elif part_upper == 'DEFAULT':
                if i + 1 < len(parts):
                    default_value = parts[i + 1].strip("'\"")
        
        # Generate appropriate rule based on column name and type
        rule = self._generate_column_rule(column_name, generator_type)
        
        return {
            "name": column_name,
            "type": generator_type,
            "constraints": constraints,
            "nullable": nullable,
            "default": default_value,
            "rule": rule
        }
    
    def _map_sql_type_to_generator_type(self, sql_type: str) -> str:
        """Map SQL data types to generator types"""
        sql_type = sql_type.upper()
        
        # Remove size specifications
        if '(' in sql_type:
            sql_type = sql_type.split('(')[0]
        
        type_mapping = {
            'INT': 'int',
            'INTEGER': 'int',
            'BIGINT': 'int',
            'SMALLINT': 'int',
            'TINYINT': 'int',
            'DECIMAL': 'float',
            'NUMERIC': 'float',
            'FLOAT': 'float',
            'DOUBLE': 'float',
            'REAL': 'float',
            'VARCHAR': 'str',
            'CHAR': 'str',
            'TEXT': 'str',
            'LONGTEXT': 'str',
            'DATE': 'date',
            'DATETIME': 'datetime',
            'TIMESTAMP': 'datetime',
            'TIME': 'time',
            'BOOLEAN': 'bool',
            'BOOL': 'bool',
            'JSON': 'str',
            'UUID': 'str'
        }
        
        return type_mapping.get(sql_type, 'str')
    
    def _generate_column_rule(self, column_name: str, data_type: str) -> Dict[str, Any]:
        """Generate appropriate rules based on column name patterns"""
        column_name_lower = column_name.lower()
        
        # Email patterns
        if any(pattern in column_name_lower for pattern in ['email', 'mail']):
            return {"type": "email"}
        
        # Phone patterns
        if any(pattern in column_name_lower for pattern in ['phone', 'mobile', 'tel']):
            return {"type": "phone_number"}
        
        # Name patterns
        if any(pattern in column_name_lower for pattern in ['first_name', 'fname', 'firstname']):
            return "first_name"
        if any(pattern in column_name_lower for pattern in ['last_name', 'lname', 'lastname', 'surname']):
            return "last_name"
        if column_name_lower in ['name', 'full_name', 'fullname']:
            return "name"
        
        # Address patterns
        if any(pattern in column_name_lower for pattern in ['address', 'street', 'city', 'state', 'country']):
            if 'city' in column_name_lower:
                return "city"
            elif 'state' in column_name_lower:
                return "state"
            elif 'country' in column_name_lower:
                return "country"
            else:
                return "address"
        
        # Date patterns
        if data_type in ['date', 'datetime']:
            if any(pattern in column_name_lower for pattern in ['birth', 'dob']):
                return {
                    "type": "date_range",
                    "start": "1950-01-01",
                    "end": "2005-12-31"
                }
            elif any(pattern in column_name_lower for pattern in ['created', 'updated', 'modified']):
                return {
                    "type": "date_range",
                    "start": "2020-01-01",
                    "end": "2024-12-31"
                }
        
        # Numeric patterns
        if data_type == 'int':
            if any(pattern in column_name_lower for pattern in ['age']):
                return {"type": "range", "min": 18, "max": 80}
            elif any(pattern in column_name_lower for pattern in ['price', 'amount', 'cost']):
                return {"type": "range", "min": 10, "max": 1000}
            elif any(pattern in column_name_lower for pattern in ['quantity', 'count']):
                return {"type": "range", "min": 1, "max": 100}
        
        # Status/Category patterns
        if any(pattern in column_name_lower for pattern in ['status', 'state']):
            return {
                "type": "choice",
                "value": ["active", "inactive", "pending", "completed"],
                "probabilities": {"active": 0.6, "inactive": 0.2, "pending": 0.15, "completed": 0.05}
            }
        
        # Default rules by type
        if data_type == 'str':
            return "text"
        elif data_type == 'int':
            return {"type": "range", "min": 1, "max": 10000}
        elif data_type == 'float':
            return {"type": "range", "min": 1.0, "max": 1000.0}
        elif data_type == 'bool':
            return "bool"
        elif data_type == 'date':
            return {
                "type": "date_range",
                "start": "2020-01-01",
                "end": "2024-12-31"
            }
        
        return {}
    
    def _extract_constraints(self, table_def: str, table_metadata: Dict[str, Any]):
        """Extract PRIMARY KEY and FOREIGN KEY constraints"""
        lines = table_def.split(',')
        
        for line in lines:
            line = line.strip().upper()
            
            # Primary key constraint
            if line.startswith('PRIMARY KEY'):
                pk_match = re.search(r'PRIMARY KEY\s*\(([^)]+)\)', line)
                if pk_match:
                    pk_columns = [col.strip().strip('`"[]') for col in pk_match.group(1).split(',')]
                    table_metadata['primary_key'] = pk_columns
                    
                    # Mark columns as PK
                    for col in table_metadata['columns']:
                        if col['name'] in pk_columns:
                            if 'PK' not in col['constraints']:
                                col['constraints'].append('PK')
            
            # Foreign key constraint
            elif line.startswith('FOREIGN KEY'):
                fk_match = re.search(r'FOREIGN KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\)', line)
                if fk_match:
                    child_columns = [col.strip().strip('`"[]') for col in fk_match.group(1).split(',')]
                    parent_table = fk_match.group(2).strip('`"[]')
                    parent_columns = [col.strip().strip('`"[]') for col in fk_match.group(3).split(',')]
                    
                    # Add to table's foreign keys
                    for child_col, parent_col in zip(child_columns, parent_columns):
                        fk_info = {
                            "child_column": child_col,
                            "parent_table": parent_table,
                            "parent_column": parent_col
                        }
                        table_metadata['foreign_keys'].append(fk_info)
    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter, respecting parentheses"""
        parts = []
        current_part = ""
        paren_count = 0
        
        for char in text:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == delimiter and paren_count == 0:
                parts.append(current_part.strip())
                current_part = ""
                continue
            
            current_part += char
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts
    
    def _process_foreign_keys(self):
        """Process and validate foreign key relationships"""
        for table_name, table_info in self.tables.items():
            for fk in table_info.get('foreign_keys', []):
                parent_table = fk['parent_table']
                
                if parent_table not in self.tables:
                    self.logger.warning(f"Foreign key reference to undefined table: {parent_table}")
                    continue
                
                parent_column = fk['parent_column']
                parent_table_info = self.tables[parent_table]
                
                # Verify parent column exists
                parent_columns = [col['name'] for col in parent_table_info['columns']]
                if parent_column not in parent_columns:
                    self.logger.warning(f"Foreign key reference to undefined column: {parent_table}.{parent_column}")
    
    def export_to_json(self, output_path: str):
        """Export parsed schema to JSON file"""
        config = {
            "locale": "en_GB",
            "output_format": "csv",
            "row_count": 1000,
            "tables": list(self.tables.values())
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Schema exported to {output_path}")
    
    def get_generation_order(self) -> List[str]:
        """Get table generation order based on dependencies"""
        ordered_tables = []
        remaining_tables = set(self.tables.keys())
        
        while remaining_tables:
            # Find tables with no unresolved dependencies
            ready_tables = []
            
            for table_name in remaining_tables:
                table_info = self.tables[table_name]
                dependencies = set()
                
                for fk in table_info.get('foreign_keys', []):
                    parent_table = fk['parent_table']
                    if parent_table != table_name:  # Ignore self-references
                        dependencies.add(parent_table)
                
                # Check if all dependencies are resolved
                if dependencies.issubset(set(ordered_tables)):
                    ready_tables.append(table_name)
            
            if not ready_tables:
                # Circular dependency or self-reference - add remaining tables
                ready_tables = list(remaining_tables)
                self.logger.warning("Circular dependencies detected, adding remaining tables")
            
            ordered_tables.extend(ready_tables)
            remaining_tables -= set(ready_tables)
        
        return ordered_tables


def main():
    """Example usage of SchemaParser"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python schema_parser.py <ddl_file>")
        sys.exit(1)
    
    ddl_file = sys.argv[1]
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse schema
    parser = SchemaParser()
    config = parser.parse_ddl_file(ddl_file)
    
    # Export to JSON
    output_file = ddl_file.replace('.sql', '_schema.json')
    parser.export_to_json(output_file)
    
    print(f"Parsed {len(config['tables'])} tables")
    print(f"Generation order: {parser.get_generation_order()}")


if __name__ == "__main__":
    main()