#!/usr/bin/env python3
"""
Complete Test Data Generator
A comprehensive solution for generating synthetic test data with schema parsing,
profiling, rule-based generation, relationship preservation, and data masking.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import string
import sys
import uuid
import yaml
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Core dependencies
try:
    import pandas as pd
    import numpy as np
    from faker import Faker
    from scipy import stats
    import sqlparse
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install pandas numpy faker scipy sqlparse")
    sys.exit(1)

# Optional dependencies
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import psycopg2
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

warnings.filterwarnings('ignore')

class Logger:
    """Centralized logging and audit service"""

    def __init__(self, log_file: str = "test_data_generator.log"):
        self.log_file = log_file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.audit_trail = []

    def log(self, message: str, level: str = "INFO"):
        getattr(self.logger, level.lower())(message)
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        })

    def save_audit_trail(self, output_path: str):
        audit_file = os.path.join(output_path, "audit_trail.json")
        with open(audit_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=2)
        self.log(f"Audit trail saved to {audit_file}")

class SchemaParser:
    """Parse database schemas from DDL files or database connections"""

    def __init__(self, logger: Logger):
        self.logger = logger

    def parse_ddl(self, ddl_content: str) -> Dict[str, Dict]:
        """Parse DDL content and extract table schemas"""
        tables = {}

        try:
            parsed = sqlparse.parse(ddl_content)

            for statement in parsed:
                if statement.get_type() == 'CREATE':
                    table_info = self._extract_table_info(statement)
                    if table_info:
                        tables[table_info['name']] = table_info

        except Exception as e:
            self.logger.log(f"Error parsing DDL: {e}", "ERROR")

        return tables

    def _extract_table_info(self, statement) -> Optional[Dict]:
        """Extract table information from CREATE TABLE statement"""
        try:
            sql_text = str(statement).strip()

            # Extract table name
            table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', sql_text, re.IGNORECASE)
            if not table_match:
                return None

            table_name = table_match.group(1)

            # Extract columns
            columns_match = re.search(r'\((.*)\)', sql_text, re.DOTALL)
            if not columns_match:
                return None

            columns_text = columns_match.group(1)
            columns = self._parse_columns(columns_text)

            return {
                'name': table_name,
                'columns': columns,
                'primary_keys': [col['name'] for col in columns if col.get('primary_key')],
                'foreign_keys': [col for col in columns if col.get('foreign_key')]
            }

        except Exception as e:
            self.logger.log(f"Error extracting table info: {e}", "ERROR")
            return None

    def _parse_columns(self, columns_text: str) -> List[Dict]:
        """Parse column definitions"""
        columns = []

        # Split by comma, but handle nested parentheses
        column_defs = self._split_column_definitions(columns_text)

        for col_def in column_defs:
            col_def = col_def.strip()
            if not col_def or col_def.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT', 'INDEX')):
                continue

            column_info = self._parse_single_column(col_def)
            if column_info:
                columns.append(column_info)

        return columns

    def _split_column_definitions(self, text: str) -> List[str]:
        """Split column definitions by comma, respecting parentheses"""
        parts = []
        current = ""
        paren_count = 0

        for char in text:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                parts.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            parts.append(current.strip())

        return parts

    def _parse_single_column(self, col_def: str) -> Optional[Dict]:
        """Parse a single column definition"""
        try:
            parts = col_def.split()
            if len(parts) < 2:
                return None

            column_name = parts[0]
            data_type = parts[1].upper()

            # Extract type details
            type_match = re.match(r'(\w+)(?:\(([^)]+)\))?', data_type)
            if type_match:
                base_type = type_match.group(1)
                type_params = type_match.group(2)
            else:
                base_type = data_type
                type_params = None

            column_info = {
                'name': column_name,
                'type': base_type,
                'type_params': type_params,
                'nullable': 'NOT NULL' not in col_def.upper(),
                'primary_key': 'PRIMARY KEY' in col_def.upper(),
                'auto_increment': 'AUTO_INCREMENT' in col_def.upper() or 'SERIAL' in col_def.upper(),
                'unique': 'UNIQUE' in col_def.upper(),
                'default': self._extract_default_value(col_def)
            }

            # Check for foreign key
            fk_match = re.search(r'REFERENCES\s+(\w+)\((\w+)\)', col_def, re.IGNORECASE)
            if fk_match:
                column_info['foreign_key'] = {
                    'table': fk_match.group(1),
                    'column': fk_match.group(2)
                }

            return column_info

        except Exception as e:
            self.logger.log(f"Error parsing column {col_def}: {e}", "ERROR")
            return None

    def _extract_default_value(self, col_def: str) -> Optional[str]:
        """Extract default value from column definition"""
        default_match = re.search(r'DEFAULT\s+([^\s,]+)', col_def, re.IGNORECASE)
        if default_match:
            return default_match.group(1).strip("'\"")
        return None

class DataProfiler:
    """Profile sample data to understand distributions and patterns"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.faker = Faker()

    def profile_dataframe(self, df: pd.DataFrame, table_name: str) -> Dict:
        """Profile a pandas DataFrame"""
        profile = {
            'table_name': table_name,
            'row_count': len(df),
            'columns': {}
        }

        for column in df.columns:
            profile['columns'][column] = self._profile_column(df[column], column)

        return profile

    def _profile_column(self, series: pd.Series, column_name: str) -> Dict:
        """Profile a single column"""
        profile = {
            'name': column_name,
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': series.nunique() / len(series) * 100
        }

        # Type-specific profiling
        if pd.api.types.is_numeric_dtype(series):
            profile.update(self._profile_numeric(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update(self._profile_datetime(series))
        else:
            profile.update(self._profile_categorical(series))

        return profile

    def _profile_numeric(self, series: pd.Series) -> Dict:
        """Profile numeric column"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {'data_type': 'numeric'}

        return {
            'data_type': 'numeric',
            'min': float(non_null.min()),
            'max': float(non_null.max()),
            'mean': float(non_null.mean()),
            'median': float(non_null.median()),
            'std': float(non_null.std()) if len(non_null) > 1 else 0,
            'quartiles': [float(q) for q in non_null.quantile([0.25, 0.5, 0.75]).tolist()]
        }

    def _profile_datetime(self, series: pd.Series) -> Dict:
        """Profile datetime column"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {'data_type': 'datetime'}

        return {
            'data_type': 'datetime',
            'min_date': non_null.min().isoformat() if hasattr(non_null.min(), 'isoformat') else str(non_null.min()),
            'max_date': non_null.max().isoformat() if hasattr(non_null.max(), 'isoformat') else str(non_null.max())
        }

    def _profile_categorical(self, series: pd.Series) -> Dict:
        """Profile categorical/text column"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {'data_type': 'categorical'}

        value_counts = non_null.value_counts()

        profile = {
            'data_type': 'categorical',
            'most_common': value_counts.head(10).to_dict(),
            'avg_length': float(non_null.astype(str).str.len().mean()),
            'max_length': int(non_null.astype(str).str.len().max()),
            'min_length': int(non_null.astype(str).str.len().min())
        }

        # Detect patterns
        patterns = self._detect_patterns(non_null)
        if patterns:
            profile['patterns'] = patterns

        return profile

    def _detect_patterns(self, series: pd.Series) -> Dict:
        """Detect common patterns in text data"""
        sample = series.astype(str).sample(min(100, len(series))).tolist()

        patterns = {}

        # Email pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email_matches = sum(1 for s in sample if re.match(email_pattern, s))
        if email_matches / len(sample) > 0.8:
            patterns['email'] = True

        # Phone pattern
        phone_pattern = r'^\+?[\d\s\-\(\)]{10,}$'
        phone_matches = sum(1 for s in sample if re.match(phone_pattern, s))
        if phone_matches / len(sample) > 0.8:
            patterns['phone'] = True

        # UUID pattern
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        uuid_matches = sum(1 for s in sample if re.match(uuid_pattern, s.lower()))
        if uuid_matches / len(sample) > 0.8:
            patterns['uuid'] = True

        return patterns

class RuleEngine:
    """Rule engine for managing data generation templates and rules"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.rules = {}
        self.templates = {}

    def load_rules(self, rules_content: Union[str, Dict]) -> None:
        """Load rules from YAML content or dictionary"""
        try:
            if isinstance(rules_content, str):
                self.rules = yaml.safe_load(rules_content)
            else:
                self.rules = rules_content

            self.logger.log(f"Loaded rules for {len(self.rules.get('tables', {}))} tables")

        except Exception as e:
            self.logger.log(f"Error loading rules: {e}", "ERROR")
            raise

    def get_table_rules(self, table_name: str) -> Dict:
        """Get rules for a specific table"""
        return self.rules.get('tables', {}).get(table_name, {})

    def get_column_rules(self, table_name: str, column_name: str) -> Dict:
        """Get rules for a specific column"""
        table_rules = self.get_table_rules(table_name)
        return table_rules.get('columns', {}).get(column_name, {})

    def infer_rules_from_profile(self, profile: Dict) -> Dict:
        """Infer generation rules from data profile"""
        rules = {
            'tables': {}
        }

        table_name = profile['table_name']
        table_rules = {
            'row_count': max(100, profile['row_count']),
            'columns': {}
        }

        for col_name, col_profile in profile['columns'].items():
            column_rules = self._infer_column_rules(col_profile)
            table_rules['columns'][col_name] = column_rules

        rules['tables'][table_name] = table_rules
        return rules

    def _infer_column_rules(self, col_profile: Dict) -> Dict:
        """Infer rules for a single column based on its profile"""
        rules = {
            'nullable': col_profile['null_percentage'] > 0,
            'null_probability': col_profile['null_percentage'] / 100
        }

        data_type = col_profile['data_type']

        if data_type == 'numeric':
            rules.update({
                'type': 'numeric',
                'min': col_profile.get('min', 0),
                'max': col_profile.get('max', 100),
                'distribution': 'normal',
                'mean': col_profile.get('mean', 50),
                'std': col_profile.get('std', 10)
            })
        elif data_type == 'datetime':
            rules.update({
                'type': 'datetime',
                'start_date': col_profile.get('min_date', '2020-01-01'),
                'end_date': col_profile.get('max_date', '2024-12-31')
            })
        elif data_type == 'categorical':
            # Check for patterns
            patterns = col_profile.get('patterns', {})

            if patterns.get('email'):
                rules['type'] = 'email'
            elif patterns.get('phone'):
                rules['type'] = 'phone'
            elif patterns.get('uuid'):
                rules['type'] = 'uuid'
            else:
                rules.update({
                    'type': 'text',
                    'min_length': col_profile.get('min_length', 1),
                    'max_length': col_profile.get('max_length', 50),
                    'values': list(col_profile.get('most_common', {}).keys())[:10] if col_profile.get('most_common') else None
                })

        return rules

class DataGenerator:
    """Core data generation engine using Faker and custom logic"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.faker = Faker()
        self.generated_data = {}
        self.foreign_key_data = {}

    def generate_table_data(self, table_name: str, schema: Dict, rules: Dict, row_count: int = None) -> pd.DataFrame:
        """Generate data for a single table"""
        if row_count is None:
            row_count = rules.get('row_count', 100)

        self.logger.log(f"Generating {row_count} rows for table {table_name}")

        data = {}
        columns = schema.get('columns', [])

        # Generate data for each column
        for column in columns:
            col_name = column['name']
            col_rules = rules.get('columns', {}).get(col_name, {})

            # Merge schema info with rules
            merged_rules = self._merge_column_info(column, col_rules)

            data[col_name] = self._generate_column_data(col_name, merged_rules, row_count, table_name)

        df = pd.DataFrame(data)
        self.generated_data[table_name] = df

        return df

    def _merge_column_info(self, column_schema: Dict, column_rules: Dict) -> Dict:
        """Merge schema information with generation rules"""
        merged = column_rules.copy()

        # Add schema information
        merged['schema_type'] = column_schema.get('type', 'VARCHAR')
        merged['nullable'] = merged.get('nullable', column_schema.get('nullable', True))
        merged['primary_key'] = column_schema.get('primary_key', False)
        merged['auto_increment'] = column_schema.get('auto_increment', False)
        merged['foreign_key'] = column_schema.get('foreign_key')
        merged['unique'] = column_schema.get('unique', False)
        merged['default'] = column_schema.get('default')

        # Infer type from schema if not specified in rules
        if 'type' not in merged:
            merged['type'] = self._infer_type_from_schema(column_schema['type'])

        return merged

    def _infer_type_from_schema(self, schema_type: str) -> str:
        """Infer generation type from schema type"""
        schema_type = schema_type.upper()

        if schema_type in ['INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT']:
            return 'integer'
        elif schema_type in ['FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC']:
            return 'float'
        elif schema_type in ['DATE']:
            return 'date'
        elif schema_type in ['DATETIME', 'TIMESTAMP']:
            return 'datetime'
        elif schema_type in ['TIME']:
            return 'time'
        elif schema_type in ['BOOLEAN', 'BOOL']:
            return 'boolean'
        else:
            return 'text'

    def _generate_column_data(self, col_name: str, rules: Dict, row_count: int, table_name: str) -> List:
        """Generate data for a single column"""
        data = []

        # Handle foreign keys
        if rules.get('foreign_key'):
            return self._generate_foreign_key_data(rules['foreign_key'], row_count)

        # Handle primary keys
        if rules.get('primary_key') and rules.get('auto_increment'):
            return list(range(1, row_count + 1))

        # Generate data based on type
        data_type = rules.get('type', 'text')

        for i in range(row_count):
            # Handle nulls
            if rules.get('nullable', True) and random.random() < rules.get('null_probability', 0.1):
                data.append(None)
                continue

            # Generate value based on type
            value = self._generate_single_value(data_type, rules, i)

            # Ensure uniqueness if required
            if rules.get('unique', False):
                while value in data:
                    value = self._generate_single_value(data_type, rules, i)

            data.append(value)

        return data

    def _generate_single_value(self, data_type: str, rules: Dict, index: int) -> Any:
        """Generate a single value based on type and rules"""

        # Use default value if specified
        if rules.get('default') is not None:
            return rules['default']

        # Use predefined values if available
        if rules.get('values'):
            return random.choice(rules['values'])

        # Generate based on type
        if data_type == 'integer':
            return random.randint(rules.get('min', 1), rules.get('max', 1000))

        elif data_type == 'float':
            return round(random.uniform(rules.get('min', 0.0), rules.get('max', 1000.0)), 2)

        elif data_type == 'boolean':
            return random.choice([True, False])

        elif data_type == 'date':
            start_date = datetime.strptime(rules.get('start_date', '2020-01-01'), '%Y-%m-%d')
            end_date = datetime.strptime(rules.get('end_date', '2024-12-31'), '%Y-%m-%d')
            return self.faker.date_between(start_date=start_date, end_date=end_date)

        elif data_type == 'datetime':
            start_date = datetime.strptime(rules.get('start_date', '2020-01-01'), '%Y-%m-%d')
            end_date = datetime.strptime(rules.get('end_date', '2024-12-31'), '%Y-%m-%d')
            return self.faker.date_time_between(start_date=start_date, end_date=end_date)

        elif data_type == 'time':
            return self.faker.time()

        elif data_type == 'email':
            return self.faker.email()

        elif data_type == 'phone':
            return self.faker.phone_number()

        elif data_type == 'uuid':
            return str(uuid.uuid4())

        elif data_type == 'name':
            return self.faker.name()

        elif data_type == 'address':
            return self.faker.address()

        elif data_type == 'text':
            min_length = rules.get('min_length', 5)
            max_length = rules.get('max_length', 50)
            length = random.randint(min_length, max_length)
            return self.faker.text(max_nb_chars=length).replace('\n', ' ').strip()

        else:
            # Default to text
            return self.faker.word()

    def _generate_foreign_key_data(self, fk_info: Dict, row_count: int) -> List:
        """Generate foreign key data based on referenced table"""
        ref_table = fk_info['table']
        ref_column = fk_info['column']

        # Check if referenced data exists
        if ref_table in self.generated_data:
            ref_values = self.generated_data[ref_table][ref_column].dropna().tolist()
            if ref_values:
                return [random.choice(ref_values) for _ in range(row_count)]

        # Check foreign key cache
        fk_key = f"{ref_table}.{ref_column}"
        if fk_key not in self.foreign_key_data:
            # Generate placeholder foreign key values
            self.foreign_key_data[fk_key] = list(range(1, 1001))

        ref_values = self.foreign_key_data[fk_key]
        return [random.choice(ref_values) for _ in range(row_count)]

class RelationshipPreserver:
    """Maintain referential integrity between tables"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.dependency_graph = {}
        self.generation_order = []

    def build_dependency_graph(self, schemas: Dict[str, Dict]) -> None:
        """Build dependency graph based on foreign key relationships"""
        self.dependency_graph = {table: [] for table in schemas.keys()}

        for table_name, schema in schemas.items():
            for column in schema.get('columns', []):
                if column.get('foreign_key'):
                    ref_table = column['foreign_key']['table']
                    if ref_table in self.dependency_graph:
                        self.dependency_graph[table_name].append(ref_table)

        self.logger.log(f"Built dependency graph: {self.dependency_graph}")

    def get_generation_order(self) -> List[str]:
        """Get the order in which tables should be generated to maintain referential integrity"""
        if self.generation_order:
            return self.generation_order

        # Topological sort
        visited = set()
        temp_visited = set()
        order = []

        def visit(table):
            if table in temp_visited:
                self.logger.log(f"Circular dependency detected involving {table}", "WARNING")
                return
            if table in visited:
                return

            temp_visited.add(table)
            for dependency in self.dependency_graph.get(table, []):
                visit(dependency)
            temp_visited.remove(table)
            visited.add(table)
            order.append(table)

        for table in self.dependency_graph.keys():
            if table not in visited:
                visit(table)

        self.generation_order = order
        self.logger.log(f"Table generation order: {order}")
        return order

class MaskingEngine:
    """Data masking and anonymization engine"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.faker = Faker()

    def apply_masking(self, df: pd.DataFrame, masking_rules: Dict) -> pd.DataFrame:
        """Apply masking rules to a DataFrame"""
        masked_df = df.copy()

        for column, rule in masking_rules.items():
            if column in masked_df.columns:
                masked_df[column] = self._mask_column(masked_df[column], rule)

        return masked_df

    def _mask_column(self, series: pd.Series, rule: Dict) -> pd.Series:
        """Apply masking to a single column"""
        method = rule.get('method', 'hash')

        if method == 'hash':
            return series.apply(lambda x: self._hash_value(x) if pd.notna(x) else x)

        elif method == 'shuffle':
            non_null_values = series.dropna().tolist()
            random.shuffle(non_null_values)
            non_null_iter = iter(non_null_values)
            return series.apply(lambda x: next(non_null_iter) if pd.notna(x) else x)

        elif method == 'null':
            return pd.Series([None] * len(series))

        elif method == 'fake':
            fake_type = rule.get('fake_type', 'word')
            return series.apply(lambda x: self._generate_fake_value(fake_type) if pd.notna(x) else x)

        elif method == 'static':
            static_value = rule.get('value', 'MASKED')
            return pd.Series([static_value] * len(series))

        else:
            self.logger.log(f"Unknown masking method: {method}", "WARNING")
            return series

    def _hash_value(self, value: Any) -> str:
        """Hash a value using SHA256"""
        return hashlib.sha256(str(value).encode()).hexdigest()[:16]

    def _generate_fake_value(self, fake_type: str) -> Any:
        """Generate fake value based on type"""
        if hasattr(self.faker, fake_type):
            return getattr(self.faker, fake_type)()
        else:
            return self.faker.word()

class Exporter:
    """Export data to various formats and destinations"""

    def __init__(self, logger: Logger):
        self.logger = logger

    def export_data(self, data: Dict[str, pd.DataFrame], output_config: Dict) -> None:
        """Export data based on configuration"""
        format_type = output_config.get('format', 'csv')
        destination = output_config.get('destination', 'local')
        output_path = output_config.get('path', './output')

        if destination == 'local':
            self._export_local(data, format_type, output_path)
        elif destination == 's3':
            self._export_s3(data, format_type, output_config)
        elif destination == 'postgres':
            self._export_postgres(data, output_config)
        else:
            self.logger.log(f"Unknown destination: {destination}", "ERROR")

    def _export_local(self, data: Dict[str, pd.DataFrame], format_type: str, output_path: str) -> None:
        """Export data to local filesystem"""
        os.makedirs(output_path, exist_ok=True)

        for table_name, df in data.items():
            if format_type == 'csv':
                file_path = os.path.join(output_path, f"{table_name}.csv")
                df.to_csv(file_path, index=False)
            elif format_type == 'json':
                file_path = os.path.join(output_path, f"{table_name}.json")
                df.to_json(file_path, orient='records', indent=2)
            elif format_type == 'parquet':
                try:
                    file_path = os.path.join(output_path, f"{table_name}.parquet")
                    df.to_parquet(file_path, index=False)
                except ImportError:
                    self.logger.log("Parquet export requires pyarrow. Falling back to CSV", "WARNING")
                    file_path = os.path.join(output_path, f"{table_name}.csv")
                    df.to_csv(file_path, index=False)

            self.logger.log(f"Exported {table_name} to {file_path}")

    def _export_s3(self, data: Dict[str, pd.DataFrame], format_type: str, config: Dict) -> None:
        """Export data to S3"""
        if not HAS_BOTO3:
            self.logger.log("S3 export requires boto3", "ERROR")
            return

        bucket = config.get('bucket')
        prefix = config.get('prefix', 'test_data')

        s3_client = boto3.client('s3')

        for table_name, df in data.items():
            key = f"{prefix}/{table_name}.{format_type}"

            if format_type == 'csv':
                csv_buffer = df.to_csv(index=False)
                s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer)
            elif format_type == 'json':
                json_buffer = df.to_json(orient='records', indent=2)
                s3_client.put_object(Bucket=bucket, Key=key, Body=json_buffer)

            self.logger.log(f"Exported {table_name} to s3://{bucket}/{key}")

    def _export_postgres(self, data: Dict[str, pd.DataFrame], config: Dict) -> None:
        """Export data to PostgreSQL"""
        if not HAS_POSTGRES:
            self.logger.log("PostgreSQL export requires psycopg2", "ERROR")
            return

        try:
            conn = psycopg2.connect(
                host=config.get('host', 'localhost'),
                database=config.get('database'),
                user=config.get('user'),
                password=config.get('password'),
                port=config.get('port', 5432)
            )

            for table_name, df in data.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                self.logger.log(f"Exported {table_name} to PostgreSQL")

            conn.close()

        except Exception as e:
            self.logger.log(f"Error exporting to PostgreSQL: {e}", "ERROR")

class ConfigLoader:
    """Configuration loader and validator"""

    def __init__(self, logger: Logger):
        self.logger = logger

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.logger.log(f"Loaded configuration from {config_path}")
            return self._validate_config(config)

        except Exception as e:
            self.logger.log(f"Error loading config: {e}", "ERROR")
            return self._get_default_config()

    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set defaults for configuration"""
        # Set defaults
        config.setdefault('output', {
            'format': 'csv',
            'destination': 'local',
            'path': './output'
        })

        config.setdefault('generation', {
            'default_row_count': 100,
            'preserve_relationships': True
        })

        config.setdefault('masking', {
            'enabled': False,
            'rules': {}
        })

        return config

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'output': {
                'format': 'csv',
                'destination': 'local',
                'path': './output'
            },
            'generation': {
                'default_row_count': 100,
                'preserve_relationships': True
            },
            'masking': {
                'enabled': False,
                'rules': {}
            }
        }

class JobScheduler:
    """Job scheduling and management"""

    def __init__(self, logger: Logger):
        self.logger = logger
        self.jobs = {}

    def schedule_job(self, job_id: str, job_config: Dict) -> None:
        """Schedule a data generation job"""
        self.jobs[job_id] = {
            'config': job_config,
            'status': 'scheduled',
            'created_at': datetime.now(),
            'started_at': None,
            'completed_at': None,
            'error': None
        }

        self.logger.log(f"Scheduled job {job_id}")

    def run_job(self, job_id: str) -> bool:
        """Run a scheduled job"""
        if job_id not in self.jobs:
            self.logger.log(f"Job {job_id} not found", "ERROR")
            return False

        job = self.jobs[job_id]
        job['status'] = 'running'
        job['started_at'] = datetime.now()

        try:
            self.logger.log(f"Starting job {job_id}")

            # Create generator instance and run
            generator = TestDataGenerator()
            generator.run(job['config'])

            job['status'] = 'completed'
            job['completed_at'] = datetime.now()

            self.logger.log(f"Job {job_id} completed successfully")
            return True

        except Exception as e:
            job['status'] = 'failed'
            job['error'] = str(e)
            job['completed_at'] = datetime.now()

            self.logger.log(f"Job {job_id} failed: {e}", "ERROR")
            return False

    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a job"""
        return self.jobs.get(job_id, {'status': 'not_found'})

class TestDataGenerator:
    """Main orchestrator class"""

    def __init__(self, config_path: str = None):
        self.logger = Logger()
        self.config_loader = ConfigLoader(self.logger)
        self.schema_parser = SchemaParser(self.logger)
        self.data_profiler = DataProfiler(self.logger)
        self.rule_engine = RuleEngine(self.logger)
        self.data_generator = DataGenerator(self.logger)
        self.relationship_preserver = RelationshipPreserver(self.logger)
        self.masking_engine = MaskingEngine(self.logger)
        self.exporter = Exporter(self.logger)
        self.job_scheduler = JobScheduler(self.logger)

        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = self.config_loader.load_config(config_path)
        else:
            self.config = self.config_loader._get_default_config()

    def run(self, args: Dict = None) -> None:
        """Main execution flow"""
        try:
            self.logger.log("Starting test data generation")

            # Parse arguments
            if args:
                self._update_config_from_args(args)

            # Step 1: Load and parse schema/data
            schemas = self._load_schemas()

            # Step 2: Load or infer rules
            rules = self._load_rules(schemas)

            # Step 3: Build dependency graph and determine generation order
            if self.config['generation']['preserve_relationships']:
                self.relationship_preserver.build_dependency_graph(schemas)
                generation_order = self.relationship_preserver.get_generation_order()
            else:
                generation_order = list(schemas.keys())

            # Step 4: Generate data for each table
            generated_data = {}
            for table_name in generation_order:
                if table_name in schemas:
                    schema = schemas[table_name]
                    table_rules = rules.get('tables', {}).get(table_name, {})

                    row_count = table_rules.get('row_count', self.config['generation']['default_row_count'])
                    df = self.data_generator.generate_table_data(table_name, schema, table_rules, row_count)
                    generated_data[table_name] = df

            # Step 5: Apply masking if enabled
            if self.config['masking']['enabled']:
                for table_name, df in generated_data.items():
                    masking_rules = self.config['masking']['rules'].get(table_name, {})
                    if masking_rules:
                        generated_data[table_name] = self.masking_engine.apply_masking(df, masking_rules)

            # Step 6: Export data
            self.exporter.export_data(generated_data, self.config['output'])

            # Step 7: Save audit trail
            self.logger.save_audit_trail(self.config['output']['path'])

            self.logger.log("Test data generation completed successfully")

        except Exception as e:
            self.logger.log(f"Error in test data generation: {e}", "ERROR")
            raise

    def _update_config_from_args(self, args: Dict) -> None:
        """Update configuration from command line arguments"""
        if args.get('output_path'):
            self.config['output']['path'] = args['output_path']

        if args.get('format'):
            self.config['output']['format'] = args['format']

        if args.get('row_count'):
            self.config['generation']['default_row_count'] = args['row_count']

    def _load_schemas(self) -> Dict[str, Dict]:
        """Load schemas from various sources"""
        schemas = {}

        # Load from DDL file
        if self.config.get('input', {}).get('ddl_file'):
            ddl_file = self.config['input']['ddl_file']
            if os.path.exists(ddl_file):
                with open(ddl_file, 'r') as f:
                    ddl_content = f.read()
                schemas.update(self.schema_parser.parse_ddl(ddl_content))

        # Load from sample data
        if self.config.get('input', {}).get('sample_data'):
            sample_data_config = self.config['input']['sample_data']

            if isinstance(sample_data_config, dict):
                for table_name, file_path in sample_data_config.items():
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        profile = self.data_profiler.profile_dataframe(df, table_name)
                        schema = self._profile_to_schema(profile)
                        schemas[table_name] = schema

        # If no schemas loaded, create a default one
        if not schemas:
            self.logger.log("No schemas found, creating default schema", "WARNING")
            schemas = self._create_default_schema()

        return schemas

    def _profile_to_schema(self, profile: Dict) -> Dict:
        """Convert data profile to schema format"""
        columns = []

        for col_name, col_profile in profile['columns'].items():
            column = {
                'name': col_name,
                'type': self._profile_type_to_schema_type(col_profile['data_type']),
                'nullable': col_profile['null_percentage'] > 0,
                'primary_key': False,
                'auto_increment': False,
                'unique': col_profile['unique_percentage'] > 95,
                'default': None
            }
            columns.append(column)

        return {
            'name': profile['table_name'],
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        }

    def _profile_type_to_schema_type(self, profile_type: str) -> str:
        """Convert profile data type to schema type"""
        type_mapping = {
            'numeric': 'INTEGER',
            'datetime': 'DATETIME',
            'categorical': 'VARCHAR'
        }
        return type_mapping.get(profile_type, 'VARCHAR')

    def _create_default_schema(self) -> Dict[str, Dict]:
        """Create a default schema for demonstration"""
        return {
            'users': {
                'name': 'users',
                'columns': [
                    {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True, 'auto_increment': True, 'unique': True, 'default': None},
                    {'name': 'name', 'type': 'VARCHAR', 'nullable': False, 'primary_key': False, 'auto_increment': False, 'unique': False, 'default': None},
                    {'name': 'email', 'type': 'VARCHAR', 'nullable': False, 'primary_key': False, 'auto_increment': False, 'unique': True, 'default': None},
                    {'name': 'age', 'type': 'INTEGER', 'nullable': True, 'primary_key': False, 'auto_increment': False, 'unique': False, 'default': None},
                    {'name': 'created_at', 'type': 'DATETIME', 'nullable': False, 'primary_key': False, 'auto_increment': False, 'unique': False, 'default': None}
                ],
                'primary_keys': ['id'],
                'foreign_keys': []
            },
            'orders': {
                'name': 'orders',
                'columns': [
                    {'name': 'id', 'type': 'INTEGER', 'nullable': False, 'primary_key': True, 'auto_increment': True, 'unique': True, 'default': None},
                    {'name': 'user_id', 'type': 'INTEGER', 'nullable': False, 'primary_key': False, 'auto_increment': False, 'unique': False, 'default': None, 'foreign_key': {'table': 'users', 'column': 'id'}},
                    {'name': 'amount', 'type': 'DECIMAL', 'nullable': False, 'primary_key': False, 'auto_increment': False, 'unique': False, 'default': None},
                    {'name': 'status', 'type': 'VARCHAR', 'nullable': False, 'primary_key': False, 'auto_increment': False, 'unique': False, 'default': None},
                    {'name': 'order_date', 'type': 'DATETIME', 'nullable': False, 'primary_key': False, 'auto_increment': False, 'unique': False, 'default': None}
                ],
                'primary_keys': ['id'],
                'foreign_keys': [{'name': 'user_id', 'foreign_key': {'table': 'users', 'column': 'id'}}]
            }
        }

    def _load_rules(self, schemas: Dict[str, Dict]) -> Dict:
        """Load or infer generation rules"""
        rules = {}

        # Load from rules file
        if self.config.get('input', {}).get('rules_file'):
            rules_file = self.config['input']['rules_file']
            if os.path.exists(rules_file):
                with open(rules_file, 'r') as f:
                    rules_content = f.read()
                self.rule_engine.load_rules(rules_content)
                rules = self.rule_engine.rules

        # If no rules loaded, create default rules
        if not rules:
            rules = self._create_default_rules(schemas)

        return rules

    def _create_default_rules(self, schemas: Dict[str, Dict]) -> Dict:
        """Create default generation rules based on schemas"""
        rules = {
            'tables': {}
        }

        for table_name, schema in schemas.items():
            table_rules = {
                'row_count': self.config['generation']['default_row_count'],
                'columns': {}
            }

            for column in schema['columns']:
                col_name = column['name']
                col_type = column['type'].upper()

                # Create rules based on column type and name
                column_rules = self._infer_column_rules_from_schema(column)
                table_rules['columns'][col_name] = column_rules

            rules['tables'][table_name] = table_rules

        return rules

    def _infer_column_rules_from_schema(self, column: Dict) -> Dict:
        """Infer generation rules from column schema"""
        col_name = column['name'].lower()
        col_type = column['type'].upper()

        rules = {
            'nullable': column.get('nullable', True),
            'null_probability': 0.1 if column.get('nullable', True) else 0
        }

        # Infer type based on column name patterns
        if 'email' in col_name:
            rules['type'] = 'email'
        elif 'phone' in col_name:
            rules['type'] = 'phone'
        elif 'name' in col_name:
            rules['type'] = 'name'
        elif 'address' in col_name:
            rules['type'] = 'address'
        elif 'id' in col_name and col_type in ['INTEGER', 'INT', 'BIGINT']:
            rules['type'] = 'integer'
            rules['min'] = 1
            rules['max'] = 10000
        elif col_type in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT']:
            rules['type'] = 'integer'
            rules['min'] = 1
            rules['max'] = 1000
        elif col_type in ['DECIMAL', 'FLOAT', 'DOUBLE', 'NUMERIC']:
            rules['type'] = 'float'
            rules['min'] = 0.0
            rules['max'] = 1000.0
        elif col_type in ['DATE']:
            rules['type'] = 'date'
            rules['start_date'] = '2020-01-01'
            rules['end_date'] = '2024-12-31'
        elif col_type in ['DATETIME', 'TIMESTAMP']:
            rules['type'] = 'datetime'
            rules['start_date'] = '2020-01-01'
            rules['end_date'] = '2024-12-31'
        elif col_type in ['BOOLEAN', 'BOOL']:
            rules['type'] = 'boolean'
        else:
            rules['type'] = 'text'
            rules['min_length'] = 5
            rules['max_length'] = 50

        # Handle specific column patterns
        if col_name in ['status', 'state', 'type', 'category']:
            rules['values'] = self._get_common_values_for_column(col_name)

        return rules

    def _get_common_values_for_column(self, col_name: str) -> List[str]:
        """Get common values for specific column types"""
        value_mappings = {
            'status': ['active', 'inactive', 'pending', 'completed', 'cancelled'],
            'state': ['draft', 'published', 'archived', 'deleted'],
            'type': ['standard', 'premium', 'basic', 'enterprise'],
            'category': ['electronics', 'clothing', 'books', 'home', 'sports']
        }

        return value_mappings.get(col_name, ['option1', 'option2', 'option3'])

def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        'input': {
            'ddl_file': 'schema.sql',
            'sample_data': {
                'users': 'sample_users.csv',
                'orders': 'sample_orders.csv'
            },
            'rules_file': 'generation_rules.yaml'
        },
        'generation': {
            'default_row_count': 1000,
            'preserve_relationships': True
        },
        'masking': {
            'enabled': True,
            'rules': {
                'users': {
                    'email': {'method': 'fake', 'fake_type': 'email'},
                    'name': {'method': 'fake', 'fake_type': 'name'}
                }
            }
        },
        'output': {
            'format': 'csv',
            'destination': 'local',
            'path': './output'
        }
    }

    return yaml.dump(sample_config, default_flow_style=False)

def create_sample_rules():
    """Create sample generation rules"""
    sample_rules = {
        'tables': {
            'users': {
                'row_count': 1000,
                'columns': {
                    'id': {
                        'type': 'integer',
                        'min': 1,
                        'max': 10000,
                        'nullable': False
                    },
                    'name': {
                        'type': 'name',
                        'nullable': False
                    },
                    'email': {
                        'type': 'email',
                        'nullable': False
                    },
                    'age': {
                        'type': 'integer',
                        'min': 18,
                        'max': 80,
                        'nullable': True,
                        'null_probability': 0.1
                    },
                    'status': {
                        'type': 'text',
                        'values': ['active', 'inactive', 'suspended']
                    }
                }
            },
            'orders': {
                'row_count': 5000,
                'columns': {
                    'id': {
                        'type': 'integer',
                        'min': 1,
                        'max': 100000,
                        'nullable': False
                    },
                    'user_id': {
                        'type': 'integer',
                        'min': 1,
                        'max': 1000,
                        'nullable': False
                    },
                    'amount': {
                        'type': 'float',
                        'min': 10.0,
                        'max': 1000.0,
                        'nullable': False
                    },
                    'status': {
                        'type': 'text',
                        'values': ['pending', 'completed', 'cancelled', 'refunded']
                    }
                }
            }
        }
    }

    return yaml.dump(sample_rules, default_flow_style=False)

def create_sample_ddl():
    """Create sample DDL for testing"""
    ddl = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(100) NOT NULL UNIQUE,
        age INTEGER,
        phone VARCHAR(20),
        address TEXT,
        created_at DATETIME NOT NULL,
        status VARCHAR(20) DEFAULT 'active'
    );
    
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        user_id INTEGER NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        order_date DATETIME NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    
    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        order_id INTEGER NOT NULL,
        product_name VARCHAR(200) NOT NULL,
        quantity INTEGER NOT NULL,
        price DECIMAL(10,2) NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id)
    );
    """
    return ddl

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Comprehensive Test Data Generator')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--ddl', help='DDL file path')
    parser.add_argument('--rules', help='Rules file path')
    parser.add_argument('--output', '-o', help='Output directory path', default='./output')
    parser.add_argument('--format', help='Output format (csv, json, parquet)', default='csv')
    parser.add_argument('--rows', type=int, help='Number of rows to generate', default=100)
    parser.add_argument('--sample-config', action='store_true', help='Generate sample configuration file')
    parser.add_argument('--sample-rules', action='store_true', help='Generate sample rules file')
    parser.add_argument('--sample-ddl', action='store_true', help='Generate sample DDL file')
    parser.add_argument('--profile', help='Profile sample data from CSV file')
    parser.add_argument('--mask', action='store_true', help='Enable data masking')

    args = parser.parse_args()

    # Generate sample files if requested
    if args.sample_config:
        with open('sample_config.yaml', 'w') as f:
            f.write(create_sample_config())
        print("Sample configuration created: sample_config.yaml")
        return

    if args.sample_rules:
        with open('sample_rules.yaml', 'w') as f:
            f.write(create_sample_rules())
        print("Sample rules created: sample_rules.yaml")
        return

    if args.sample_ddl:
        with open('sample_schema.sql', 'w') as f:
            f.write(create_sample_ddl())
        print("Sample DDL created: sample_schema.sql")
        return

    # Profile data if requested
    if args.profile:
        logger = Logger()
        profiler = DataProfiler(logger)

        try:
            df = pd.read_csv(args.profile)
            table_name = os.path.splitext(os.path.basename(args.profile))[0]
            profile = profiler.profile_dataframe(df, table_name)

            profile_file = f"{table_name}_profile.json"
            with open(profile_file, 'w') as f:
                json.dump(profile, f, indent=2, default=str)

            print(f"Data profile saved to: {profile_file}")

        except Exception as e:
            print(f"Error profiling data: {e}")

        return

    # Create generator configuration
    config_args = {
        'output_path': args.output,
        'format': args.format,
        'row_count': args.rows
    }

    # Update configuration with file paths
    file_config = {}

    if args.ddl:
        file_config['input'] = file_config.get('input', {})
        file_config['input']['ddl_file'] = args.ddl

    if args.rules:
        file_config['input'] = file_config.get('input', {})
        file_config['input']['rules_file'] = args.rules

    if args.mask:
        file_config['masking'] = {
            'enabled': True,
            'rules': {
                'users': {
                    'email': {'method': 'fake', 'fake_type': 'email'},
                    'name': {'method': 'fake', 'fake_type': 'name'},
                    'phone': {'method': 'fake', 'fake_type': 'phone_number'}
                }
            }
        }

    try:
        # Create and run generator
        generator = TestDataGenerator(args.config)

        # Update config with file paths
        if file_config:
            for key, value in file_config.items():
                if isinstance(value, dict):
                    generator.config[key] = generator.config.get(key, {})
                    generator.config[key].update(value)
                else:
                    generator.config[key] = value

        generator.run(config_args)

        print(f"Test data generation completed successfully!")
        print(f"Output directory: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()