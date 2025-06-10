import sqlite3
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
import random


class DatabaseHandler(ABC):
    """Abstract base class for database operations"""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    @abstractmethod
    def create_table(self, table_name: str, columns: List[Dict]):
        pass
    
    @abstractmethod
    def insert_data(self, table_name: str, data: List[Dict]):
        pass
    
    @abstractmethod
    def get_existing_tables(self) -> set:
        pass
    
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        pass
    
    @abstractmethod
    def get_pk_values(self, table_name: str, pk_column: str) -> set:
        pass
    
    @abstractmethod
    def get_fk_values(self, parent_table: str, parent_column: str) -> List[Any]:
        pass
    
    @abstractmethod
    def is_pk_used(self, table_name: str, pk_column: str, pk_value: Any) -> bool:
        pass
    
    @abstractmethod
    def is_composite_key_used(self, table_name: str, composite_columns: List[str], composite_values: List[Any]) -> bool:
        pass
    
    @abstractmethod
    def get_next_sequential_pk(self, table_name: str, pk_column: str) -> int:
        pass
    
    @abstractmethod
    def get_table_data(self, table_name: str) -> pd.DataFrame:
        pass
    
    # New methods for enhanced constraint handling
    @abstractmethod
    def is_unique_constraint_violated(self, table_name: str, column_name: str, value: Any) -> bool:
        pass
    
    @abstractmethod
    def is_composite_unique_violated(self, table_name: str, columns: List[str], values: List[Any]) -> bool:
        pass
    
    @abstractmethod
    def get_unique_values(self, table_name: str, column_name: str) -> Set[Any]:
        pass


class SQLiteHandler(DatabaseHandler):
    """SQLite database handler implementation"""
    
    def __init__(self, db_file: str, logger: Optional[logging.Logger] = None):
        self.db_file = db_file
        self.connection = None
        self.logger = logger or logging.getLogger(__name__)
        self._existing_tables = set()
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_file)
            self.connection.row_factory = sqlite3.Row
            self._update_existing_tables()
            return self.connection
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")
    
    def _update_existing_tables(self):
        """Update the set of existing tables in the database"""
        try:
            cursor = self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            self._existing_tables = {table[0] for table in tables}
            self.logger.debug(f"Found existing tables: {self._existing_tables}")
        except Exception as e:
            self.logger.error(f"Error getting existing tables: {e}")
    
    def create_table(self, table_name: str, columns: List[Dict]):
        """Create table with given schema"""
        if not self.connection:
            self.connect()
            
        try:
            # Build CREATE TABLE statement
            column_defs = []
            unique_constraints = []
            
            for col in columns:
                col_name = col['name']
                col_type = col['type'].upper()
                
                # Map data types to SQLite types
                if col_type in ['INT', 'INTEGER']:
                    sql_type = 'INTEGER'
                elif col_type in ['FLOAT', 'DOUBLE', 'DECIMAL']:
                    sql_type = 'REAL'
                elif col_type in ['BOOL', 'BOOLEAN']:
                    sql_type = 'INTEGER'
                elif col_type in ['DATE', 'DATETIME', 'TIMESTAMP']:
                    sql_type = 'TEXT'
                else:
                    sql_type = 'TEXT'
                
                col_def = f'"{col_name}" {sql_type}'
                
                # Add constraints
                constraints = col.get('constraints', []) + col.get('constraint', [])
                if 'PK' in constraints:
                    col_def += ' PRIMARY KEY'
                if 'UNIQUE' in constraints:
                    col_def += ' UNIQUE'
                if not col.get('nullable', True):
                    col_def += ' NOT NULL'
                
                column_defs.append(col_def)
            
            create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(column_defs)})'
            self.connection.execute(create_sql)
            self.connection.commit()
            
            self._existing_tables.add(table_name)
            self.logger.info(f"Created table: {table_name}")
            
        except Exception as e:
            self.logger.error(f"Error creating table {table_name}: {e}")
            raise
    
    def insert_data(self, table_name: str, data: List[Dict]):
        """Insert data into table"""
        if not data or not self.connection:
            return
            
        try:
            # Get column names from first record
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_names = ', '.join([f'"{col}"' for col in columns])
            
            insert_sql = f'INSERT INTO "{table_name}" ({column_names}) VALUES ({placeholders})'
            
            # Convert data to tuples
            rows = []
            for record in data:
                row = tuple(record.get(col) for col in columns)
                rows.append(row)
            
            self.connection.executemany(insert_sql, rows)
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error inserting data into {table_name}: {e}")
            raise
    
    def get_existing_tables(self) -> set:
        """Get set of existing table names"""
        return self._existing_tables.copy()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        return table_name in self._existing_tables
    
    def get_pk_values(self, table_name: str, pk_column: str) -> set:
        """Get existing primary key values"""
        if not self.connection or not self.table_exists(table_name):
            return set()
            
        try:
            cursor = self.connection.execute(
                f'SELECT "{pk_column}" FROM "{table_name}" WHERE "{pk_column}" IS NOT NULL'
            )
            return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            self.logger.error(f"Error getting PK values from {table_name}.{pk_column}: {e}")
            return set()
    
    def get_fk_values(self, parent_table: str, parent_column: str) -> List[Any]:
        """Get foreign key values from parent table"""
        if not self.connection or not self.table_exists(parent_table):
            return []
            
        try:
            cursor = self.connection.execute(
                f'SELECT DISTINCT "{parent_column}" FROM "{parent_table}" WHERE "{parent_column}" IS NOT NULL'
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting FK values from {parent_table}.{parent_column}: {e}")
            return []
    
    def is_pk_used(self, table_name: str, pk_column: str, pk_value: Any) -> bool:
        """Check if primary key value is already used"""
        if not self.connection or not self.table_exists(table_name):
            return False
            
        try:
            cursor = self.connection.execute(
                f'SELECT 1 FROM "{table_name}" WHERE "{pk_column}" = ? LIMIT 1',
                (pk_value,)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Error checking PK usage in {table_name}.{pk_column}: {e}")
            return False
    
    def is_composite_key_used(self, table_name: str, composite_columns: List[str], composite_values: List[Any]) -> bool:
        """Check if composite key is already used"""
        if not self.connection or not self.table_exists(table_name):
            return False
            
        try:
            where_conditions = []
            params = []
            for col, val in zip(composite_columns, composite_values):
                where_conditions.append(f'"{col}" = ?')
                params.append(val)
            
            where_clause = ' AND '.join(where_conditions)
            query = f'SELECT 1 FROM "{table_name}" WHERE {where_clause} LIMIT 1'
            
            cursor = self.connection.execute(query, params)
            return cursor.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Error checking composite key usage in {table_name}: {e}")
            return False
    
    def get_next_sequential_pk(self, table_name: str, pk_column: str) -> int:
        """Get next sequential primary key value"""
        if not self.connection or not self.table_exists(table_name):
            return 1
            
        try:
            cursor = self.connection.execute(
                f'SELECT MAX("{pk_column}") FROM "{table_name}" WHERE "{pk_column}" IS NOT NULL'
            )
            result = cursor.fetchone()
            max_value = result[0] if result and result[0] is not None else 0
            return max_value + 1
        except Exception as e:
            self.logger.error(f"Error getting next sequential PK from {table_name}.{pk_column}: {e}")
            return 1
    
    def get_table_data(self, table_name: str) -> pd.DataFrame:
        """Get all data from table as DataFrame"""
        if not self.connection or not self.table_exists(table_name):
            return pd.DataFrame()
            
        try:
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', self.connection)
        except Exception as e:
            self.logger.error(f"Error reading data from {table_name}: {e}")
            return pd.DataFrame()
    
    def is_unique_constraint_violated(self, table_name: str, column_name: str, value: Any) -> bool:
        """Check if unique constraint would be violated"""
        if not self.connection or not self.table_exists(table_name):
            return False
            
        try:
            cursor = self.connection.execute(
                f'SELECT 1 FROM "{table_name}" WHERE "{column_name}" = ? LIMIT 1',
                (value,)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            self.logger.error(f"Error checking unique constraint in {table_name}.{column_name}: {e}")
            return False
    
    def is_composite_unique_violated(self, table_name: str, columns: List[str], values: List[Any]) -> bool:
        """Check if composite unique constraint would be violated"""
        return self.is_composite_key_used(table_name, columns, values)
    
    def get_unique_values(self, table_name: str, column_name: str) -> Set[Any]:
        """Get all unique values from a column"""
        if not self.connection or not self.table_exists(table_name):
            return set()
            
        try:
            cursor = self.connection.execute(
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL'
            )
            return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            self.logger.error(f"Error getting unique values from {table_name}.{column_name}: {e}")
            return set()

class PandasHandler(DatabaseHandler):
    """Enhanced in-memory pandas DataFrame handler implementation with full constraint support"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.tables = {}  # Dictionary to store DataFrames
        self.table_schemas = {}  # Store table schema information
        self.logger = logger or logging.getLogger(__name__)
        self._pk_cache = {}  # Cache for primary key values
        self._unique_cache = {}  # Cache for unique constraint values
        self._fk_cache = {}  # Cache for foreign key values
        self._cache_size_limit = 10000
        self._primary_keys = {}  # table_name -> [pk_columns]
        self._unique_constraints = {}  # table_name -> [unique_columns]
        self._composite_unique = {}  # table_name -> [[col1, col2], [col3, col4]]
        self._foreign_keys = {}  # table_name -> [(child_col, parent_table, parent_col)]
        self._check_constraints = {}  # table_name -> [check_functions]
        self._not_null_constraints = {}  # table_name -> [not_null_columns]
        
        self._indexes = {}  # table_name -> {column_name: index_dict}
    
    def create_table(self, table_name: str, columns: List[Dict]):
        """Create empty DataFrame with specified schema and comprehensive constraint tracking"""
        try:
            # Store schema information
            self.table_schemas[table_name] = columns
            
            # Create empty DataFrame with column names and types
            column_names = [col['name'] for col in columns]
            self.tables[table_name] = pd.DataFrame(columns=column_names)
            
            # Initialize constraint tracking
            self._primary_keys[table_name] = []
            self._unique_constraints[table_name] = []
            self._composite_unique[table_name] = []
            self._foreign_keys[table_name] = []
            self._check_constraints[table_name] = []
            self._not_null_constraints[table_name] = []
            
            # Set appropriate dtypes and extract constraints
            dtype_map = {}
            for col in columns:
                col_name = col['name']
                col_type = col['type'].lower()
                constraints = col.get('constraints', []) + col.get('constraint', [])
                
                # Set data type with nullable support
                if col_type in ['int', 'integer']:
                    dtype_map[col_name] = 'Int64'  # Nullable integer
                elif col_type in ['float', 'double', 'decimal']:
                    dtype_map[col_name] = 'float64'
                elif col_type in ['bool', 'boolean']:
                    dtype_map[col_name] = 'boolean'
                elif col_type in ['date']:
                    dtype_map[col_name] = 'datetime64[ns]'
                elif col_type in ['datetime', 'timestamp']:
                    dtype_map[col_name] = 'datetime64[ns]'
                else:
                    dtype_map[col_name] = 'object'
                
                # Track constraints
                if 'PK' in constraints:
                    self._primary_keys[table_name].append(col_name)
                if 'UNIQUE' in constraints:
                    self._unique_constraints[table_name].append(col_name)
                if not col.get('nullable', True):
                    self._not_null_constraints[table_name].append(col_name)
            
            # Handle composite primary keys from table metadata
            composite_pk = getattr(columns, 'composite_primary_key', None)
            if composite_pk:
                self._primary_keys[table_name] = composite_pk
            
            self.tables[table_name] = self.tables[table_name].astype(dtype_map)
            self.logger.info(f"Created DataFrame table: {table_name} with constraints")
            self.logger.debug(f"Primary keys: {self._primary_keys[table_name]}")
            self.logger.debug(f"Unique constraints: {self._unique_constraints[table_name]}")
            self.logger.debug(f"Not null constraints: {self._not_null_constraints[table_name]}")
            
        except Exception as e:
            self.logger.error(f"Error creating DataFrame table {table_name}: {e}")
            raise
    
    def add_check_constraint(self, table_name: str, constraint_name: str, check_function):
        """Add custom check constraint"""
        if table_name in self._check_constraints:
            self._check_constraints[table_name].append({
                'name': constraint_name,
                'function': check_function
            })
            self.logger.debug(f"Added check constraint '{constraint_name}' to {table_name}")
    
    def _validate_constraints_before_insert(self, table_name: str, data: List[Dict]) -> List[Dict]:
        """Validate all constraints before inserting data"""
        if table_name not in self.tables:
            return data
        
        valid_records = []
        
        for record in data:
            try:
                # Check NOT NULL constraints
                if self._validate_not_null_constraints(table_name, record):
                    # Check primary key constraints
                    if self._validate_primary_key_constraint(table_name, record):
                        # Check unique constraints
                        if self._validate_unique_constraints(table_name, record):
                            # Check foreign key constraints
                            if self._validate_foreign_key_constraints(table_name, record):
                                # Check custom constraints
                                if self._validate_check_constraints(table_name, record):
                                    valid_records.append(record)
                                else:
                                    self.logger.warning(f"Check constraint violation for record: {record}")
                            else:
                                self.logger.warning(f"Foreign key constraint violation for record: {record}")
                        else:
                            self.logger.warning(f"Unique constraint violation for record: {record}")
                    else:
                        self.logger.warning(f"Primary key constraint violation for record: {record}")
                else:
                    self.logger.warning(f"NOT NULL constraint violation for record: {record}")
            except Exception as e:
                self.logger.error(f"Error validating record {record}: {e}")
                continue
        
        return valid_records
    
    def _validate_not_null_constraints(self, table_name: str, record: Dict) -> bool:
        """Validate NOT NULL constraints for a record"""
        not_null_columns = self._not_null_constraints.get(table_name, [])
        
        for col in not_null_columns:
            if col not in record or record[col] is None or pd.isna(record[col]):
                self.logger.debug(f"NOT NULL constraint violated for column {col}")
                return False
        
        return True
    
    def _validate_check_constraints(self, table_name: str, record: Dict) -> bool:
        """Validate custom check constraints for a record"""
        check_constraints = self._check_constraints.get(table_name, [])
        
        for constraint in check_constraints:
            try:
                if not constraint['function'](record):
                    self.logger.debug(f"Check constraint '{constraint['name']}' violated")
                    return False
            except Exception as e:
                self.logger.error(f"Error executing check constraint '{constraint['name']}': {e}")
                return False
        
        return True
    
    def get_constraint_violations(self, table_name: str, record: Dict) -> List[str]:
        """Get detailed list of constraint violations for a record"""
        violations = []
        
        if not self._validate_not_null_constraints(table_name, record):
            not_null_cols = self._not_null_constraints.get(table_name, [])
            for col in not_null_cols:
                if col not in record or record[col] is None or pd.isna(record[col]):
                    violations.append(f"NOT NULL constraint violated for column '{col}'")
        
        if not self._validate_primary_key_constraint(table_name, record):
            pk_cols = self._primary_keys.get(table_name, [])
            violations.append(f"Primary key constraint violated for columns: {pk_cols}")
        
        if not self._validate_unique_constraints(table_name, record):
            unique_cols = self._unique_constraints.get(table_name, [])
            for col in unique_cols:
                if col in record and self.is_unique_constraint_violated(table_name, col, record[col]):
                    violations.append(f"Unique constraint violated for column '{col}' with value '{record[col]}'")
        
        if not self._validate_foreign_key_constraints(table_name, record):
            fk_constraints = self._foreign_keys.get(table_name, [])
            for child_col, parent_table, parent_col in fk_constraints:
                if child_col in record and record[child_col] is not None:
                    parent_values = self.get_fk_values(parent_table, parent_col)
                    if record[child_col] not in parent_values:
                        violations.append(f"Foreign key constraint violated: {child_col}='{record[child_col]}' not found in {parent_table}.{parent_col}")
        
        return violations
    
    def insert_data_with_validation_report(self, table_name: str, data: List[Dict]) -> Dict:
        """Insert data and return detailed validation report"""
        if not data:
            return {'inserted': 0, 'rejected': 0, 'violations': []}
        
        try:
            validation_report = {
                'inserted': 0,
                'rejected': 0,
                'violations': []
            }
            
            valid_records = []
            
            for i, record in enumerate(data):
                violations = self.get_constraint_violations(table_name, record)
                if not violations:
                    valid_records.append(record)
                    validation_report['inserted'] += 1
                else:
                    validation_report['rejected'] += 1
                    validation_report['violations'].append({
                        'record_index': i,
                        'record': record,
                        'violations': violations
                    })
            
            if valid_records:
                new_df = pd.DataFrame(valid_records)
                
                if table_name not in self.tables:
                    self.tables[table_name] = new_df
                else:
                    self.tables[table_name] = pd.concat([self.tables[table_name], new_df], ignore_index=True)
                
                # Clear relevant caches
                self._clear_table_caches(table_name)
            
            self.logger.info(f"Insertion report for {table_name}: {validation_report['inserted']} inserted, {validation_report['rejected']} rejected")
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error inserting data into DataFrame table {table_name}: {e}")
            raise
    
    def create_index(self, table_name: str, column_names: List[str], index_name: str = None):
        """Create index for faster lookups (conceptual - pandas doesn't have real indexes)"""
        if table_name not in self.tables:
            return
        
        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"
        
        if table_name not in self._indexes:
            self._indexes[table_name] = {}
        
        # Store index metadata (in real implementation, this could create actual index structures)
        self._indexes[table_name][index_name] = {
            'columns': column_names,
            'created_at': datetime.now()
        }
        
        self.logger.info(f"Created index '{index_name}' on {table_name}({', '.join(column_names)})")
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get comprehensive table information"""
        if table_name not in self.tables:
            return {}
        
        df = self.tables[table_name]
        
        return {
            'table_name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'constraints': self.get_constraint_info(table_name),
            'indexes': self._indexes.get(table_name, {}),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict()
        }
    
    def validate_referential_integrity(self) -> Dict[str, List[str]]:
        """Validate referential integrity across all tables"""
        violations = defaultdict(list)
        
        for table_name, fk_constraints in self._foreign_keys.items():
            if table_name not in self.tables:
                continue
                
            df = self.tables[table_name]
            
            for child_col, parent_table, parent_col in fk_constraints:
                if parent_table not in self.tables:
                    violations[table_name].append(f"Referenced table '{parent_table}' does not exist")
                    continue
                
                if child_col not in df.columns:
                    violations[table_name].append(f"Foreign key column '{child_col}' does not exist")
                    continue
                
                parent_df = self.tables[parent_table]
                if parent_col not in parent_df.columns:
                    violations[table_name].append(f"Referenced column '{parent_col}' does not exist in '{parent_table}'")
                    continue
                
                # Check for orphaned records
                child_values = set(df[child_col].dropna().unique())
                parent_values = set(parent_df[parent_col].dropna().unique())
                
                orphaned = child_values - parent_values
                if orphaned:
                    violations[table_name].append(f"Orphaned foreign key values in '{child_col}': {list(orphaned)}")
        
        return dict(violations)
    
    def optimize_table(self, table_name: str):
        """Optimize table by removing duplicates and reorganizing data"""
        if table_name not in self.tables:
            return
        
        df = self.tables[table_name]
        original_size = len(df)
        
        # Remove duplicates based on primary key if exists
        pk_cols = self._primary_keys.get(table_name, [])
        if pk_cols:
            df = df.drop_duplicates(subset=pk_cols, keep='first')
        
        # Optimize data types
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert string columns to categorical if they have few unique values
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        self.tables[table_name] = df
        
        # Clear caches for this table
        self._clear_table_caches(table_name)
        
        optimized_size = len(df)
        self.logger.info(f"Optimized table '{table_name}': {original_size} -> {optimized_size} rows")


# Additional utility functions for the DataGenerator class
class DataGeneratorEnhancements:
    """Additional methods to enhance DataGenerator for better constraint handling"""
    
    @staticmethod
    def validate_composite_unique_constraint(db_handler, table_name: str, record: Dict, 
                                           composite_unique_groups: List[List[str]]) -> bool:
        """Validate composite unique constraints"""
        for col_group in composite_unique_groups:
            values = [record.get(col) for col in col_group if col in record]
            if len(values) == len(col_group) and all(v is not None for v in values):
                if db_handler.is_composite_unique_violated(table_name, col_group, values):
                    return False
        return True
    
    @staticmethod
    def generate_unique_value_with_retries(generator_func, validator_func, max_retries: int = 100):
        """Generate unique value with retry logic"""
        for attempt in range(max_retries):
            value = generator_func()
            if not validator_func(value):
                return value
        
        raise ValueError(f"Could not generate unique value after {max_retries} attempts")
    
    @staticmethod
    def create_constraint_validator(table_metadata: Dict):
        """Create a comprehensive constraint validator function"""
        def validate_record(record: Dict, db_handler) -> Tuple[bool, List[str]]:
            violations = []
            table_name = table_metadata.get('table_name', '')
            
            # Get constraint violations from the database handler
            if hasattr(db_handler, 'get_constraint_violations'):
                violations = db_handler.get_constraint_violations(table_name, record)
                return len(violations) == 0, violations
            
            # Fallback validation logic
            return True, []
        
        return validate_record


# Enhanced DataGenerator integration methods
class DataGeneratorDatabaseIntegration:
    """Integration methods for DataGenerator to work with enhanced database constraints"""
    
    def __init__(self, data_generator, db_handler):
        self.data_generator = data_generator
        self.db_handler = db_handler
        self.logger = data_generator.logger
    
    def generate_with_constraint_validation(self, table_metadata: Dict, record_count: int, 
                                          foreign_key_data: Dict = None) -> List[Dict]:
        """Generate data with full constraint validation"""
        table_name = table_metadata.get('table_name', '')
        generated_records = []
        failed_attempts = 0
        max_failed_attempts = record_count * 10  # Allow some failures
        
        while len(generated_records) < record_count and failed_attempts < max_failed_attempts:
            try:
                # Generate a single record
                record_generator = self.data_generator.generate_row(
                    table_metadata, 1, foreign_key_data
                )
                record = next(record_generator)
                
                # Validate against all constraints
                if hasattr(self.db_handler, 'get_constraint_violations'):
                    violations = self.db_handler.get_constraint_violations(table_name, record)
                    if not violations:
                        generated_records.append(record)
                    else:
                        failed_attempts += 1
                        self.logger.debug(f"Record rejected due to violations: {violations}")
                else:
                    # Fallback validation
                    generated_records.append(record)
                
            except Exception as e:
                failed_attempts += 1
                self.logger.warning(f"Failed to generate record: {e}")
        
        success_rate = len(generated_records) / (len(generated_records) + failed_attempts) * 100
        self.logger.info(f"Generated {len(generated_records)} records with {success_rate:.1f}% success rate")
        
        return generated_records
    
    def bulk_insert_with_validation(self, table_name: str, records: List[Dict]) -> Dict:
        """Bulk insert with detailed validation reporting"""
        if hasattr(self.db_handler, 'insert_data_with_validation_report'):
            return self.db_handler.insert_data_with_validation_report(table_name, records)
        else:
            # Fallback for SQLite handler
            try:
                self.db_handler.insert_data(table_name, records)
                return {
                    'inserted': len(records),
                    'rejected': 0,
                    'violations': []
                }
            except Exception as e:
                self.logger.error(f"Bulk insert failed: {e}")
                return {
                    'inserted': 0,
                    'rejected': len(records),
                    'violations': [{'error': str(e)}]
                }
    
    def generate_foreign_key_data(self, schema: Dict) -> Dict:
        """Generate foreign key lookup data from existing tables"""
        fk_data = {}
        
        for table_name, table_meta in schema.items():
            if not self.db_handler.table_exists(table_name):
                continue
                
            # Get all columns that could be referenced as foreign keys
            columns = table_meta.get('columns', [])
            pk_columns = [col for col in columns if 'PK' in col.get('constraints', [])]
            unique_columns = [col for col in columns if 'UNIQUE' in col.get('constraints', [])]
            
            for col in pk_columns + unique_columns:
                col_name = col['name']
                values = list(self.db_handler.get_unique_values(table_name, col_name))
                if values:
                    fk_data[f"{table_name}.{col_name}"] = values
        
        return fk_data


def setup_enhanced_database_handler(use_sqlite: bool = False, db_file: str = None):
    """Setup database handler with enhanced constraint support"""
    if use_sqlite:
        handler = SQLiteHandler(db_file)
    else:
        handler = PandasHandler()
   
    handler.connect()
    
    return handler