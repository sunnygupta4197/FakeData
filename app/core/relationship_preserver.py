import random
import logging
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict, deque
import pandas as pd


class RelationshipPreserver:
    """
    Handles foreign key relationships and referential integrity 
    for synthetic data generation.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or self._setup_default_logger()
        
        # FK relationship tracking
        self._fk_pools = defaultdict(list)  # table.column -> list of available values
        self._dependency_graph = {}  # table -> list of dependent tables
        self._reverse_dependencies = defaultdict(list)  # table -> list of parent tables
        
        # Performance optimizations
        self._cache_size_limit = 50000
        self._batch_fk_refresh_threshold = 10000
        
    def _setup_default_logger(self):
        """Setup default logger if none provided"""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        return logging.getLogger(__name__)
    
    def build_dependency_graph(self, tables_metadata: List[Dict]) -> Dict[str, List[str]]:
        """
        Build dependency graph to determine table generation order.
        Returns dictionary with table -> [dependent_tables] mapping.
        """
        self._dependency_graph = {}
        self._reverse_dependencies = defaultdict(list)
        
        # Initialize all tables in the graph
        for table in tables_metadata:
            table_name = table.get('table_name')
            if table_name:
                self._dependency_graph[table_name] = []
        
        # Build dependencies based on foreign keys
        for table in tables_metadata:
            table_name = table.get('table_name')
            if not table_name:
                continue
                
            foreign_keys = table.get('foreign_keys', [])
            for fk in foreign_keys:
                parent_table = fk.get('parent_table')
                if parent_table and parent_table in self._dependency_graph:
                    # parent_table must be generated before table_name
                    if table_name not in self._dependency_graph[parent_table]:
                        self._dependency_graph[parent_table].append(table_name)
                    
                    # Track reverse dependencies
                    if parent_table not in self._reverse_dependencies[table_name]:
                        self._reverse_dependencies[table_name].append(parent_table)
        
        self.logger.info(f"Built dependency graph for {len(self._dependency_graph)} tables")
        return self._dependency_graph
    
    def get_generation_order(self, table_names: List[str]) -> List[str]:
        """
        Get the correct order for table generation based on FK dependencies.
        Uses topological sort to ensure parent tables are generated before children.
        """
        if not self._dependency_graph:
            self.logger.warning("Dependency graph not built. Call build_dependency_graph() first.")
            return table_names
        
        # Filter dependency graph to only include requested tables
        filtered_graph = {
            table: [dep for dep in deps if dep in table_names]
            for table, deps in self._dependency_graph.items()
            if table in table_names
        }
        
        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for table in table_names:
            in_degree[table] = 0
        
        for table, dependencies in filtered_graph.items():
            for dep in dependencies:
                in_degree[dep] += 1
        
        # Initialize queue with tables having no dependencies
        queue = deque([table for table in table_names if in_degree[table] == 0])
        generation_order = []
        
        while queue:
            current_table = queue.popleft()
            generation_order.append(current_table)
            
            # Update in-degrees of dependent tables
            for dependent in filtered_graph.get(current_table, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(generation_order) != len(table_names):
            remaining_tables = set(table_names) - set(generation_order)
            self.logger.warning(f"Circular dependencies detected for tables: {remaining_tables}")
            # Add remaining tables at the end
            generation_order.extend(remaining_tables)
        
        self.logger.info(f"Generation order determined: {generation_order}")
        return generation_order
    
    def refresh_fk_pools(self, generated_data: Dict[str, pd.DataFrame], 
                        tables_metadata: List[Dict], table_name: str = None):
        """
        Refresh foreign key pools from generated DataFrame data.
        """
        tables_to_refresh = [table_name] if table_name else list(generated_data.keys())
        
        for table in tables_to_refresh:
            if table not in generated_data or generated_data[table].empty:
                continue
            
            df = generated_data[table]
            table_metadata = self._get_table_metadata(tables_metadata, table)
            if not table_metadata:
                continue
            
            # Get primary key columns for this table
            pk_columns = self._get_primary_key_columns(table_metadata)
            
            # Update FK pools for each PK column
            for pk_col in pk_columns:
                if pk_col in df.columns:
                    fk_key = f"{table}.{pk_col}"
                    # Use pandas for efficient unique value extraction
                    unique_values = df[pk_col].dropna().unique().tolist()
                    
                    # Limit cache size for performance
                    if len(unique_values) > self._cache_size_limit:
                        unique_values = random.sample(unique_values, self._cache_size_limit)
                    
                    self._fk_pools[fk_key] = unique_values
                    self.logger.debug(f"Refreshed FK pool {fk_key} with {len(unique_values)} values")
    
    def get_fk_values(self, parent_table: str, parent_column: str, 
                     expected_data_type: str = None, sample_size: int = None) -> List[Any]:
        """
        Get available foreign key values with type conversion and sampling.
        """
        fk_key = f"{parent_table}.{parent_column}"
        
        # Get cached values
        available_values = self._fk_pools.get(fk_key, [])
        
        if not available_values:
            self.logger.warning(f"No FK values available for {fk_key}")
            return []
        
        # Convert to expected data type if specified
        if expected_data_type:
            available_values = self._convert_values_to_type(available_values, expected_data_type)
        
        # Sample if too many values (for performance)
        if sample_size and len(available_values) > sample_size:
            available_values = random.sample(available_values, sample_size)
        
        self.logger.debug(f"Retrieved {len(available_values)} FK values from {parent_table}.{parent_column}")
        return available_values
    
    def maintain_referential_integrity(self, df_map: Dict[str, pd.DataFrame], 
                                     tables_metadata: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Ensure referential integrity across all generated DataFrames.
        """
        for table_name, df in df_map.items():
            if df.empty:
                continue
                
            table_metadata = self._get_table_metadata(tables_metadata, table_name)
            if not table_metadata:
                continue
            
            foreign_keys = table_metadata.get('foreign_keys', [])
            for fk in foreign_keys:
                parent_table = fk.get('parent_table')
                parent_column = fk.get('parent_column')
                child_column = fk.get('child_column')
                
                if not all([parent_table, parent_column, child_column]):
                    continue
                
                if parent_table in df_map and not df_map[parent_table].empty:
                    parent_df = df_map[parent_table]
                    
                    if parent_column in parent_df.columns and child_column in df.columns:
                        # Get available parent values
                        available_values = parent_df[parent_column].dropna().values
                        
                        if len(available_values) > 0:
                            # Assign FK values using random selection
                            df_map[table_name][child_column] = pd.Series(
                                random.choices(available_values, k=len(df))
                            )
                            self.logger.debug(
                                f"Updated FK relationship: {table_name}.{child_column} -> "
                                f"{parent_table}.{parent_column}"
                            )
        
        return df_map
    
    def validate_relationships(self, df_map: Dict[str, pd.DataFrame], 
                              tables_metadata: List[Dict]) -> Dict[str, Any]:
        """
        Validate referential integrity and return validation report.
        """
        validation_report = {
            'valid': True,
            'violations': [],
            'statistics': {}
        }
        
        for table_name, df in df_map.items():
            if df.empty:
                continue
                
            table_metadata = self._get_table_metadata(tables_metadata, table_name)
            if not table_metadata:
                continue
            
            foreign_keys = table_metadata.get('foreign_keys', [])
            for fk in foreign_keys:
                parent_table = fk.get('parent_table')
                parent_column = fk.get('parent_column')
                child_column = fk.get('child_column')
                
                if not all([parent_table, parent_column, child_column]):
                    continue
                
                if parent_table not in df_map or child_column not in df.columns:
                    continue
                
                parent_df = df_map[parent_table]
                if parent_column not in parent_df.columns:
                    continue
                
                # Check for orphaned records
                child_values = set(df[child_column].dropna())
                parent_values = set(parent_df[parent_column].dropna())
                
                orphaned_values = child_values - parent_values
                if orphaned_values:
                    validation_report['valid'] = False
                    validation_report['violations'].append({
                        'type': 'orphaned_records',
                        'child_table': table_name,
                        'child_column': child_column,
                        'parent_table': parent_table,
                        'parent_column': parent_column,
                        'orphaned_count': len(orphaned_values),
                        'sample_orphaned_values': list(orphaned_values)[:5]
                    })
                
                # Statistics
                fk_name = f"{table_name}.{child_column} -> {parent_table}.{parent_column}"
                validation_report['statistics'][fk_name] = {
                    'child_unique_values': len(child_values),
                    'parent_unique_values': len(parent_values),
                    'orphaned_count': len(orphaned_values),
                    'integrity_percentage': (1 - len(orphaned_values) / max(len(child_values), 1)) * 100
                }
        
        self.logger.info(f"Relationship validation completed. Valid: {validation_report['valid']}")
        return validation_report
    
    def get_fk_pools_status(self) -> Dict[str, Any]:
        """
        Get status information about current FK pools.
        """
        status = {
            'total_pools': len(self._fk_pools),
            'pools': {}
        }
        
        for fk_key, values in self._fk_pools.items():
            status['pools'][fk_key] = {
                'available_values': len(values),
                'sample_values': values[:5] if values else []
            }
        
        return status
    
    def clear_fk_pools(self, table_name: str = None):
        """
        Clear FK pools for a specific table or all tables.
        """
        if table_name:
            keys_to_remove = [key for key in self._fk_pools.keys() if key.startswith(f"{table_name}.")]
            for key in keys_to_remove:
                del self._fk_pools[key]
            self.logger.info(f"Cleared FK pools for table: {table_name}")
        else:
            self._fk_pools.clear()
            self.logger.info("Cleared all FK pools")
    
    def _get_table_metadata(self, tables_metadata: List[Dict], table_name: str) -> Dict:
        """Get table metadata by name"""
        for table in tables_metadata:
            if table.get('table_name') == table_name:
                return table
        return {}
    
    def _get_primary_key_columns(self, table_metadata: Dict) -> List[str]:
        """Get primary key columns from table metadata"""
        # Check for composite primary key first
        composite_pk = table_metadata.get("composite_primary_key", [])
        if composite_pk:
            return composite_pk
        
        # Look for individual primary key columns
        pk_columns = []
        for column in table_metadata.get("columns", []):
            constraints = column.get("constraints", [])
            constraint = column.get("constraint", [])  # Alternative field name
            
            if "PK" in constraints or "PK" in constraint:
                pk_columns.append(column["name"])
        
        return pk_columns
    
    def _convert_values_to_type(self, values: List[Any], data_type: str) -> List[Any]:
        """Convert list of values to expected data type"""
        converted_values = []
        
        for value in values:
            try:
                if data_type.lower() in ['int', 'integer']:
                    converted_values.append(int(value))
                elif data_type.lower() in ['float', 'double', 'decimal']:
                    converted_values.append(float(value))
                elif data_type.lower() in ['bool', 'boolean']:
                    converted_values.append(str(value).lower() in ('true', '1', 'yes', 'on'))
                else:  # string types
                    converted_values.append(str(value))
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Could not convert value '{value}' to type '{data_type}': {e}")
                converted_values.append(value)
        
        return converted_values