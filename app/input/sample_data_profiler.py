import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, date
import re
from collections import Counter, defaultdict
import json
from pathlib import Path

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Optional: For advanced profiling
try:
    import pandas_profiling
    PANDAS_PROFILING_AVAILABLE = True
except ImportError:
    PANDAS_PROFILING_AVAILABLE = False

try:
    import sweetviz as sv
    SWEETVIZ_AVAILABLE = True
except ImportError:
    SWEETVIZ_AVAILABLE = False


class SampleDataProfiler:
    """
    Analyze existing data to infer generation rules and patterns
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()
        self.profile_results = {}
        self.generation_rules = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def load_data(self, data_source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Load data from various sources
        
        Args:
            data_source: File path, URL, or pandas DataFrame
            **kwargs: Additional parameters for pandas read functions
        
        Returns:
            pandas DataFrame
        """
        if isinstance(data_source, pd.DataFrame):
            return data_source
        
        file_path = Path(data_source)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data source not found: {data_source}")
        
        # Determine file type and load accordingly
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                df = pd.read_csv(data_source, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(data_source, **kwargs)
            elif suffix == '.json':
                df = pd.read_json(data_source, **kwargs)
            elif suffix == '.parquet':
                df = pd.read_parquet(data_source, **kwargs)
            elif suffix in ['.tsv', '.txt']:
                df = pd.read_csv(data_source, sep='\t', **kwargs)
            else:
                # Try CSV as default
                df = pd.read_csv(data_source, **kwargs)
                
            self.logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {data_source}: {e}")
            raise
    
    def profile_dataset(self, df: pd.DataFrame, 
                       table_name: str = "unknown_table",
                       sample_size: Optional[int] = None,
                       correlation_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Comprehensive profiling of the dataset
        
        Args:
            df: Input DataFrame
            table_name: Name of the table being profiled
            sample_size: Sample size for large datasets (None = use all data)
            correlation_threshold: Threshold for detecting correlations
        
        Returns:
            Dictionary containing profiling results
        """
        self.logger.info(f"Starting profiling for table: {table_name}")
        
        # Sample data if needed
        if sample_size and len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Using sample of {sample_size} rows for profiling")
        else:
            df_sample = df
        
        profile = {
            'table_name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'profiling_timestamp': datetime.now().isoformat(),
            'columns': {},
            'relationships': {},
            'data_quality': {},
            'patterns': {}
        }
        
        # Profile each column
        for column in df_sample.columns:
            self.logger.debug(f"Profiling column: {column}")
            profile['columns'][column] = self._profile_column(df_sample[column], column)
        
        # Analyze relationships between columns
        profile['relationships'] = self._analyze_relationships(df_sample, correlation_threshold)
        
        # Data quality analysis
        profile['data_quality'] = self._analyze_data_quality(df_sample)
        
        # Pattern detection
        profile['patterns'] = self._detect_patterns(df_sample)
        
        self.profile_results[table_name] = profile
        self.logger.info(f"Profiling completed for table: {table_name}")
        
        return profile
    
    def _profile_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Profile individual column"""
        column_profile = {
            'name': column_name,
            'dtype': str(series.dtype),
            'inferred_type': self._infer_data_type(series),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'uniqueness_percentage': (series.nunique() / len(series)) * 100,
            'constraints': [],
            'generation_hints': {}
        }
        
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            column_profile['generation_hints'] = {'type': 'null', 'probability': 1.0}
            return column_profile
        
        # Detect constraints
        column_profile['constraints'] = self._detect_constraints(non_null_series)
        
        # Type-specific analysis
        inferred_type = column_profile['inferred_type']
        
        if inferred_type in ['int', 'float']:
            column_profile.update(self._analyze_numeric_column(non_null_series))
        elif inferred_type == 'datetime':
            column_profile.update(self._analyze_datetime_column(non_null_series))
        elif inferred_type == 'bool':
            column_profile.update(self._analyze_boolean_column(non_null_series))
        else:  # string/categorical
            column_profile.update(self._analyze_categorical_column(non_null_series))
        
        # Generate generation hints
        column_profile['generation_hints'] = self._generate_column_hints(
            non_null_series, column_profile
        )
        
        return column_profile
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer the most appropriate data type"""
        # Remove nulls for type inference
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return 'unknown'
        
        # Check for boolean
        if series.dtype == 'bool' or non_null.isin([True, False, 0, 1]).all():
            return 'bool'
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        
        # Try to parse as datetime
        if series.dtype == 'object':
            try:
                pd.to_datetime(non_null.head(100), errors='raise')
                return 'datetime'
            except:
                pass
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                return 'int'
            else:
                return 'float'
        
        # Try to convert to numeric
        if series.dtype == 'object':
            try:
                numeric_series = pd.to_numeric(non_null, errors='raise')
                if (numeric_series % 1 == 0).all():
                    return 'int'
                else:
                    return 'float'
            except:
                pass
        
        # Default to string/categorical
        return 'str'
    
    def _detect_constraints(self, series: pd.Series) -> List[str]:
        """Detect constraints on the column"""
        constraints = []
        
        # Check for uniqueness
        if series.nunique() == len(series):
            constraints.append('unique')
        
        # Check for primary key patterns
        if (series.nunique() == len(series) and 
            ('id' in series.name.lower() or series.name.lower().endswith('_id'))):
            constraints.append('primary_key')
        
        # Check for foreign key patterns
        if ('id' in series.name.lower() and 
            series.name.lower() != 'id' and 
            '_id' in series.name.lower()):
            constraints.append('foreign_key')
        
        # Check for not null (if all values are present)
        if series.isnull().sum() == 0:
            constraints.append('not_null')
        
        return constraints
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column"""
        return {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75)),
            'distribution': self._detect_distribution(series),
            'outliers': self._detect_outliers(series)
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column"""
        try:
            dt_series = pd.to_datetime(series)
            return {
                'min_date': dt_series.min().isoformat(),
                'max_date': dt_series.max().isoformat(),
                'date_range_days': (dt_series.max() - dt_series.min()).days,
                'most_common_year': dt_series.dt.year.mode().iloc[0] if len(dt_series.dt.year.mode()) > 0 else None,
                'most_common_month': dt_series.dt.month.mode().iloc[0] if len(dt_series.dt.month.mode()) > 0 else None,
                'weekday_distribution': dt_series.dt.day_name().value_counts().to_dict()
            }
        except:
            return {'error': 'Could not parse datetime'}
    
    def _analyze_boolean_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze boolean column"""
        value_counts = series.value_counts()
        return {
            'true_count': int(value_counts.get(True, 0)),
            'false_count': int(value_counts.get(False, 0)),
            'true_percentage': float((value_counts.get(True, 0) / len(series)) * 100)
        }
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical/string column"""
        value_counts = series.value_counts()
        
        analysis = {
            'top_values': value_counts.head(10).to_dict(),
            'value_distribution': value_counts.to_dict(),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'avg_length': float(series.astype(str).str.len().mean()),
            'min_length': int(series.astype(str).str.len().min()),
            'max_length': int(series.astype(str).str.len().max()),
            'patterns': self._detect_string_patterns(series)
        }
        
        # Check if it's a low-cardinality categorical
        if series.nunique() / len(series) < 0.1:  # Less than 10% unique values
            analysis['is_categorical'] = True
            analysis['categories'] = list(value_counts.index)
        else:
            analysis['is_categorical'] = False
        
        return analysis
    
    def _detect_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Detect statistical distribution of numeric data"""
        try:
            # Test for normal distribution
            _, p_normal = stats.normaltest(series)
            
            # Test for uniform distribution
            _, p_uniform = stats.kstest(series, 'uniform')
            
            # Test for exponential distribution
            _, p_exp = stats.kstest(series, 'expon')
            
            return {
                'normal_p_value': float(p_normal),
                'uniform_p_value': float(p_uniform),
                'exponential_p_value': float(p_exp),
                'likely_distribution': 'normal' if p_normal > 0.05 else 'non_normal',
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series))
            }
        except:
            return {'error': 'Could not determine distribution'}
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            return {
                'outlier_count': len(outliers),
                'outlier_percentage': float((len(outliers) / len(series)) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_values': outliers.tolist()[:10]  # First 10 outliers
            }
        except:
            return {'error': 'Could not detect outliers'}
    
    def _detect_string_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Detect common string patterns"""
        patterns = {
            'email': 0,
            'phone': 0,
            'uuid': 0,
            'url': 0,
            'numeric': 0,
            'alphanumeric': 0,
            'alpha_only': 0,
            'mixed_case': 0,
            'uppercase': 0,
            'lowercase': 0
        }
        
        # Regular expressions for pattern detection
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        phone_pattern = re.compile(r'^[\+]?[1-9]?[\d\s\-\(\)]{7,15}$')
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
        url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.I)
        
        str_series = series.astype(str)
        
        for value in str_series:
            if email_pattern.match(value):
                patterns['email'] += 1
            elif phone_pattern.match(value):
                patterns['phone'] += 1
            elif uuid_pattern.match(value):
                patterns['uuid'] += 1
            elif url_pattern.match(value):
                patterns['url'] += 1
            elif value.isdigit():
                patterns['numeric'] += 1
            elif value.isalnum():
                patterns['alphanumeric'] += 1
            elif value.isalpha():
                patterns['alpha_only'] += 1
            
            if value.islower():
                patterns['lowercase'] += 1
            elif value.isupper():
                patterns['uppercase'] += 1
            elif any(c.islower() for c in value) and any(c.isupper() for c in value):
                patterns['mixed_case'] += 1
        
        # Convert to percentages
        total = len(str_series)
        return {k: (v / total) * 100 for k, v in patterns.items()}
    
    def _analyze_relationships(self, df: pd.DataFrame, threshold: float = 0.7) -> Dict[str, Any]:
        """Analyze relationships between columns"""
        relationships = {
            'correlations': {},
            'dependencies': {},
            'potential_foreign_keys': []
        }
        
        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) >= 0.8 else 'moderate'
                        })
            
            relationships['correlations'] = strong_correlations
        
        # Categorical dependencies
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        dependencies = []
        
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 != col2:
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        if p_value < 0.05:  # Significant dependency
                            dependencies.append({
                                'dependent_column': col1,
                                'independent_column': col2,
                                'chi2_statistic': float(chi2),
                                'p_value': float(p_value),
                                'strength': 'strong' if p_value < 0.001 else 'moderate'
                            })
                    except:
                        continue
        
        relationships['dependencies'] = dependencies
        
        # Potential foreign key detection
        for col in df.columns:
            if ('id' in col.lower() and col.lower() != 'id') or col.lower().endswith('_id'):
                relationships['potential_foreign_keys'].append({
                    'column': col,
                    'unique_values': int(df[col].nunique()),
                    'null_count': int(df[col].isnull().sum()),
                    'sample_values': df[col].dropna().head(5).tolist()
                })
        
        return relationships
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': float((df.duplicated().sum() / len(df)) * 100),
            'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
            'completely_null_columns': df.columns[df.isnull().all()].tolist(),
            'constant_columns': [col for col in df.columns if df[col].nunique() <= 1]
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect high-level patterns in the dataset"""
        patterns = {
            'time_series_candidates': [],
            'hierarchical_relationships': [],
            'enumeration_patterns': []
        }
        
        # Time series detection
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower():
                patterns['time_series_candidates'].append(col)
        
        # Hierarchical relationships (parent-child patterns)
        for col in df.columns:
            if col.lower().endswith('_id') and col.lower() != 'id':
                parent_name = col.lower().replace('_id', '')
                if any(parent_name in other_col.lower() for other_col in df.columns if other_col != col):
                    patterns['hierarchical_relationships'].append({
                        'child_column': col,
                        'potential_parent': parent_name
                    })
        
        # Enumeration patterns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 20:
                patterns['enumeration_patterns'].append({
                    'column': col,
                    'unique_values': df[col].nunique(),
                    'values': df[col].value_counts().to_dict()
                })
        
        return patterns
    
    def _generate_column_hints(self, series: pd.Series, column_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hints for data generation based on column analysis"""
        hints = {}
        inferred_type = column_profile['inferred_type']
        
        if inferred_type in ['int', 'float']:
            hints = {
                'type': 'range',
                'min': column_profile['min'],
                'max': column_profile['max'],
                'distribution': column_profile.get('distribution', {}).get('likely_distribution', 'uniform')
            }
            
            # Add specific distribution parameters if detected
            if column_profile.get('distribution', {}).get('likely_distribution') == 'normal':
                hints.update({
                    'mean': column_profile['mean'],
                    'std': column_profile['std']
                })
        
        elif inferred_type == 'datetime':
            hints = {
                'type': 'date_range',
                'start': column_profile.get('min_date', '2020-01-01'),
                'end': column_profile.get('max_date', '2024-12-31')
            }
        
        elif inferred_type == 'bool':
            true_pct = column_profile.get('true_percentage', 50)
            hints = {
                'type': 'choice',
                'values': [True, False],
                'probabilities': {True: true_pct / 100, False: (100 - true_pct) / 100}
            }
        
        else:  # string/categorical
            if column_profile.get('is_categorical', False):
                value_counts = column_profile['value_distribution']
                total_count = sum(value_counts.values())
                probabilities = {k: v / total_count for k, v in value_counts.items()}
                
                hints = {
                    'type': 'choice',
                    'values': list(value_counts.keys()),
                    'probabilities': probabilities
                }
            else:
                # Check for specific patterns
                patterns = column_profile.get('patterns', {})
                
                if patterns.get('email', 0) > 80:
                    hints = {'type': 'email'}
                elif patterns.get('phone', 0) > 80:
                    hints = {'type': 'phone_number'}
                elif patterns.get('uuid', 0) > 80:
                    hints = {'type': 'uuid'}
                elif patterns.get('url', 0) > 80:
                    hints = {'type': 'url'}
                else:
                    # Default string generation
                    hints = {
                        'type': 'text',
                        'min_length': column_profile.get('min_length', 1),
                        'max_length': column_profile.get('max_length', 50),
                        'avg_length': column_profile.get('avg_length', 25)
                    }
        
        # Add null probability if column has nulls
        null_pct = column_profile.get('null_percentage', 0)
        if null_pct > 0:
            hints['null_probability'] = null_pct / 100
        
        return hints
    
    def generate_test_data_config(self, table_name: str = None) -> Dict[str, Any]:
        """
        Generate a test data configuration file based on profiling results
        
        Args:
            table_name: Specific table to generate config for (None = all tables)
        
        Returns:
            Dictionary containing test data generation configuration
        """
        if not self.profile_results:
            raise ValueError("No profiling results available. Run profile_dataset first.")
        
        tables_to_process = [table_name] if table_name else list(self.profile_results.keys())
        
        config = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'SampleDataProfiler',
                'source': 'profiled_data'
            },
            'tables': []
        }
        
        for table in tables_to_process:
            if table not in self.profile_results:
                self.logger.warning(f"No profile results for table: {table}")
                continue
            
            profile = self.profile_results[table]
            
            table_config = {
                'table_name': table,
                'row_count': profile['row_count'],
                'columns': [],
                'foreign_keys': [],
                'constraints': []
            }
            
            # Generate column configurations
            for col_name, col_profile in profile['columns'].items():
                column_config = {
                    'name': col_name,
                    'type': col_profile['inferred_type'],
                    'nullable': col_profile['null_percentage'] > 0,
                    'constraints': col_profile['constraints'],
                    'rule': col_profile['generation_hints']
                }
                
                table_config['columns'].append(column_config)
            
            # Add potential foreign key relationships
            for fk in profile['relationships']['potential_foreign_keys']:
                table_config['foreign_keys'].append({
                    'child_column': fk['column'],
                    'parent_table': 'unknown',  # This needs to be determined manually
                    'parent_column': 'id'  # Common assumption
                })
            
            config['tables'].append(table_config)
        
        return config
    
    def save_profile_report(self, output_path: str, format: str = 'json') -> None:
        """
        Save profiling report to file
        
        Args:
            output_path: Path to save the report
            format: Output format ('json', 'yaml', 'html')
        """
        if not self.profile_results:
            raise ValueError("No profiling results to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.profile_results, f, indent=2, default=str)
        
        elif format.lower() == 'yaml':
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(self.profile_results, f, default_flow_style=False)
        
        elif format.lower() == 'html':
            self._generate_html_report(output_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Profile report saved to: {output_path}")
    
    def _generate_html_report(self, output_path: Path) -> None:
        """Generate HTML report (basic implementation)"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profiling Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
                .section {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Data Profiling Report</h1>
            <div class="section">
                <h2>Summary</h2>
                <p>Generated at: {datetime.now().isoformat()}</p>
                <p>Tables profiled: {len(self.profile_results)}</p>
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <pre>{json.dumps(self.profile_results, indent=2, default=str)}</pre>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def generate_advanced_profile(self, df: pd.DataFrame, 
                                 output_path: str = None,
                                 use_pandas_profiling: bool = True) -> Optional[str]:
        """
        Generate advanced profiling report using pandas-profiling or sweetviz
        
        Args:
            df: Input DataFrame
            output_path: Path to save the report
            use_pandas_profiling: Whether to use pandas-profiling (vs sweetviz)
        
        Returns:
            Path to generated report (if successful)
        """
        if use_pandas_profiling and PANDAS_PROFILING_AVAILABLE:
            try:
                profile = pandas_profiling.ProfileReport(
                    df,
                    title="Advanced Data Profiling Report",
                    explorative=True
                )
                
                if output_path:
                    profile.to_file(output_path)
                    self.logger.info(f"Advanced profile saved to: {output_path}")
                    return output_path
                else:
                    return profile.to_html()
                    
            except Exception as e:
                self.logger.error(f"Error generating pandas-profiling report: {e}")
        
        elif SWEETVIZ_AVAILABLE:
            try:
                report = sv.analyze(df)
                if output_path:
                    report.show_html(output_path)
                    self.logger.info(f"Sweetviz report saved to: {output_path}")
                    return output_path
                else:
                    # Return HTML content as string
                    temp_path = "temp_sweetviz_report.html"
                    report.show_html(temp_path)
                    with open(temp_path, 'r') as f:
                        html_content = f.read()
                    Path(temp_path).unlink()  # Clean up temp file
                    return html_content
                    
            except Exception as e:
                self.logger.error(f"Error generating sweetviz report: {e}")
        
        else:
            self.logger.warning("Neither pandas-profiling nor sweetviz is available for advanced profiling")
        
        return None
    
    def compare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame,
                        table1_name: str = "dataset1", 
                        table2_name: str = "dataset2") -> Dict[str, Any]:
        """
        Compare two datasets and identify differences
        
        Args:
            df1: First dataset
            df2: Second dataset
            table1_name: Name for first dataset
            table2_name: Name for second dataset
        
        Returns:
            Comparison results
        """
        comparison = {
            'metadata': {
                'comparison_timestamp': datetime.now().isoformat(),
                'dataset1_name': table1_name,
                'dataset2_name': table2_name
            },
            'structure_comparison': {},
            'data_distribution_comparison': {},
            'quality_comparison': {}
        }
        
        # Structure comparison
        comparison['structure_comparison'] = {
            'rows': {'dataset1': len(df1), 'dataset2': len(df2), 'difference': len(df1) - len(df2)},
            'columns': {'dataset1': len(df1.columns), 'dataset2': len(df2.columns), 'difference': len(df1.columns) - len(df2.columns)},
            'common_columns': list(set(df1.columns) & set(df2.columns)),
            'unique_to_dataset1': list(set(df1.columns) - set(df2.columns)),
            'unique_to_dataset2': list(set(df2.columns) - set(df1.columns))
        }
        
        # Compare common columns
        common_cols = comparison['structure_comparison']['common_columns']
        column_comparisons = {}
        
        for col in common_cols:
            col_comparison = {
                'data_type_match': str(df1[col].dtype) == str(df2[col].dtype),
                'null_percentage_diff': abs((df1[col].isnull().sum() / len(df1)) - (df2[col].isnull().sum() / len(df2))) * 100,
                'unique_count_diff': abs(df1[col].nunique() - df2[col].nunique())
            }
            
            # For numeric columns, compare distributions
            if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                col_comparison.update({
                    'mean_diff': abs(df1[col].mean() - df2[col].mean()),
                    'std_diff': abs(df1[col].std() - df2[col].std()),
                    'median_diff': abs(df1[col].median() - df2[col].median())
                })
            
            # For categorical columns, compare value distributions
            elif df1[col].dtype == 'object' and df2[col].dtype == 'object':
                common_values = set(df1[col].unique()) & set(df2[col].unique())
                col_comparison.update({
                    'common_values_count': len(common_values),
                    'unique_values_overlap_percentage': (len(common_values) / max(df1[col].nunique(), df2[col].nunique())) * 100
                })
            
            column_comparisons[col] = col_comparison
        
        comparison['data_distribution_comparison'] = column_comparisons
        
        # Quality comparison
        comparison['quality_comparison'] = {
            'missing_data_percentage': {
                'dataset1': (df1.isnull().sum().sum() / (len(df1) * len(df1.columns))) * 100,
                'dataset2': (df2.isnull().sum().sum() / (len(df2) * len(df2.columns))) * 100
            },
            'duplicate_percentage': {
                'dataset1': (df1.duplicated().sum() / len(df1)) * 100,
                'dataset2': (df2.duplicated().sum() / len(df2)) * 100
            }
        }
        
        return comparison
    
    def suggest_data_improvements(self, table_name: str = None) -> Dict[str, List[str]]:
        """
        Suggest data quality improvements based on profiling results
        
        Args:
            table_name: Specific table to analyze (None = all tables)
        
        Returns:
            Dictionary of improvement suggestions
        """
        if not self.profile_results:
            raise ValueError("No profiling results available. Run profile_dataset first.")
        
        tables_to_analyze = [table_name] if table_name else list(self.profile_results.keys())
        suggestions = {}
        
        for table in tables_to_analyze:
            if table not in self.profile_results:
                continue
                
            profile = self.profile_results[table]
            table_suggestions = []
            
            # Check data quality issues
            quality = profile['data_quality']
            
            if quality['missing_data_percentage'] > 20:
                table_suggestions.append(f"High missing data percentage ({quality['missing_data_percentage']:.1f}%). Consider data cleaning or imputation strategies.")
            
            if quality['duplicate_percentage'] > 5:
                table_suggestions.append(f"Significant duplicate rows ({quality['duplicate_percentage']:.1f}%). Consider deduplication.")
            
            if quality['completely_null_columns']:
                table_suggestions.append(f"Completely null columns found: {', '.join(quality['completely_null_columns'])}. Consider removing these columns.")
            
            if quality['constant_columns']:
                table_suggestions.append(f"Constant columns found: {', '.join(quality['constant_columns'])}. These provide no information variance.")
            
            # Check column-specific issues
            for col_name, col_profile in profile['columns'].items():
                if col_profile['null_percentage'] > 50:
                    table_suggestions.append(f"Column '{col_name}' has high null percentage ({col_profile['null_percentage']:.1f}%).")
                
                if col_profile['inferred_type'] in ['int', 'float']:
                    outliers = col_profile.get('outliers', {})
                    if outliers.get('outlier_percentage', 0) > 10:
                        table_suggestions.append(f"Column '{col_name}' has significant outliers ({outliers['outlier_percentage']:.1f}%).")
                
                if col_profile['uniqueness_percentage'] == 100 and 'unique' not in col_profile['constraints']:
                    table_suggestions.append(f"Column '{col_name}' appears to be unique but not marked as such. Consider adding unique constraint.")
            
            # Check for potential optimization opportunities
            relationships = profile['relationships']
            if not relationships['potential_foreign_keys'] and any('id' in col.lower() for col in profile['columns'].keys()):
                table_suggestions.append("Potential foreign key relationships detected but not defined. Consider establishing proper relationships.")
            
            suggestions[table] = table_suggestions
        
        return suggestions
    
    def export_generation_rules(self, output_path: str, format: str = 'json') -> None:
        """
        Export generation rules that can be used by data generators
        
        Args:
            output_path: Path to save the rules
            format: Output format ('json', 'yaml', 'python')
        """
        if not self.profile_results:
            raise ValueError("No profiling results available. Run profile_dataset first.")
        
        rules = {}
        
        for table_name, profile in self.profile_results.items():
            table_rules = {
                'table_name': table_name,
                'estimated_rows': profile['row_count'],
                'columns': {}
            }
            
            for col_name, col_profile in profile['columns'].items():
                table_rules['columns'][col_name] = {
                    'type': col_profile['inferred_type'],
                    'nullable': col_profile['null_percentage'] > 0,
                    'null_probability': col_profile['null_percentage'] / 100,
                    'generation_rule': col_profile['generation_hints'],
                    'constraints': col_profile['constraints']
                }
            
            rules[table_name] = table_rules
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(rules, f, indent=2, default=str)
        
        elif format.lower() == 'yaml':
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(rules, f, default_flow_style=False)
        
        elif format.lower() == 'python':
            self._export_python_rules(rules, output_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Generation rules exported to: {output_path}")
    
    def _export_python_rules(self, rules: Dict[str, Any], output_path: Path) -> None:
        """Export rules as Python configuration"""
        python_content = f"""# Generated data rules from SampleDataProfiler
# Generated at: {datetime.now().isoformat()}

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Union

# Data generation rules
DATA_GENERATION_RULES = {json.dumps(rules, indent=4, default=str)}

def generate_sample_data(table_name: str, num_rows: int = 100) -> List[Dict[str, Any]]:
    \"\"\"
    Generate sample data based on profiled rules
    
    Args:
        table_name: Name of the table to generate data for
        num_rows: Number of rows to generate
    
    Returns:
        List of dictionaries representing rows
    \"\"\"
    if table_name not in DATA_GENERATION_RULES:
        raise ValueError(f"No rules found for table: {{table_name}}")
    
    rules = DATA_GENERATION_RULES[table_name]
    data = []
    
    for _ in range(num_rows):
        row = {{}}
        
        for col_name, col_rules in rules['columns'].items():
            # Generate value based on rules
            if random.random() < col_rules.get('null_probability', 0):
                row[col_name] = None
            else:
                row[col_name] = generate_column_value(col_rules)
        
        data.append(row)
    
    return data

def generate_column_value(col_rules: Dict[str, Any]) -> Any:
    \"\"\"Generate a single column value based on rules\"\"\"
    generation_rule = col_rules.get('generation_rule', {{}})
    rule_type = generation_rule.get('type', 'text')
    
    if rule_type == 'range':
        return random.uniform(generation_rule['min'], generation_rule['max'])
    elif rule_type == 'choice':
        values = generation_rule['values']
        probabilities = generation_rule.get('probabilities', {{}})
        if probabilities:
            return random.choices(values, weights=[probabilities.get(v, 1) for v in values])[0]
        else:
            return random.choice(values)
    elif rule_type == 'email':
        domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com']
        username = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))
        return f"{{username}}@{{random.choice(domains)}}"
    elif rule_type == 'text':
        length = random.randint(
            generation_rule.get('min_length', 1),
            generation_rule.get('max_length', 50)
        )
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ', k=length))
    elif rule_type == 'date_range':
        start_date = datetime.fromisoformat(generation_rule['start'])
        end_date = datetime.fromisoformat(generation_rule['end'])
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        return (start_date + timedelta(days=random_days)).isoformat()
    else:
        return "generated_value"

# Example usage:
# data = generate_sample_data('your_table_name', 1000)
"""
        
        with open(output_path, 'w') as f:
            f.write(python_content)

# Example usage and main execution
if __name__ == "__main__":
    # Example usage
    profiler = SampleDataProfiler()
    
    # Load and profile sample data
    df = profiler.load_data("sample_data_profiler.csv")
    profile = profiler.profile_dataset(df, "customer")
    # 
    # # Generate configuration for test data
    config = profiler.generate_test_data_config()
    # 
    # # Save reports
    profiler.save_profile_report("profile_report.json")
    profiler.export_generation_rules("generation_rules.json")
    # 
    # # Get improvement suggestions
    suggestions = profiler.suggest_data_improvements()
    print("Data Quality Suggestions:")
    for table, suggestions_list in suggestions.items():
        print(f"\n{table}:")
        for suggestion in suggestions_list:
            print(f"  - {suggestion}")
    
    print("SampleDataProfiler initialized and ready for use!")
    print("Example usage:")
    print("  profiler = SampleDataProfiler()")
    print("  df = profiler.load_data('your_file.csv')")
    print("  profile = profiler.profile_dataset(df, 'table_name')")
    print("  config = profiler.generate_test_data_config()")
    print("  profiler.save_profile_report('report.json')")