import random
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Set, Union, Optional
from collections import defaultdict
import json
import pandas as pd
import re
import string
from pathlib import Path

# Additional libraries for enhanced data generation
try:
    from mimesis import Person, Address, Finance, Datetime, Text, Internet, Code
    from mimesis.locales import Locale
    MIMESIS_AVAILABLE = True
except ImportError:
    MIMESIS_AVAILABLE = False
    print("Warning: mimesis not available. Install with: pip install mimesis")

try:
    from scipy import stats
    import scipy.stats as stats_dist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

from app.core.validator import DataValidator
from app.core.relationship_preserver import RelationshipPreserver
from app.core.rule_engine import RuleEngine


class EnhancedDataGenerator:
    """Enhanced data generator with multiple libraries and advanced profiling integration"""
    
    def __init__(self, config, locale=None, logger=None):
        self.config = config
        self.logger = logger

        if self.logger is None:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
            self.logger = logging.getLogger()

        # Initialize Faker
        self.faker = None
        if locale is not None:
            self.faker = Faker(locale=locale)
        else:
            self.faker = Faker()
        
        # Initialize Mimesis providers if available
        self.mimesis_providers = {}
        if MIMESIS_AVAILABLE:
            mimesis_locale = Locale.EN if locale is None else getattr(Locale, locale.upper(), Locale.EN)
            self.mimesis_providers = {
                'person': Person(mimesis_locale),
                'address': Address(mimesis_locale),
                'finance': Finance(mimesis_locale),
                'datetime': Datetime(mimesis_locale),
                'text': Text(mimesis_locale),
                'internet': Internet(),
                'code': Code()
            }
            self.logger.info("Mimesis providers initialized")
        
        self.validator = DataValidator(logger=self.logger)
        self.relationship_preserver = RelationshipPreserver(logger=self.logger)
        self.rule_engine = RuleEngine(logger=self.logger)

        # Efficient in-memory storage for generated data
        self._generated_data = {}  # table_name -> DataFrame
        
        # Optimized caches for constraint tracking
        self._unique_constraints = defaultdict(set)  # table.column -> set of used values
        
        # Performance optimizations
        self._cache_size_limit = 50000
        self._batch_fk_refresh_threshold = 10000
        
        # Advanced generation patterns
        self._learned_distributions = {}  # Store learned statistical distributions
        self._pattern_cache = {}  # Cache for regex and pattern-based generation
        
        # Build dependency graph if config is available
        if self._is_valid_config():
            self._build_dependency_graph()

    def _is_valid_config(self) -> bool:
        """Check if config is valid and contains tables"""
        return (isinstance(self.config, dict) and 
                'tables' in self.config and 
                isinstance(self.config['tables'], list))

    def _build_dependency_graph(self):
        """Build dependency graph for table generation order"""
        if self._is_valid_config():
            self.relationship_preserver.build_dependency_graph(self.config['tables'])
            self.logger.info("Dependency graph built successfully")

    def integrate_profiling_results(self, profiling_results: Dict):
        """Integrate profiling results from SampleDataProfiler into generation rules"""
        if not profiling_results:
            return
        
        self.logger.info("Integrating profiling results into generation rules")
        
        # Store learned distributions for statistical generation
        for table_name, table_profile in profiling_results.items():
            if isinstance(table_profile, dict) and 'columns' in table_profile:
                self._learned_distributions[table_name] = {}
                
                for col_name, col_profile in table_profile['columns'].items():
                    if isinstance(col_profile, dict):
                        # Store distribution information
                        if 'distribution' in col_profile:
                            self._learned_distributions[table_name][col_name] = col_profile['distribution']
                        
                        # Store pattern information for regex-based generation
                        if 'patterns' in col_profile:
                            pattern_key = f"{table_name}.{col_name}"
                            self._pattern_cache[pattern_key] = col_profile['patterns']

    def generate_value_advanced(self, rule, data_type, column_name=None, table_name=None):
        """Advanced value generation using multiple libraries and learned patterns"""
        
        # First try to use learned distributions if available
        if table_name and column_name and table_name in self._learned_distributions:
            col_dist = self._learned_distributions[table_name].get(column_name)
            if col_dist and self._generate_from_distribution(col_dist, data_type):
                return self._generate_from_distribution(col_dist, data_type)
        
        # Enhanced rule-based generation
        if isinstance(rule, str):
            return self._generate_from_string_rule(rule, data_type)
        elif isinstance(rule, dict):
            return self._generate_from_dict_rule(rule, data_type, column_name, table_name)
        
        # Fallback to basic generation
        return self._generate_default_value(data_type)

    def _generate_from_distribution(self, distribution_info: Dict, data_type: str):
        """Generate values based on learned statistical distributions"""
        if not SCIPY_AVAILABLE or not distribution_info:
            return None
        
        dist_name = distribution_info.get('best_fit')
        dist_params = distribution_info.get('parameters', {})
        
        try:
            if dist_name == 'normal' and 'mean' in dist_params and 'std' in dist_params:
                value = np.random.normal(dist_params['mean'], dist_params['std'])
                if data_type == 'int':
                    return int(round(value))
                return round(value, 2)
            
            elif dist_name == 'uniform' and 'min' in dist_params and 'max' in dist_params:
                value = np.random.uniform(dist_params['min'], dist_params['max'])
                if data_type == 'int':
                    return int(round(value))
                return round(value, 2)
            
            elif dist_name == 'exponential' and 'scale' in dist_params:
                value = np.random.exponential(dist_params['scale'])
                if data_type == 'int':
                    return int(round(value))
                return round(value, 2)
                
        except Exception as e:
            self.logger.warning(f"Failed to generate from distribution {dist_name}: {e}")
        
        return None

    def _generate_from_string_rule(self, rule: str, data_type: str):
        """Enhanced string rule generation with multiple libraries"""
        cleaned_rule = rule.replace(" ", "").replace("_", "").lower()
        
        # Try Mimesis first if available
        if MIMESIS_AVAILABLE:
            mimesis_value = self._generate_with_mimesis(cleaned_rule)
            if mimesis_value is not None:
                return mimesis_value
        
        # Enhanced Faker mapping
        faker_mapping = {
            "bool": lambda: random.choice([True, False]),
            "uuid": lambda: self.faker.uuid4(),
            "cc": lambda: self.faker.credit_card_number(),
            "ccnumber": lambda: self.faker.credit_card_number(),
            "creditcard": lambda: self.faker.credit_card_number(),
            "cvv": lambda: self.faker.credit_card_security_code(),
            "ccexpiry": lambda: self.faker.credit_card_expire(),
            "phone": lambda: self.faker.phone_number(),
            "phonenumber": lambda: self.faker.phone_number(),
            "mobile": lambda: self.faker.phone_number(),
            "firstname": lambda: self.faker.first_name(),
            "lastname": lambda: self.faker.last_name(),
            "fullname": lambda: self.faker.name(),
            "username": lambda: self.faker.user_name(),
            "password": lambda: self.faker.password(),
            "timestamp": lambda: self.faker.date_time().strftime("%Y-%m-%d %H:%M:%S"),
            "ipaddress": lambda: self.faker.ipv4(),
            "ipv4": lambda: self.faker.ipv4(),
            "ipv6": lambda: self.faker.ipv6(),
            "mac": lambda: self.faker.mac_address(),
            "macaddress": lambda: self.faker.mac_address(),
            "url": lambda: self.faker.url(),
            "domain": lambda: self.faker.domain_name(),
            "company": lambda: self.faker.company(),
            "jobtitle": lambda: self.faker.job(),
            "ssn": lambda: self.faker.ssn(),
            "iban": lambda: self.faker.iban(),
            "currency": lambda: self.faker.currency_code(),
            "license": lambda: self.faker.license_plate(),
            "vin": lambda: self._generate_vin(),
        }
        
        # Try direct mapping
        if cleaned_rule in faker_mapping:
            return faker_mapping[cleaned_rule]()
        
        # Try Faker attribute lookup
        try:
            faker_dict = {x.replace("_", ""): x for x in dir(self.faker) if not x.startswith("_")}
            faker_dict.update({''.join([y[0] for y in x.split('_')]): x for x in dir(self.faker) if
                               '_' in x and not x.startswith('_')})
            
            if cleaned_rule in faker_dict:
                return getattr(self.faker, faker_dict[cleaned_rule])()
        except AttributeError:
            self.logger.warning(f"Faker does not support {rule}")
        
        return None

    def _generate_with_mimesis(self, rule: str):
        """Generate values using Mimesis library"""
        if not MIMESIS_AVAILABLE:
            return None
        
        mimesis_mapping = {
            # Person data
            "firstname": lambda: self.mimesis_providers['person'].first_name(),
            "lastname": lambda: self.mimesis_providers['person'].last_name(),
            "fullname": lambda: self.mimesis_providers['person'].full_name(),
            "username": lambda: self.mimesis_providers['person'].username(),
            "password": lambda: self.mimesis_providers['person'].password(),
            "gender": lambda: self.mimesis_providers['person'].gender(),
            "age": lambda: self.mimesis_providers['person'].age(),
            "occupation": lambda: self.mimesis_providers['person'].occupation(),
            
            # Address data
            "address": lambda: self.mimesis_providers['address'].address(),
            "street": lambda: self.mimesis_providers['address'].street_name(),
            "city": lambda: self.mimesis_providers['address'].city(),
            "state": lambda: self.mimesis_providers['address'].state(),
            "country": lambda: self.mimesis_providers['address'].country(),
            "zipcode": lambda: self.mimesis_providers['address'].postal_code(),
            "latitude": lambda: self.mimesis_providers['address'].latitude(),
            "longitude": lambda: self.mimesis_providers['address'].longitude(),
            
            # Finance data
            "bankaccount": lambda: self.mimesis_providers['finance'].bank_account(),
            "creditcard": lambda: self.mimesis_providers['finance'].credit_card_number(),
            "currency": lambda: self.mimesis_providers['finance'].currency_iso_code(),
            "price": lambda: self.mimesis_providers['finance'].price(),
            
            # Internet data
            "email": lambda: self.mimesis_providers['internet'].email(),
            "domain": lambda: self.mimesis_providers['internet'].domain_name(),
            "url": lambda: self.mimesis_providers['internet'].url(),
            "ipv4": lambda: self.mimesis_providers['internet'].ip_v4(),
            "ipv6": lambda: self.mimesis_providers['internet'].ip_v6(),
            "mac": lambda: self.mimesis_providers['internet'].mac_address(),
            
            # Text data
            "text": lambda: self.mimesis_providers['text'].text(),
            "sentence": lambda: self.mimesis_providers['text'].sentence(),
            "word": lambda: self.mimesis_providers['text'].word(),
            
            # Code data
            "isbn": lambda: self.mimesis_providers['code'].isbn(),
            "imei": lambda: self.mimesis_providers['code'].imei(),
        }
        
        if rule in mimesis_mapping:
            try:
                return mimesis_mapping[rule]()
            except Exception as e:
                self.logger.warning(f"Mimesis generation failed for {rule}: {e}")
        
        return None

    def _generate_from_dict_rule(self, rule: Dict, data_type: str, column_name: str = None, table_name: str = None):
        """Enhanced dictionary rule generation"""
        rule_type = rule.get("type")
        
        if rule_type == "pattern" and "regex" in rule:
            return self._generate_from_regex(rule["regex"])
        
        elif rule_type == "ml_distribution" and SKLEARN_AVAILABLE:
            return self._generate_from_ml_distribution(rule, data_type)
        
        elif rule_type == "correlation" and table_name and column_name:
            return self._generate_correlated_value(rule, table_name, column_name)
        
        elif rule_type == "sequence":
            return self._generate_sequence_value(rule, data_type)
        
        # Enhanced existing rule types
        elif rule_type == "choice":
            return self.generate_value_with_distribution(rule, data_type)
        
        elif rule_type in ["date", "date_range"]:
            start_date = rule.get("start", "1950-01-01")
            end_date = rule.get("end", datetime.now().strftime("%Y-%m-%d"))
            return self.random_date(start_date, end_date)
        
        elif rule_type == "range":
            min_val = rule.get("min", 0)
            max_val = rule.get("max", 10000)
            if data_type == "int":
                return random.randint(int(min_val), int(max_val))
            elif data_type == "float":
                return round(random.uniform(min_val, max_val), 2)
        
        elif rule_type in ["fixed", "default"]:
            return rule.get("value")
        
        return None

    def _generate_from_regex(self, pattern: str):
        """Generate string matching regex pattern"""
        try:
            # Simple regex pattern generation (basic implementation)
            # For more complex patterns, consider using rstr library
            if pattern == r'\d{10}':  # Phone number pattern
                return ''.join([str(random.randint(0, 9)) for _ in range(10)])
            elif pattern == r'[A-Z]{2}\d{4}':  # License plate pattern
                return ''.join([random.choice(string.ascii_uppercase) for _ in range(2)]) + \
                       ''.join([str(random.randint(0, 9)) for _ in range(4)])
            elif r'\d{' in pattern:  # Generic digit pattern
                match = re.search(r'\\d\{(\d+)\}', pattern)
                if match:
                    length = int(match.group(1))
                    return ''.join([str(random.randint(0, 9)) for _ in range(length)])
            
            # Fallback: generate based on pattern structure
            return self._basic_regex_generation(pattern)
            
        except Exception as e:
            self.logger.warning(f"Regex generation failed for pattern {pattern}: {e}")
            return "pattern_match_failed"

    def _basic_regex_generation(self, pattern: str) -> str:
        """Basic regex pattern generation"""
        result = ""
        i = 0
        while i < len(pattern):
            if pattern[i] == '\\' and i + 1 < len(pattern):
                if pattern[i + 1] == 'd':
                    result += str(random.randint(0, 9))
                    i += 2
                elif pattern[i + 1] == 'w':
                    result += random.choice(string.ascii_letters + string.digits)
                    i += 2
                else:
                    result += pattern[i + 1]
                    i += 2
            elif pattern[i] == '[' and ']' in pattern[i:]:
                close_idx = pattern.index(']', i)
                char_class = pattern[i + 1:close_idx]
                if char_class == 'A-Z':
                    result += random.choice(string.ascii_uppercase)
                elif char_class == 'a-z':
                    result += random.choice(string.ascii_lowercase)
                elif char_class == '0-9':
                    result += str(random.randint(0, 9))
                else:
                    result += random.choice(char_class.replace('-', ''))
                i = close_idx + 1
            else:
                if pattern[i] not in '{+*?}':
                    result += pattern[i]
                i += 1
        
        return result

    def _generate_from_ml_distribution(self, rule: Dict, data_type: str):
        """Generate values using ML-learned distributions"""
        if not SKLEARN_AVAILABLE:
            return None
        
        try:
            distribution_data = rule.get("learned_data", [])
            if not distribution_data:
                return None
            
            # Use Gaussian Mixture Model for complex distributions
            data = np.array(distribution_data).reshape(-1, 1)
            gmm = GaussianMixture(n_components=min(3, len(set(distribution_data))))
            gmm.fit(data)
            
            # Generate new sample
            sample = gmm.sample(1)[0][0][0]
            
            if data_type == 'int':
                return int(round(sample))
            return round(sample, 2)
            
        except Exception as e:
            self.logger.warning(f"ML distribution generation failed: {e}")
            return None

    def _generate_sequence_value(self, rule: Dict, data_type: str):
        """Generate sequential values (useful for IDs, timestamps, etc.)"""
        sequence_type = rule.get("sequence_type", "increment")
        start_value = rule.get("start", 1)
        step = rule.get("step", 1)
        
        # Simple implementation - in practice, you'd want to track sequence state
        if sequence_type == "increment":
            return start_value + (random.randint(0, 1000) * step)
        elif sequence_type == "timestamp":
            base_time = datetime.now()
            return base_time + timedelta(seconds=random.randint(0, 86400))
        
        return start_value

    def _generate_vin(self) -> str:
        """Generate a realistic VIN number"""
        # Simplified VIN generation
        chars = string.ascii_uppercase + string.digits
        chars = chars.replace('I', '').replace('O', '').replace('Q', '')  # VIN excludes I, O, Q
        return ''.join(random.choices(chars, k=17))

    def generate_batch_enhanced(self, table_metadata, batch_size, foreign_key_data=None, 
                              profiling_results=None):
        """Enhanced batch generation with profiling integration"""
        
        # Integrate profiling results if provided
        if profiling_results:
            self.integrate_profiling_results(profiling_results)
        
        table_name = table_metadata["table_name"]
        columns = table_metadata["columns"]
        
        batch_data = []
        
        for record_idx in range(batch_size):
            row = {}
            
            # Generate values for each column
            for column in columns:
                column_name = column["name"]
                data_type = column["type"]
                rule = column.get("rule", {})
                constraints = column.get("constraints", []) + column.get("constraint", [])
                
                # Use enhanced generation
                if "unique" in constraints:
                    row[column_name] = self.generate_unique_value_enhanced(
                        column, table_name, profiling_results
                    )
                else:
                    # Apply conditional rules first
                    conditional_value = self.apply_conditional_rules(row, column)
                    if conditional_value is not None:
                        row[column_name] = conditional_value
                    else:
                        # Use advanced generation
                        row[column_name] = self.generate_value_advanced(
                            rule, data_type, column_name, table_name
                        )
                
                # Handle nullable columns
                nullable = column.get("nullable", True)
                if nullable and random.random() < 0.1:  # 10% chance of null
                    row[column_name] = None
            
            batch_data.append(row)
        
        return batch_data

    def generate_unique_value_enhanced(self, column_def, table_name, profiling_results=None):
        """Enhanced unique value generation using profiling insights"""
        column_name = column_def["name"]
        data_type = column_def["type"]
        rule = column_def.get("rule", {})
        
        # Check if we have profiling data for this column
        if (profiling_results and table_name in profiling_results and 
            'columns' in profiling_results[table_name] and 
            column_name in profiling_results[table_name]['columns']):
            
            col_profile = profiling_results[table_name]['columns'][column_name]
            
            # Use value distribution from profiling
            if 'value_counts' in col_profile:
                # Generate values following the observed distribution
                values = list(col_profile['value_counts'].keys())
                weights = list(col_profile['value_counts'].values())
                base_value = random.choices(values, weights=weights, k=1)[0]
            else:
                base_value = self.generate_value_advanced(rule, data_type, column_name, table_name)
        else:
            base_value = self.generate_value_advanced(rule, data_type, column_name, table_name)
        
        # Ensure uniqueness
        max_attempts = 100
        for attempt in range(max_attempts):
            if not self._is_unique_value_used(table_name, column_name, base_value):
                self._add_unique_value(table_name, column_name, base_value)
                return base_value
            
            # Modify the value to make it unique
            if isinstance(base_value, str):
                base_value = f"{base_value}_{attempt}"
            elif isinstance(base_value, (int, float)):
                base_value = base_value + attempt + 1
            else:
                base_value = f"unique_{attempt}_{random.randint(1000, 9999)}"
        
        # Final fallback
        return f"unique_{column_name}_{random.randint(10000, 99999)}"

    # Include all the existing methods from OptimizedDataGenerator
    def _convert_value_to_type(self, value: str, data_type: str) -> Any:
        """Convert string value back to original data type"""
        if value is None or value == 'None':
            return None

        try:
            if data_type.lower() in ['int', 'integer']:
                return int(value)
            elif data_type.lower() in ['float', 'double', 'decimal']:
                return float(value)
            elif data_type.lower() in ['bool', 'boolean']:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif data_type.lower() in ['date']:
                if isinstance(value, str):
                    return datetime.strptime(value, '%Y-%m-%d').date()
                return value
            elif data_type.lower() in ['datetime', 'timestamp']:
                if isinstance(value, str):
                    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                return value
            else:  # string types
                return str(value)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not convert value '{value}' to type '{data_type}': {e}")
            return value

    def reset_constraint_tracking(self):
        """Reset all constraint tracking for fresh generation"""
        self._unique_constraints.clear()
        self._generated_data.clear()
        self.relationship_preserver.clear_fk_pools()
        self._learned_distributions.clear()
        self._pattern_cache.clear()
        self.logger.info("Enhanced constraint tracking cache reset")

    # [Include all other existing methods from OptimizedDataGenerator...]
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics about the generation process"""
        stats = {
            'libraries_available': {
                'faker': True,
                'mimesis': MIMESIS_AVAILABLE,
                'scipy': SCIPY_AVAILABLE,
                'sklearn': SKLEARN_AVAILABLE
            },
            'learned_distributions': len(self._learned_distributions),
            'pattern_cache_size': len(self._pattern_cache),
            'unique_constraints_tracked': len(self._unique_constraints),
            'generated_tables': list(self._generated_data.keys()),
            'total_records': sum(len(df) for df in self._generated_data.values())
        }
        return stats

    # Add this method to the EnhancedDataGenerator class

    def apply_conditional_rules(self, row_data: Dict[str, Any], column_def: Dict[str, Any]) -> Any:
        """Apply conditional rules based on current row data"""

        # Get conditional rules from column definition
        conditional_rules = column_def.get("conditional_rules", [])

        if not conditional_rules:
            return None

        # Check each conditional rule
        for condition_rule in conditional_rules:
            condition = condition_rule.get("condition", {})

            # Evaluate the condition
            if self._evaluate_condition(condition, row_data):
                # Apply the conditional rule
                rule = condition_rule.get("rule", {})
                data_type = condition_rule.get("type", column_def.get("type"))
                column_name = column_def.get("name")
                table_name = getattr(self, '_current_table_name', None)

                return self.generate_value_advanced(rule, data_type, column_name, table_name)

        return None

    def _evaluate_condition(self, condition: Dict[str, Any], row_data: Dict[str, Any]) -> bool:
        """Evaluate a conditional rule condition"""
        if not condition:
            return True

        column = condition.get('column')
        operator = condition.get('operator', '==')
        value = condition.get('value')

        if column not in row_data:
            return False

        row_value = row_data[column]

        try:
            if operator == '==':
                return row_value == value
            elif operator == '!=':
                return row_value != value
            elif operator == '>':
                return row_value > value
            elif operator == '<':
                return row_value < value
            elif operator == '>=':
                return row_value >= value
            elif operator == '<=':
                return row_value <= value
            elif operator == 'in':
                return row_value in value if isinstance(value, (list, tuple, set)) else False
            elif operator == 'not_in':
                return row_value not in value if isinstance(value, (list, tuple, set)) else True
            elif operator == 'contains':
                return value in str(row_value)
            elif operator == 'not_contains':
                return value not in str(row_value)
            elif operator == 'regex':
                return bool(re.match(str(value), str(row_value)))
            elif operator == 'is_null':
                return row_value is None
            elif operator == 'is_not_null':
                return row_value is not None
        except Exception as e:
            self.logger.warning(f"Error evaluating condition {condition}: {e}")
            return False

        return False

    # Also update the generate_batch_enhanced method to set current table name
    def generate_batch_enhanced(self, table_metadata, batch_size, foreign_key_data=None,
                                profiling_results=None):
        """Enhanced batch generation with profiling integration"""

        # Integrate profiling results if provided
        if profiling_results:
            self.integrate_profiling_results(profiling_results)

        table_name = table_metadata["table_name"]
        columns = table_metadata["columns"]

        # Set current table name for conditional rules
        self._current_table_name = table_name

        batch_data = []

        for record_idx in range(batch_size):
            row = {}

            # Generate values for each column
            for column in columns:
                column_name = column["name"]
                data_type = column["type"]
                rule = column.get("rule", {})
                constraints = column.get("constraints", []) + column.get("constraint", [])

                # Use enhanced generation
                if "unique" in constraints:
                    row[column_name] = self.generate_unique_value_enhanced(
                        column, table_name, profiling_results
                    )
                else:
                    # Apply conditional rules first
                    conditional_value = self.apply_conditional_rules(row, column)
                    if conditional_value is not None:
                        row[column_name] = conditional_value
                    else:
                        # Use advanced generation
                        row[column_name] = self.generate_value_advanced(
                            rule, data_type, column_name, table_name
                        )

                # Handle nullable columns
                nullable = column.get("nullable", True)
                if nullable and random.random() < 0.1:  # 10% chance of null
                    row[column_name] = None

            batch_data.append(row)

        # Clear current table name
        self._current_table_name = None

        return batch_data

    # Add these missing methods to the EnhancedDataGenerator class

    def store_generated_batch(self, table_name: str, batch_data: List[Dict[str, Any]]):
        """Store generated batch data in memory"""
        if not batch_data:
            return

        # Convert batch data to DataFrame
        df = pd.DataFrame(batch_data)

        # Store or append to existing data
        if table_name in self._generated_data:
            self._generated_data[table_name] = pd.concat([self._generated_data[table_name], df], ignore_index=True)
        else:
            self._generated_data[table_name] = df

        self.logger.info(f"Stored {len(batch_data)} records for table '{table_name}'")

    def get_generated_data_df(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get generated data as DataFrame for a specific table"""
        return self._generated_data.get(table_name)

    def get_all_generated_data(self) -> Dict[str, pd.DataFrame]:
        """Get all generated data as dictionary of DataFrames"""
        return self._generated_data.copy()

    def clear_generated_data(self, table_name: str = None):
        """Clear generated data for a specific table or all tables"""
        if table_name:
            if table_name in self._generated_data:
                del self._generated_data[table_name]
                self.logger.info(f"Cleared data for table '{table_name}'")
        else:
            self._generated_data.clear()
            self.logger.info("Cleared all generated data")

    def export_generated_data(self, table_name: str, file_path: str, format: str = 'csv'):
        """Export generated data to file"""
        if table_name not in self._generated_data:
            raise ValueError(f"No data found for table '{table_name}'")

        df = self._generated_data[table_name]
        file_path = Path(file_path)

        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Exported {len(df)} records from '{table_name}' to {file_path}")

    def get_data_summary(self, table_name: str = None) -> Dict[str, Any]:
        """Get summary statistics of generated data"""
        if table_name:
            if table_name not in self._generated_data:
                return {}

            df = self._generated_data[table_name]
            return {
                'table_name': table_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'null_counts': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        else:
            # Summary for all tables
            summary = {}
            total_rows = 0
            total_memory = 0

            for tbl_name, df in self._generated_data.items():
                summary[tbl_name] = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
                total_rows += len(df)
                total_memory += df.memory_usage(deep=True).sum()

            summary['_total'] = {
                'table_count': len(self._generated_data),
                'total_rows': total_rows,
                'total_memory_bytes': total_memory
            }

            return summary

    def _is_unique_value_used(self, table_name: str, column_name: str, value: Any) -> bool:
        """Check if a unique value has already been used"""
        key = f"{table_name}.{column_name}"
        return value in self._unique_constraints[key]

    def _add_unique_value(self, table_name: str, column_name: str, value: Any):
        """Add a value to the unique constraints tracking"""
        key = f"{table_name}.{column_name}"
        self._unique_constraints[key].add(value)

        # Prevent memory issues by limiting cache size
        if len(self._unique_constraints[key]) > self._cache_size_limit:
            # Remove oldest 10% of values (simplified approach)
            values_list = list(self._unique_constraints[key])
            keep_count = int(len(values_list) * 0.9)
            self._unique_constraints[key] = set(values_list[-keep_count:])

    def generate_value_with_distribution(self, rule: Dict[str, Any], data_type: str) -> Any:
        """Generate value with probability distribution for choice rules"""
        if rule.get("type") != "choice":
            return None

        values = rule.get("value", [])
        if not values:
            return None

        probabilities = rule.get("probabilities")

        if probabilities:
            # Use weighted random choice
            weights = [probabilities.get(str(v), 1.0) for v in values]
            return random.choices(values, weights=weights, k=1)[0]
        else:
            # Equal probability
            return random.choice(values)

    def random_date(self, start_date: str, end_date: str) -> str:
        """Generate random date between start and end dates"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            # Calculate random date
            time_between = end - start
            days_between = time_between.days
            random_days = random.randrange(days_between)
            random_date = start + timedelta(days=random_days)

            return random_date.strftime("%Y-%m-%d")
        except ValueError as e:
            self.logger.warning(f"Error generating random date: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    def _generate_default_value(self, data_type: str) -> Any:
        """Generate default value based on data type"""
        data_type = data_type.lower()

        if data_type in ['int', 'integer', 'bigint']:
            return random.randint(1, 1000)
        elif data_type in ['float', 'double', 'decimal', 'numeric']:
            return round(random.uniform(1.0, 1000.0), 2)
        elif data_type in ['varchar', 'text', 'string', 'char']:
            return self.faker.text(max_nb_chars=50)
        elif data_type in ['date']:
            return self.random_date("2020-01-01", "2024-12-31")
        elif data_type in ['datetime', 'timestamp']:
            return self.faker.date_time().strftime("%Y-%m-%d %H:%M:%S")
        elif data_type in ['boolean', 'bool']:
            return random.choice([True, False])
        elif data_type == 'uuid':
            return self.faker.uuid4()
        else:
            return self.faker.text(max_nb_chars=20)

    # Additional helper methods that might be needed

    def _generate_correlated_value(self, rule: Dict[str, Any], table_name: str, column_name: str) -> Any:
        """Generate value correlated with other columns (placeholder implementation)"""
        # This is a simplified implementation - in practice, you'd implement
        # more sophisticated correlation logic
        correlation_column = rule.get("correlation_column")
        correlation_type = rule.get("correlation_type", "positive")

        # For now, return a random value
        # In a full implementation, you'd look up the correlated column value
        # and generate a value based on the correlation type
        return self._generate_default_value(rule.get("data_type", "string"))

    def validate_generated_data(self, table_name: str) -> Dict[str, Any]:
        """Validate generated data against constraints"""
        if table_name not in self._generated_data:
            return {"valid": False, "error": f"Table '{table_name}' not found"}

        df = self._generated_data[table_name]
        validation_results = {
            "valid": True,
            "issues": [],
            "statistics": {
                "total_rows": len(df),
                "null_counts": df.isnull().sum().to_dict(),
                "duplicate_counts": {}
            }
        }

        # Check for duplicates in columns that should be unique
        for column in df.columns:
            duplicates = df[column].duplicated().sum()
            if duplicates > 0:
                validation_results["statistics"]["duplicate_counts"][column] = duplicates

        return validation_results


# Integration with SampleDataProfiler
# Corrected ProfilerIntegratedGenerator class

class ProfilerIntegratedGenerator(EnhancedDataGenerator):
    """Data generator with built-in profiling integration"""

    def __init__(self, config, locale=None, logger=None):
        super().__init__(config, locale, logger)
        self.profiler = None

    def set_profiler(self, profiler):
        """Set the profiler instance for integration"""
        self.profiler = profiler
        self.logger.info("Profiler instance set successfully")

    def generate_from_sample_data(self, sample_data_source, table_name: str,
                                  batch_size: int = 1000) -> pd.DataFrame:
        """Generate synthetic data based on sample data analysis"""
        if self.profiler is None:
            raise ValueError("Profiler not set. Use set_profiler() first.")

        try:
            # Load and profile the sample data
            self.logger.info(f"Loading sample data from: {sample_data_source}")
            df_sample = self.profiler.load_data(sample_data_source)

            if df_sample is None or df_sample.empty:
                raise ValueError("Failed to load sample data or data is empty")

            self.logger.info(f"Profiling dataset for table: {table_name}")
            profiling_results = self.profiler.profile_dataset(df_sample, table_name)

            # Generate test data config from profiling
            self.logger.info("Generating test data configuration from profiling results")
            config = self.profiler.generate_test_data_config(table_name)

            # Validate the generated config
            if not config or 'tables' not in config or not config['tables']:
                raise ValueError("Failed to generate valid config from profiling results")

            # Update our config with the generated config
            table_metadata = config['tables'][0]  # Assuming single table

            # Validate table metadata
            if 'table_name' not in table_metadata or 'columns' not in table_metadata:
                raise ValueError("Invalid table metadata generated from profiling")

            self.logger.info(f"Generating {batch_size} synthetic records")

            # Generate synthetic data using profiling insights
            batch_data = self.generate_batch_enhanced(
                table_metadata, batch_size, profiling_results=profiling_results
            )

            if not batch_data:
                raise ValueError("Failed to generate batch data")

            # Store and return as DataFrame
            self.store_generated_batch(table_name, batch_data)
            result_df = self.get_generated_data_df(table_name)

            if result_df is None or result_df.empty:
                raise ValueError("Failed to retrieve generated data")

            self.logger.info(f"Successfully generated {len(result_df)} records for table '{table_name}'")
            return result_df

        except Exception as e:
            self.logger.error(f"Error in generate_from_sample_data: {e}")
            raise

    def generate_from_multiple_samples(self, sample_sources: Dict[str, str],
                                       batch_sizes: Dict[str, int] = None) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data from multiple sample data sources"""
        if self.profiler is None:
            raise ValueError("Profiler not set. Use set_profiler() first.")

        results = {}
        batch_sizes = batch_sizes or {}

        for table_name, source in sample_sources.items():
            try:
                batch_size = batch_sizes.get(table_name, 1000)
                self.logger.info(f"Processing table '{table_name}' from source: {source}")

                df = self.generate_from_sample_data(source, table_name, batch_size)
                results[table_name] = df

            except Exception as e:
                self.logger.error(f"Failed to generate data for table '{table_name}': {e}")
                # Continue with other tables instead of failing completely
                continue

        return results

    def update_generation_rules_from_profiling(self, table_name: str,
                                               profiling_results: Dict[str, Any]):
        """Update generation rules based on new profiling results"""
        if not profiling_results:
            self.logger.warning("No profiling results provided for rule updates")
            return

        try:
            # Integrate the new profiling results
            self.integrate_profiling_results({table_name: profiling_results})

            # Update rule engine if available
            if hasattr(self, 'rule_engine') and self.rule_engine:
                # Create updated template from profiling results
                schema = self._convert_profiling_to_schema(profiling_results)
                if schema:
                    template = self.rule_engine.create_template_from_schema(schema)
                    self.rule_engine.add_template(template)
                    self.logger.info(f"Updated rules for table '{table_name}' from profiling")

        except Exception as e:
            self.logger.error(f"Error updating generation rules: {e}")

    def _convert_profiling_to_schema(self, profiling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert profiling results to schema format for rule engine"""
        try:
            schema = {
                'table_name': 'profiled_table',
                'columns': {}
            }

            if 'columns' in profiling_results:
                for col_name, col_profile in profiling_results['columns'].items():
                    if isinstance(col_profile, dict):
                        schema['columns'][col_name] = {
                            'type': col_profile.get('data_type', 'string'),
                            'nullable': col_profile.get('null_percentage', 0) > 0,
                            'constraints': []
                        }

                        # Add unique constraint if appropriate
                        if col_profile.get('unique_percentage', 0) > 0.95:
                            schema['columns'][col_name]['constraints'].append('unique')

            return schema

        except Exception as e:
            self.logger.error(f"Error converting profiling to schema: {e}")
            return {}

    def get_profiling_statistics(self) -> Dict[str, Any]:
        """Get statistics about profiling integration"""
        stats = {
            'profiler_set': self.profiler is not None,
            'learned_distributions_count': len(self._learned_distributions),
            'pattern_cache_size': len(self._pattern_cache),
            'profiled_tables': list(self._learned_distributions.keys()) if hasattr(self,
                                                                                   '_learned_distributions') else []
        }

        if hasattr(self, '_generated_data'):
            stats['generated_tables'] = list(self._generated_data.keys())
            stats['total_generated_records'] = sum(len(df) for df in self._generated_data.values())

        return stats

    def export_profiling_integrated_data(self, output_directory: str,
                                         include_profiling_report: bool = True):
        """Export all generated data with optional profiling reports"""
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = []

        # Export generated data
        for table_name, df in self._generated_data.items():
            file_path = output_path / f"{table_name}_synthetic.csv"
            df.to_csv(file_path, index=False)
            exported_files.append(str(file_path))

            # Export profiling report if requested
            if include_profiling_report and self.profiler:
                try:
                    # Generate a summary report
                    summary = self.get_data_summary(table_name)
                    if summary:
                        report_path = output_path / f"{table_name}_generation_report.json"
                        with open(report_path, 'w') as f:
                            json.dump(summary, f, indent=2, default=str)
                        exported_files.append(str(report_path))
                except Exception as e:
                    self.logger.warning(f"Could not generate report for {table_name}: {e}")

        # Export overall statistics
        try:
            stats_path = output_path / "generation_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(self.get_profiling_statistics(), f, indent=2, default=str)
            exported_files.append(str(stats_path))
        except Exception as e:
            self.logger.warning(f"Could not export statistics: {e}")

        self.logger.info(f"Exported {len(exported_files)} files to {output_directory}")
        return exported_files