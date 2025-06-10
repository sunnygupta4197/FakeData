import re
import logging
from typing import Any, List, Dict, Union, Optional, Tuple
from datetime import datetime, date
from email.utils import parseaddr


class DataValidator:
    """Enhanced data validator with comprehensive validation rules"""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Common regex patterns
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?[0-9]{10,15}$',
            'phone_us': r'^\+?1?[2-9]\d{2}[2-9]\d{2}\d{4}$',
            'phone_international': r'^\+[1-9]\d{1,14}$',
            'zipcode': r'^\d{5}(-\d{4})?$',
            'zipcode_ca': r'^[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d$',
            'credit_card': r'^\d{13,19}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'ip_address': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            'url': r'^https?:\/\/(?:[-\w.])+(?:\:[0-9]+)?(?:\/(?:[\w\/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            'alphanumeric': r'^[a-zA-Z0-9]+$',
            'alpha': r'^[a-zA-Z]+$',
            'numeric': r'^\d+$',
            'decimal': r'^\d+(\.\d+)?$'
        }

    def regex_validator(self, pattern: str, value: str) -> bool:
        """Validate value against regex pattern"""
        try:
            if not isinstance(value, str):
                value = str(value)
            return bool(re.match(pattern, value))
        except Exception as e:
            self.logger.warning(f"Regex validation failed for pattern '{pattern}' and value '{value}': {e}")
            return False

    def email_validator(self, email: str) -> bool:
        """Validate email address"""
        if not isinstance(email, str):
            return False

        # Basic regex check
        if not self.regex_validator(self.patterns['email'], email):
            return False

        # Additional validation using email.utils
        try:
            parsed = parseaddr(email)
            return '@' in parsed[1] and '.' in parsed[1].split('@')[1]
        except Exception:
            return False

    def phone_validator(self, phone: str, country_code: str = None) -> bool:
        """Validate phone number with optional country-specific validation"""
        if not isinstance(phone, str):
            return False

        # Remove common formatting characters
        cleaned_phone = re.sub(r'[^\d+]', '', phone)

        if country_code == 'US':
            return self.regex_validator(self.patterns['phone_us'], cleaned_phone)
        elif country_code == 'INTL':
            return self.regex_validator(self.patterns['phone_international'], cleaned_phone)
        else:
            return self.regex_validator(self.patterns['phone'], cleaned_phone)

    def date_validator(self, value: Any, date_format: str = None) -> bool:
        """Validate date value"""
        if isinstance(value, (date, datetime)):
            return True

        if isinstance(value, str):
            formats_to_try = []
            if date_format:
                formats_to_try.append(date_format)

            # Common date formats
            formats_to_try.extend([
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ])

            for fmt in formats_to_try:
                try:
                    datetime.strptime(value, fmt)
                    return True
                except ValueError:
                    continue

        return False

    def range_validator(self, value: Union[int, float], min_val: Optional[Union[int, float]] = None,
                        max_val: Optional[Union[int, float]] = None) -> bool:
        """Validate numeric value within range"""
        try:
            if not isinstance(value, (int, float)):
                value = float(value)

            if min_val is not None and value < min_val:
                return False

            if max_val is not None and value > max_val:
                return False

            return True
        except (ValueError, TypeError):
            return False

    def length_validator(self, value: str, min_length: Optional[int] = None,
                         max_length: Optional[int] = None) -> bool:
        """Validate string length"""
        if not isinstance(value, str):
            return False

        length = len(value)

        if min_length is not None and length < min_length:
            return False

        if max_length is not None and length > max_length:
            return False

        return True

    def choice_validator(self, value: Any, choices: List[Any]) -> bool:
        """Validate value is in allowed choices"""
        return value in choices

    def credit_card_validator(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        if not isinstance(card_number, str):
            return False

        # Remove spaces and hyphens
        card_number = re.sub(r'[^\d]', '', card_number)

        # Check basic format
        if not self.regex_validator(self.patterns['credit_card'], card_number):
            return False

        # Luhn algorithm
        def luhn_check(card_num):
            digits = [int(d) for d in card_num[::-1]]
            for i in range(1, len(digits), 2):
                digits[i] *= 2
                if digits[i] > 9:
                    digits[i] -= 9
            return sum(digits) % 10 == 0

        return luhn_check(card_number)

    def validate_rule(self, value: Any, rule: Union[str, Dict[str, Any]],
                      data_type: str = None) -> Tuple[bool, str]:
        """Comprehensive rule validation"""

        if value is None:
            return True, "Value is None"

        # Handle string rules (simple validation types)
        if isinstance(rule, str):
            rule_lower = rule.lower().replace('_', '').replace(' ', '')

            if rule_lower == 'email':
                return self.email_validator(value), "Email validation"
            elif rule_lower in ['phone', 'phonenumber']:
                return self.phone_validator(value), "Phone validation"
            elif rule_lower == 'creditcard':
                return self.credit_card_validator(value), "Credit card validation"
            elif rule_lower in self.patterns:
                return self.regex_validator(self.patterns[rule_lower], str(value)), f"Pattern validation: {rule}"
            else:
                return True, f"No specific validation for rule: {rule}"

        # Handle dictionary rules (complex validation)
        elif isinstance(rule, dict):
            rule_type = rule.get('type', '').lower()

            if rule_type == 'email':
                is_valid = self.email_validator(value)
                if is_valid and rule.get('regex'):
                    is_valid = self.regex_validator(rule['regex'], str(value))
                return is_valid, "Email validation with regex"

            elif rule_type == 'phone_number':
                is_valid = self.phone_validator(value, rule.get('country'))
                if is_valid and rule.get('regex'):
                    is_valid = self.regex_validator(rule['regex'], str(value))
                return is_valid, "Phone validation with regex"

            elif rule_type == 'range':
                min_val = rule.get('min')
                max_val = rule.get('max')
                return self.range_validator(value, min_val, max_val), f"Range validation: {min_val}-{max_val}"

            elif rule_type == 'choice':
                choices = rule.get('value', [])
                return self.choice_validator(value, choices), f"Choice validation: {choices}"

            elif rule_type in ['date', 'date_range']:
                is_valid = self.date_validator(value, rule.get('format'))
                if is_valid and rule_type == 'date_range':
                    # Additional date range validation
                    start_date = rule.get('start')
                    end_date = rule.get('end')
                    if start_date or end_date:
                        try:
                            if isinstance(value, str):
                                value_date = datetime.strptime(value, rule.get('format', '%Y-%m-%d')).date()
                            elif isinstance(value, datetime):
                                value_date = value.date()
                            elif isinstance(value, date):
                                value_date = value
                            else:
                                return False, "Invalid date format"

                            if start_date:
                                start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
                                if value_date < start_dt:
                                    return False, f"Date before minimum: {start_date}"

                            if end_date:
                                end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
                                if value_date > end_dt:
                                    return False, f"Date after maximum: {end_date}"
                        except Exception as e:
                            return False, f"Date range validation error: {e}"

                return is_valid, "Date validation"

            elif rule_type == 'length':
                min_len = rule.get('min_length')
                max_len = rule.get('max_length')
                return self.length_validator(str(value), min_len, max_len), f"Length validation: {min_len}-{max_len}"

            elif rule.get('regex'):
                return self.regex_validator(rule['regex'], str(value)), f"Regex validation: {rule['regex']}"

            else:
                return True, f"No specific validation for rule type: {rule_type}"

        return True, "No validation rule applied"

    def validate_data_type(self, value: Any, expected_type: str) -> Tuple[bool, str]:
        """Validate that value matches expected data type"""

        if value is None:
            return True, "Null value accepted"

        type_mapping = {
            'int': int,
            'integer': int,
            'float': (int, float),
            'double': (int, float),
            'decimal': (int, float),
            'str': str,
            'string': str,
            'text': str,
            'bool': bool,
            'boolean': bool,
            'date': (date, datetime, str),
            'datetime': (date, datetime, str),
            'timestamp': (date, datetime, str)
        }

        expected_python_type = type_mapping.get(expected_type.lower())

        if expected_python_type is None:
            return True, f"Unknown data type: {expected_type}"

        if isinstance(value, expected_python_type):
            return True, f"Valid {expected_type}"

        # Try type conversion for strings
        if expected_type.lower() in ['int', 'integer'] and isinstance(value, str):
            try:
                int(value)
                return True, "String convertible to int"
            except ValueError:
                return False, f"String '{value}' not convertible to int"

        elif expected_type.lower() in ['float', 'double', 'decimal'] and isinstance(value, str):
            try:
                float(value)
                return True, "String convertible to float"
            except ValueError:
                return False, f"String '{value}' not convertible to float"

        elif expected_type.lower() in ['bool', 'boolean'] and isinstance(value, str):
            if value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
                return True, "String convertible to bool"
            return False, f"String '{value}' not convertible to bool"

        return False, f"Value type {type(value).__name__} does not match expected type {expected_type}"

    def validate_constraints(self, row, table_metadata):
        """Validate that a row meets all defined constraints"""
        columns = table_metadata["columns"]

        for column in columns:
            column_name = column["name"]
            value = row.get(column_name)

            # Check nullable constraint
            if not column.get("nullable", True) and value is None:
                return False, f"Column {column_name} cannot be null"

            # Check data type
            expected_type = column["type"]
            if value is not None:
                if expected_type == "int" and not isinstance(value, int):
                    return False, f"Column {column_name} must be integer. Data: {value} {type(value)}"
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    return False, f"Column {column_name} must be numeric"
                elif expected_type == "str" and not isinstance(value, str):
                    return False, f"Column {column_name} must be string"
                elif expected_type == "bool" and not isinstance(value, bool):
                    return False, f"Column {column_name} must be boolean"

            # Check rule constraints
            rule = column.get("rule")
            if rule and value is not None:
                if isinstance(rule, dict):
                    if rule.get("type") == "range":
                        min_val = rule.get("min")
                        max_val = rule.get("max")
                        if min_val is not None and value < min_val:
                            return False, f"Column {column_name} value {value} below minimum {min_val}"
                        if max_val is not None and value > max_val:
                            return False, f"Column {column_name} value {value} above maximum {max_val}"
                    elif rule.get("type") == "choice":
                        valid_choices = rule.get("value", [])
                        if value not in valid_choices:
                            return False, f"Column {column_name} value {value} not in valid choices {valid_choices}"
                    elif rule.get("regex"):
                        if not self.regex_validator(rule["regex"], str(value)):
                            return False, f"Column {column_name} value {value} does not match regex {rule['regex']}"

        return True, "Valid"

    def validate_record(self, record: Dict[str, Any],
                        table_metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate entire record against table metadata"""

        errors = []
        columns = table_metadata.get('columns', [])

        for column in columns:
            column_name = column['name']
            expected_type = column['type']
            nullable = column.get('nullable', True)
            rule = column.get('rule')

            value = record.get(column_name)

            # Check nullable constraint
            if not nullable and value is None:
                errors.append(f"Column '{column_name}' cannot be null")
                continue

            if value is not None:
                # Validate data type
                type_valid, type_msg = self.validate_data_type(value, expected_type)
                if not type_valid:
                    errors.append(f"Column '{column_name}': {type_msg}")

                # Validate rule if present
                if rule:
                    rule_valid, rule_msg = self.validate_rule(value, rule, expected_type)
                    if not rule_valid:
                        errors.append(f"Column '{column_name}': {rule_msg}")

        return len(errors) == 0, errors

    def validate_batch(self, batch: List[Dict[str, Any]],
                       table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate batch of records"""

        results = {
            'total_records': len(batch),
            'valid_records': 0,
            'invalid_records': 0,
            'errors': [],
            'valid_indices': [],
            'invalid_indices': []
        }

        for i, record in enumerate(batch):
            is_valid, errors = self.validate_record(record, table_metadata)

            if is_valid:
                results['valid_records'] += 1
                results['valid_indices'].append(i)
            else:
                results['invalid_records'] += 1
                results['invalid_indices'].append(i)
                results['errors'].extend([f"Record {i}: {error}" for error in errors])

        return results