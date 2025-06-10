"""
PII Data Masking Engine for Synthetic Data Platform
Handles static and dynamic masking based on sensitivity levels
"""

import re
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass
from faker import Faker
import pandas as pd

logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Data sensitivity classification"""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    PII = "PII"
    FINANCIAL = "FINANCIAL"
    MEDICAL = "MEDICAL"
    CONFIDENTIAL = "CONFIDENTIAL"


class MaskingMethod(Enum):
    """Available masking methods"""
    HASH = "hash"
    REDACT = "redact"
    SHUFFLE = "shuffle"
    SUBSTITUTE = "substitute"
    ANONYMIZE = "anonymize"
    ENCRYPT = "encrypt"
    FORMAT_PRESERVE = "format_preserve"


@dataclass
class MaskingRule:
    """Configuration for masking a specific field"""
    field_name: str
    sensitivity: SensitivityLevel
    method: MaskingMethod
    preserve_format: bool = True
    preserve_nulls: bool = True
    custom_mask_char: str = "*"
    salt: Optional[str] = None


class MaskingEngine:
    """
    Advanced data masking engine with support for:
    - Static masking (deterministic)
    - Dynamic masking (random)
    - Format preservation
    - Multiple masking methods
    """

    def __init__(self, locale: str = "en_US", seed: Optional[int] = None):
        self.locale = locale
        self.faker = Faker(locale)
        if seed:
            Faker.seed(seed)

        self.default_rules = self._load_default_rules()
        self.custom_rules: Dict[str, MaskingRule] = {}

        # Masking method registry
        self.masking_methods = {
            MaskingMethod.HASH: self._hash_mask,
            MaskingMethod.REDACT: self._redact_mask,
            MaskingMethod.SHUFFLE: self._shuffle_mask,
            MaskingMethod.SUBSTITUTE: self._substitute_mask,
            MaskingMethod.ANONYMIZE: self._anonymize_mask,
            MaskingMethod.FORMAT_PRESERVE: self._format_preserve_mask,
        }

    def _load_default_rules(self) -> Dict[SensitivityLevel, MaskingRule]:
        """Load default masking rules for different sensitivity levels"""
        return {
            SensitivityLevel.PII: MaskingRule(
                field_name="default_pii",
                sensitivity=SensitivityLevel.PII,
                method=MaskingMethod.ANONYMIZE,
                preserve_format=True
            ),
            SensitivityLevel.FINANCIAL: MaskingRule(
                field_name="default_financial",
                sensitivity=SensitivityLevel.FINANCIAL,
                method=MaskingMethod.FORMAT_PRESERVE,
                preserve_format=True
            ),
            SensitivityLevel.MEDICAL: MaskingRule(
                field_name="default_medical",
                sensitivity=SensitivityLevel.MEDICAL,
                method=MaskingMethod.HASH,
                preserve_format=False
            ),
            SensitivityLevel.CONFIDENTIAL: MaskingRule(
                field_name="default_confidential",
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                method=MaskingMethod.REDACT,
                preserve_format=False
            )
        }

    def add_masking_rule(self, rule: MaskingRule) -> None:
        """Add custom masking rule for specific field"""
        self.custom_rules[rule.field_name] = rule
        logger.info(f"Added masking rule for field: {rule.field_name}")

    def mask_dataframe(self, df: pd.DataFrame, schema_config: Dict) -> pd.DataFrame:
        """
        Apply masking to entire dataframe based on schema configuration

        Args:
            df: DataFrame to mask
            schema_config: Schema configuration with sensitivity metadata

        Returns:
            Masked DataFrame
        """
        masked_df = df.copy()

        for table in schema_config.get('tables', []):
            table_name = table['table_name']

            for column in table.get('columns', []):
                column_name = column['name']
                sensitivity = column.get('sensitivity')

                if sensitivity and column_name in masked_df.columns:
                    sensitivity_level = SensitivityLevel(sensitivity)
                    masked_df[column_name] = self._mask_column(
                        masked_df[column_name],
                        column_name,
                        sensitivity_level,
                        column
                    )

        return masked_df

    def _mask_column(self, series: pd.Series, column_name: str,
                    sensitivity: SensitivityLevel, column_config: Dict) -> pd.Series:
        """Apply masking to a specific column"""

        # Get masking rule (custom or default)
        rule = self.custom_rules.get(column_name)
        if not rule:
            rule = self.default_rules.get(sensitivity)
            if rule:
                rule.field_name = column_name

        if not rule:
            logger.warning(f"No masking rule found for {column_name}, skipping")
            return series

        # Apply masking method
        masking_func = self.masking_methods.get(rule.method)
        if not masking_func:
            logger.error(f"Unknown masking method: {rule.method}")
            return series

        logger.info(f"Masking column {column_name} with method {rule.method}")
        return series.apply(lambda x: masking_func(x, rule, column_config))

    def _hash_mask(self, value: Any, rule: MaskingRule, config: Dict) -> str:
        """Hash-based masking (deterministic)"""
        if pd.isna(value) and rule.preserve_nulls:
            return value

        salt = rule.salt or rule.field_name
        hash_input = f"{str(value)}{salt}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _redact_mask(self, value: Any, rule: MaskingRule, config: Dict) -> str:
        """Replace with redaction characters"""
        if pd.isna(value) and rule.preserve_nulls:
            return value

        if rule.preserve_format and isinstance(value, str):
            # Preserve format but redact content
            masked = ""
            for char in str(value):
                if char.isalnum():
                    masked += rule.custom_mask_char
                else:
                    masked += char
            return masked
        else:
            return rule.custom_mask_char * 8

    def _shuffle_mask(self, value: Any, rule: MaskingRule, config: Dict) -> str:
        """Shuffle characters/digits while preserving format"""
        if pd.isna(value) and rule.preserve_nulls:
            return value

        str_value = str(value)
        if rule.preserve_format:
            chars = list(str_value)
            # Only shuffle alphanumeric characters
            alpha_chars = [c for c in chars if c.isalnum()]
            self.faker.random.shuffle(alpha_chars)

            result = []
            alpha_idx = 0
            for char in chars:
                if char.isalnum() and alpha_idx < len(alpha_chars):
                    result.append(alpha_chars[alpha_idx])
                    alpha_idx += 1
                else:
                    result.append(char)
            return ''.join(result)
        else:
            chars = list(str_value)
            self.faker.random.shuffle(chars)
            return ''.join(chars)

    def _substitute_mask(self, value: Any, rule: MaskingRule, config: Dict) -> Any:
        """Substitute with fake data of same type"""
        if pd.isna(value) and rule.preserve_nulls:
            return value

        data_type = config.get('type', 'str')
        faker_rule = config.get('rule')

        if isinstance(faker_rule, str):
            # Simple faker method
            return getattr(self.faker, faker_rule)()
        elif isinstance(faker_rule, dict):
            # Complex faker rule
            faker_type = faker_rule.get('type')
            if faker_type == 'email':
                return self.faker.email()
            elif faker_type == 'phone_number':
                return self.faker.phone_number()
            elif faker_type == 'range':
                min_val = faker_rule.get('min', 0)
                max_val = faker_rule.get('max', 100)
                if data_type == 'int':
                    return self.faker.random_int(min=min_val, max=max_val)
                else:
                    return round(self.faker.random.uniform(min_val, max_val), 2)

        # Fallback based on data type
        if data_type == 'str':
            return self.faker.word()
        elif data_type == 'int':
            return self.faker.random_int()
        elif data_type == 'float':
            return round(self.faker.random.uniform(0, 1000), 2)
        else:
            return str(value)

    def _anonymize_mask(self, value: Any, rule: MaskingRule, config: Dict) -> Any:
        """Smart anonymization based on data patterns"""
        if pd.isna(value) and rule.preserve_nulls:
            return value

        str_value = str(value)

        # Email anonymization
        if re.match(r'^[^@]+@[^@]+\.[^@]+$', str_value):
            return self._anonymize_email(str_value)

        # Phone number anonymization
        if re.match(r'^[\+]?[0-9\-\(\)\s]{10,}$', str_value):
            return self._anonymize_phone(str_value)

        # Credit card anonymization
        if re.match(r'^[0-9\-\s]{13,19}$', str_value.replace(' ', '').replace('-', '')):
            return self._anonymize_credit_card(str_value)

        # Default anonymization
        return self._substitute_mask(value, rule, config)

    def _format_preserve_mask(self, value: Any, rule: MaskingRule, config: Dict) -> str:
        """Preserve exact format but change content"""
        if pd.isna(value) and rule.preserve_nulls:
            return value

        str_value = str(value)
        result = ""

        for char in str_value:
            if char.isdigit():
                result += str(self.faker.random_digit())
            elif char.isalpha():
                if char.isupper():
                    result += self.faker.random_uppercase_letter()
                else:
                    result += self.faker.random_lowercase_letter()
            else:
                result += char

        return result

    def _anonymize_email(self, email: str) -> str:
        """Anonymize email while preserving domain structure"""
        local, domain = email.split('@')

        # Generate new local part
        new_local = self.faker.user_name()

        # Optionally preserve domain or anonymize it too
        domain_parts = domain.split('.')
        if len(domain_parts) > 1:
            # Keep TLD, randomize domain name
            new_domain = f"{self.faker.word()}.{domain_parts[-1]}"
        else:
            new_domain = f"{self.faker.word()}.com"

        return f"{new_local}@{new_domain}"

    def _anonymize_phone(self, phone: str) -> str:
        """Anonymize phone number while preserving format"""
        # Extract format (non-digit characters)
        format_chars = re.findall(r'[^\d]', phone)

        # Generate new digits
        digits = [str(self.faker.random_digit()) for _ in re.findall(r'\d', phone)]

        # Reconstruct with original format
        result = ""
        digit_idx = 0
        for char in phone:
            if char.isdigit():
                result += digits[digit_idx]
                digit_idx += 1
            else:
                result += char

        return result

    def _anonymize_credit_card(self, cc: str) -> str:
        """Anonymize credit card while preserving format"""
        # Keep first 4 and last 4 digits, mask middle
        digits_only = re.sub(r'[^\d]', '', cc)
        if len(digits_only) >= 8:
            new_digits = (digits_only[:4] +
                         ''.join([str(self.faker.random_digit())
                                for _ in range(len(digits_only) - 8)]) +
                         digits_only[-4:])
        else:
            new_digits = ''.join([str(self.faker.random_digit())
                                for _ in range(len(digits_only))])

        # Restore original format
        result = ""
        digit_idx = 0
        for char in cc:
            if char.isdigit():
                result += new_digits[digit_idx]
                digit_idx += 1
            else:
                result += char

        return result

    def validate_masking_config(self, schema_config: Dict) -> List[str]:
        """Validate masking configuration and return any issues"""
        issues = []

        for table in schema_config.get('tables', []):
            for column in table.get('columns', []):
                sensitivity = column.get('sensitivity')
                if sensitivity:
                    try:
                        SensitivityLevel(sensitivity)
                    except ValueError:
                        issues.append(
                            f"Invalid sensitivity level '{sensitivity}' "
                            f"for column {column['name']}"
                        )

        return issues


# Utility functions for easy integration
def create_masking_engine(config: Dict) -> MaskingEngine:
    """Factory function to create masking engine from config"""
    locale = config.get('locale', 'en_US')
    seed = config.get('seed')

    engine = MaskingEngine(locale=locale, seed=seed)

    # Add any custom rules from config
    custom_rules = config.get('masking_rules', {})
    for field_name, rule_config in custom_rules.items():
        rule = MaskingRule(
            field_name=field_name,
            sensitivity=SensitivityLevel(rule_config['sensitivity']),
            method=MaskingMethod(rule_config.get('method', 'anonymize')),
            preserve_format=rule_config.get('preserve_format', True),
            preserve_nulls=rule_config.get('preserve_nulls', True),
            custom_mask_char=rule_config.get('mask_char', '*'),
            salt=rule_config.get('salt')
        )
        engine.add_masking_rule(rule)

    return engine


if __name__ == "__main__":
    # Example usage
    engine = MaskingEngine()

    # Sample data
    data = {
        'email': ['john.doe@example.com', 'jane@test.org'],
        'phone': ['+1-555-123-4567', '555.987.6543'],
        'income': [50000.0, 75000.0]
    }
    df = pd.DataFrame(data)

    # Sample schema config
    schema_config = {
        'tables': [{
            'table_name': 'users',
            'columns': [
                {'name': 'email', 'sensitivity': 'PII'},
                {'name': 'phone', 'sensitivity': 'PII'},
                {'name': 'income', 'sensitivity': 'FINANCIAL'}
            ]
        }]
    }

    masked_df = engine.mask_dataframe(df, schema_config)
    print("Original:")
    print(df)
    print("\nMasked:")
    print(masked_df)