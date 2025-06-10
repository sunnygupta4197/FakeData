"""
Data Masking and Anonymization Engine
Supports static and dynamic masking for PII, FINANCIAL, and sensitive data fields.
"""

import hashlib
import random
import re
import string
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from faker import Faker

logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Data sensitivity classification levels."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    PII = "PII"
    FINANCIAL = "FINANCIAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class MaskingMethod(Enum):
    """Available masking methods."""
    HASH = "hash"
    TOKENIZE = "tokenize"
    SHUFFLE = "shuffle"
    NULLIFY = "nullify"
    REDACT = "redact"
    SUBSTITUTE = "substitute"
    PRESERVE_FORMAT = "preserve_format"
    ENCRYPT = "encrypt"
    PSEUDONYMIZE = "pseudonymize"


@dataclass
class MaskingRule:
    """Configuration for field-specific masking rules."""
    field_name: str
    sensitivity_level: SensitivityLevel
    masking_method: MaskingMethod
    preserve_format: bool = False
    custom_logic: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseMasker(ABC):
    """Base class for all masking implementations."""

    @abstractmethod
    def mask(self, value: Any, rule: MaskingRule) -> Any:
        """Apply masking to a single value."""
        pass


class HashMasker(BaseMasker):
    """Hash-based masking for irreversible anonymization."""

    def __init__(self, salt: str = "default_salt", algorithm: str = "sha256"):
        self.salt = salt
        self.algorithm = algorithm

    def mask(self, value: Any, rule: MaskingRule) -> str:
        """Hash the value with salt."""
        if value is None:
            return None

        hash_input = f"{self.salt}{str(value)}"
        hash_obj = hashlib.new(self.algorithm)
        hash_obj.update(hash_input.encode('utf-8'))

        if rule.preserve_format and isinstance(value, str):
            # Preserve original format but with hashed content
            return self._preserve_format(value, hash_obj.hexdigest())

        return hash_obj.hexdigest()

    def _preserve_format(self, original: str, hashed: str) -> str:
        """Preserve the format of the original string."""
        result = []
        hash_idx = 0

        for char in original:
            if hash_idx >= len(hashed):
                hash_idx = 0

            if char.isalpha():
                result.append(hashed[hash_idx].upper() if char.isupper() else hashed[hash_idx].lower())
            elif char.isdigit():
                # Use numeric representation of hash character
                result.append(str(ord(hashed[hash_idx]) % 10))
            else:
                result.append(char)

            hash_idx += 1

        return ''.join(result)


class TokenizeMasker(BaseMasker):
    """Token-based masking with reversible mapping."""

    def __init__(self):
        self.token_map: Dict[str, str] = {}
        self.reverse_map: Dict[str, str] = {}
        self.token_counter = 1000

    def mask(self, value: Any, rule: MaskingRule) -> str:
        """Replace value with a consistent token."""
        if value is None:
            return None

        str_value = str(value).strip()

        if str_value in self.token_map:
            return self.token_map[str_value]

        # Generate new token
        token_prefix = rule.parameters.get('token_prefix', 'TOKEN')
        token = f"{token_prefix}_{self.token_counter:06d}"

        self.token_map[str_value] = token
        self.reverse_map[token] = str_value
        self.token_counter += 1

        return token

    def reverse_mask(self, token: str) -> Optional[str]:
        """Reverse the tokenization if possible."""
        return self.reverse_map.get(token)


class ShuffleMasker(BaseMasker):
    """Shuffle characters or values while preserving data type."""

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)

    def mask(self, value: Any, rule: MaskingRule) -> Any:
        """Shuffle the value while preserving its structure."""
        if value is None:
            return None

        if isinstance(value, str):
            return self._shuffle_string(value, rule.preserve_format)
        elif isinstance(value, (int, float)):
            return self._shuffle_number(value)
        else:
            return self._shuffle_string(str(value), rule.preserve_format)

    def _shuffle_string(self, value: str, preserve_format: bool) -> str:
        """Shuffle string characters."""
        if preserve_format:
            return self._shuffle_preserve_format(value)
        else:
            chars = list(value)
            self.random.shuffle(chars)
            return ''.join(chars)

    def _shuffle_preserve_format(self, value: str) -> str:
        """Shuffle while preserving character types and positions."""
        letters = []
        digits = []

        # Extract letters and digits
        for char in value:
            if char.isalpha():
                letters.append(char)
            elif char.isdigit():
                digits.append(char)

        # Shuffle extracted characters
        self.random.shuffle(letters)
        self.random.shuffle(digits)

        # Reconstruct string with shuffled characters
        result = []
        letter_idx = digit_idx = 0

        for char in value:
            if char.isalpha() and letter_idx < len(letters):
                result.append(letters[letter_idx])
                letter_idx += 1
            elif char.isdigit() and digit_idx < len(digits):
                result.append(digits[digit_idx])
                digit_idx += 1
            else:
                result.append(char)

        return ''.join(result)

    def _shuffle_number(self, value: Union[int, float]) -> Union[int, float]:
        """Shuffle digits in a number."""
        is_float = isinstance(value, float)
        str_num = str(abs(value)).replace('.', '')
        digits = list(str_num)
        self.random.shuffle(digits)
        shuffled_str = ''.join(digits)

        if is_float:
            # Insert decimal point at random position
            decimal_pos = self.random.randint(1, len(shuffled_str))
            shuffled_str = shuffled_str[:decimal_pos] + '.' + shuffled_str[decimal_pos:]
            result = float(shuffled_str)
        else:
            result = int(shuffled_str)

        return -result if value < 0 else result


class SubstituteMasker(BaseMasker):
    """Substitute with realistic fake data."""

    def __init__(self, locale: str = 'en_US'):
        self.faker = Faker(locale)

    def mask(self, value: Any, rule: MaskingRule) -> Any:
        """Substitute with appropriate fake data."""
        if value is None:
            return None

        field_name = rule.field_name.lower()

        # Email substitution
        if 'email' in field_name:
            return self.faker.email()

        # Name substitution
        elif 'first_name' in field_name or 'fname' in field_name:
            return self.faker.first_name()
        elif 'last_name' in field_name or 'lname' in field_name:
            return self.faker.last_name()
        elif 'name' in field_name:
            return self.faker.name()

        # Phone substitution
        elif 'phone' in field_name:
            return self.faker.phone_number()

        # Address substitution
        elif 'address' in field_name:
            return self.faker.address()
        elif 'city' in field_name:
            return self.faker.city()
        elif 'state' in field_name:
            return self.faker.state()
        elif 'zip' in field_name or 'postal' in field_name:
            return self.faker.postcode()

        # Financial substitution
        elif 'ssn' in field_name or 'social' in field_name:
            return self.faker.ssn()
        elif 'credit_card' in field_name or 'cc_number' in field_name:
            return self.faker.credit_card_number()
        elif 'account' in field_name and 'number' in field_name:
            return self.faker.bban()

        # Date substitution
        elif 'birth' in field_name or 'dob' in field_name:
            return self.faker.date_of_birth()
        elif 'date' in field_name:
            return self.faker.date()

        # Default: preserve type but generate fake content
        elif isinstance(value, str):
            return self.faker.word()
        elif isinstance(value, int):
            return self.faker.random_int(min=1, max=999999)
        elif isinstance(value, float):
            return round(self.faker.random.uniform(1.0, 999999.99), 2)

        return str(self.faker.word())


class PseudonymizeMasker(BaseMasker):
    """Pseudonymization with consistent mapping."""

    def __init__(self, seed: Optional[int] = None):
        self.faker = Faker()
        if seed:
            Faker.seed(seed)
        self.mapping: Dict[str, Any] = {}

    def mask(self, value: Any, rule: MaskingRule) -> Any:
        """Create consistent pseudonyms."""
        if value is None:
            return None

        key = f"{rule.field_name}:{str(value)}"

        if key in self.mapping:
            return self.mapping[key]

        # Generate consistent pseudonym based on field type
        field_name = rule.field_name.lower()

        if 'email' in field_name:
            pseudo = self.faker.email()
        elif 'name' in field_name:
            pseudo = self.faker.name()
        elif 'phone' in field_name:
            pseudo = self.faker.phone_number()
        elif 'address' in field_name:
            pseudo = self.faker.address()
        elif isinstance(value, (int, float)):
            pseudo = self.faker.random_int(min=1, max=999999) if isinstance(value, int) else round(self.faker.random.uniform(1.0, 999999.99), 2)
        else:
            pseudo = self.faker.word()

        self.mapping[key] = pseudo
        return pseudo


class MaskingEngine:
    """Main masking engine orchestrating different masking strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rules: Dict[str, MaskingRule] = {}
        self.maskers = self._initialize_maskers()

    def _initialize_maskers(self) -> Dict[MaskingMethod, BaseMasker]:
        """Initialize all available maskers."""
        salt = self.config.get('salt', 'default_salt')
        seed = self.config.get('seed', None)
        locale = self.config.get('locale', 'en_US')

        return {
            MaskingMethod.HASH: HashMasker(salt=salt),
            MaskingMethod.TOKENIZE: TokenizeMasker(),
            MaskingMethod.SHUFFLE: ShuffleMasker(seed=seed),
            MaskingMethod.SUBSTITUTE: SubstituteMasker(locale=locale),
            MaskingMethod.PSEUDONYMIZE: PseudonymizeMasker(seed=seed),
        }

    def add_rule(self, rule: MaskingRule) -> None:
        """Add a masking rule."""
        self.rules[rule.field_name] = rule
        logger.debug(f"Added masking rule for field '{rule.field_name}': {rule.masking_method.value}")

    def add_rules_from_config(self, config: Dict[str, Any]) -> None:
        """Add masking rules from configuration."""
        for table_config in config.get('tables', []):
            table_name = table_config.get('table_name')

            for column_config in table_config.get('columns', []):
                sensitivity = column_config.get('sensitivity')
                if sensitivity:
                    field_name = f"{table_name}.{column_config['name']}"
                    masking_method = self._get_masking_method(sensitivity, column_config)

                    rule = MaskingRule(
                        field_name=field_name,
                        sensitivity_level=SensitivityLevel(sensitivity),
                        masking_method=masking_method,
                        preserve_format=column_config.get('preserve_format', False),
                        parameters=column_config.get('masking_parameters', {})
                    )

                    self.add_rule(rule)

    def _get_masking_method(self, sensitivity: str, column_config: Dict[str, Any]) -> MaskingMethod:
        """Determine appropriate masking method based on sensitivity and configuration."""
        method_mapping = {
            'PII': MaskingMethod.PSEUDONYMIZE,
            'FINANCIAL': MaskingMethod.HASH,
            'CONFIDENTIAL': MaskingMethod.TOKENIZE,
            'RESTRICTED': MaskingMethod.HASH,
        }

        # Check for explicit masking method in config
        explicit_method = column_config.get('masking_method')
        if explicit_method:
            try:
                return MaskingMethod(explicit_method)
            except ValueError:
                logger.warning(f"Invalid masking method '{explicit_method}', using default")

        return method_mapping.get(sensitivity, MaskingMethod.SUBSTITUTE)

    def mask_value(self, table_name: str, field_name: str, value: Any) -> Any:
        """Mask a single value based on configured rules."""
        full_field_name = f"{table_name}.{field_name}"
        rule = self.rules.get(full_field_name)

        if not rule:
            return value  # No masking rule, return original value

        try:
            if rule.custom_logic:
                return rule.custom_logic(value, rule)

            if rule.masking_method == MaskingMethod.NULLIFY:
                return None
            elif rule.masking_method == MaskingMethod.REDACT:
                return "***REDACTED***" if value else value
            else:
                masker = self.maskers.get(rule.masking_method)
                if masker:
                    return masker.mask(value, rule)
                else:
                    logger.warning(f"No masker found for method {rule.masking_method}")
                    return value

        except Exception as e:
            logger.error(f"Error masking field {full_field_name}: {str(e)}")
            return value  # Return original value on error

    def mask_row(self, table_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
        """Mask an entire row of data."""
        masked_row = {}

        for field_name, value in row.items():
            masked_row[field_name] = self.mask_value(table_name, field_name, value)

        return masked_row

    def mask_dataset(self, table_name: str, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mask an entire dataset."""
        logger.info(f"Starting masking for table '{table_name}' with {len(dataset)} rows")

        masked_dataset = []
        for i, row in enumerate(dataset):
            try:
                masked_row = self.mask_row(table_name, row)
                masked_dataset.append(masked_row)

                if (i + 1) % 10000 == 0:  # Log progress for large datasets
                    logger.info(f"Masked {i + 1}/{len(dataset)} rows")

            except Exception as e:
                logger.error(f"Error masking row {i} in table {table_name}: {str(e)}")
                masked_dataset.append(row)  # Keep original row on error

        logger.info(f"Completed masking for table '{table_name}'")
        return masked_dataset

    def get_masking_summary(self) -> Dict[str, Any]:
        """Get summary of masking configuration."""
        summary = {
            'total_rules': len(self.rules),
            'rules_by_sensitivity': {},
            'rules_by_method': {},
            'configured_tables': set()
        }

        for rule in self.rules.values():
            # Count by sensitivity
            sensitivity = rule.sensitivity_level.value
            summary['rules_by_sensitivity'][sensitivity] = summary['rules_by_sensitivity'].get(sensitivity, 0) + 1

            # Count by method
            method = rule.masking_method.value
            summary['rules_by_method'][method] = summary['rules_by_method'].get(method, 0) + 1

            # Track tables
            table_name = rule.field_name.split('.')[0]
            summary['configured_tables'].add(table_name)

        summary['configured_tables'] = list(summary['configured_tables'])
        return summary