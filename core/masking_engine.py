"""
Data Masking and Anonymization Engine

Supports static/dynamic masking, PII handling, and configurable anonymization strategies.
"""

import hashlib
import random
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass
from faker import Faker

logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Data sensitivity classification"""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    PII = "PII"
    FINANCIAL = "FINANCIAL"
    HEALTHCARE = "HEALTHCARE"
    CONFIDENTIAL = "CONFIDENTIAL"


class MaskingStrategy(Enum):
    """Available masking strategies"""
    HASH = "hash"
    SHUFFLE = "shuffle"
    NULLIFY = "nullify"
    TOKENIZE = "tokenize"
    PARTIAL_MASK = "partial_mask"
    FORMAT_PRESERVE = "format_preserve"
    FAKE_REPLACE = "fake_replace"
    RANGE_SHIFT = "range_shift"


@dataclass
class MaskingRule:
    """Configuration for field-specific masking"""
    field_name: str
    sensitivity_level: SensitivityLevel
    strategy: MaskingStrategy
    strategy_params: Dict[str, Any] = None
    condition: Optional[str] = None  # SQL-like condition for conditional masking
    preserve_format: bool = True
    deterministic: bool = True  # Same input -> same output


class BaseMaskingProvider(ABC):
    """Abstract base class for masking providers"""

    @abstractmethod
    def mask(self, value: Any, rule: MaskingRule) -> Any:
        """Apply masking to a value"""
        pass

    @abstractmethod
    def can_handle(self, rule: MaskingRule) -> bool:
        """Check if provider can handle the masking rule"""
        pass


class HashMaskingProvider(BaseMaskingProvider):
    """Hash-based masking provider"""

    def __init__(self, salt: str = "default_salt"):
        self.salt = salt

    def mask(self, value: Any, rule: MaskingRule) -> str:
        if value is None:
            return None

        # Convert to string and hash
        value_str = str(value)
        if rule.deterministic:
            hash_obj = hashlib.sha256((value_str + self.salt).encode())
            return hash_obj.hexdigest()[:16]  # Truncate for readability
        else:
            return hashlib.sha256((value_str + str(random.random())).encode()).hexdigest()[:16]

    def can_handle(self, rule: MaskingRule) -> bool:
        return rule.strategy == MaskingStrategy.HASH


class ShuffleMaskingProvider(BaseMaskingProvider):
    """Shuffle-based masking provider"""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def mask(self, value: Any, rule: MaskingRule) -> Any:
        if value is None:
            return None

        value_str = str(value)
        char_list = list(value_str)

        if rule.deterministic:
            # Use hash as seed for deterministic shuffling
            hash_seed = int(hashlib.md5(value_str.encode()).hexdigest()[:8], 16)
            random.seed(hash_seed)

        random.shuffle(char_list)
        return ''.join(char_list)

    def can_handle(self, rule: MaskingRule) -> bool:
        return rule.strategy == MaskingStrategy.SHUFFLE


class PartialMaskingProvider(BaseMaskingProvider):
    """Partial masking provider (e.g., show first/last N characters)"""

    def mask(self, value: Any, rule: MaskingRule) -> str:
        if value is None:
            return None

        value_str = str(value)
        params = rule.strategy_params or {}

        preserve_start = params.get('preserve_start', 2)
        preserve_end = params.get('preserve_end', 2)
        mask_char = params.get('mask_char', '*')

        if len(value_str) <= preserve_start + preserve_end:
            return mask_char * len(value_str)

        start = value_str[:preserve_start]
        end = value_str[-preserve_end:] if preserve_end > 0 else ''
        middle_len = len(value_str) - preserve_start - preserve_end

        return start + mask_char * middle_len + end

    def can_handle(self, rule: MaskingRule) -> bool:
        return rule.strategy == MaskingStrategy.PARTIAL_MASK


class FakeReplaceMaskingProvider(BaseMaskingProvider):
    """Replace with fake data provider"""

    def __init__(self, locale: str = 'en_US'):
        self.faker = Faker(locale)
        self.faker_methods = {
            'email': self.faker.email,
            'phone': self.faker.phone_number,
            'name': self.faker.name,
            'first_name': self.faker.first_name,
            'last_name': self.faker.last_name,
            'address': self.faker.address,
            'ssn': self.faker.ssn,
            'credit_card': self.faker.credit_card_number,
            'company': self.faker.company,
            'job': self.faker.job,
        }

    def mask(self, value: Any, rule: MaskingRule) -> Any:
        if value is None:
            return None

        params = rule.strategy_params or {}
        fake_type = params.get('fake_type', 'name')

        if rule.deterministic:
            # Use hash as seed for deterministic fake data
            hash_seed = int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16)
            self.faker.seed_instance(hash_seed)

        faker_method = self.faker_methods.get(fake_type, self.faker.word)
        return faker_method()

    def can_handle(self, rule: MaskingRule) -> bool:
        return rule.strategy == MaskingStrategy.FAKE_REPLACE


class FormatPreserveMaskingProvider(BaseMaskingProvider):
    """Format-preserving masking provider"""

    def mask(self, value: Any, rule: MaskingRule) -> str:
        if value is None:
            return None

        value_str = str(value)
        params = rule.strategy_params or {}

        # Preserve format patterns (digits, letters, special chars)
        pattern = ''
        for char in value_str:
            if char.isdigit():
                pattern += random.choice('0123456789')
            elif char.isalpha():
                pattern += random.choice('abcdefghijklmnopqrstuvwxyz')
            else:
                pattern += char

        return pattern

    def can_handle(self, rule: MaskingRule) -> bool:
        return rule.strategy == MaskingStrategy.FORMAT_PRESERVE


class RangeShiftMaskingProvider(BaseMaskingProvider):
    """Numeric range shifting provider"""

    def mask(self, value: Any, rule: MaskingRule) -> Union[int, float]:
        if value is None:
            return None

        try:
            num_value = float(value)
            params = rule.strategy_params or {}
            shift_range = params.get('shift_range', 0.1)  # 10% shift by default

            if rule.deterministic:
                # Use hash for deterministic shifting
                hash_seed = int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16)
                random.seed(hash_seed)

            shift = random.uniform(-shift_range, shift_range) * num_value
            result = num_value + shift

            # Preserve integer type if input was integer
            if isinstance(value, int):
                return int(result)
            return result

        except (ValueError, TypeError):
            logger.warning(f"Cannot apply range shift to non-numeric value: {value}")
            return value

    def can_handle(self, rule: MaskingRule) -> bool:
        return rule.strategy == MaskingStrategy.RANGE_SHIFT


class MaskingEngine:
    """Main masking engine that orchestrates different masking providers"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers: List[BaseMaskingProvider] = []
        self.rules: Dict[str, MaskingRule] = {}
        self.field_cache: Dict[str, Dict[str, Any]] = {}  # For deterministic masking

        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all masking providers"""
        salt = self.config.get('hash_salt', 'default_salt')
        locale = self.config.get('locale', 'en_US')

        self.providers = [
            HashMaskingProvider(salt=salt),
            ShuffleMaskingProvider(),
            PartialMaskingProvider(),
            FakeReplaceMaskingProvider(locale=locale),
            FormatPreserveMaskingProvider(),
            RangeShiftMaskingProvider(),
        ]

    def add_masking_rule(self, rule: MaskingRule):
        """Add a masking rule for a specific field"""
        self.rules[rule.field_name] = rule
        logger.info(f"Added masking rule for field '{rule.field_name}': {rule.strategy.value}")

    def add_masking_rules_from_config(self, rules_config: List[Dict[str, Any]]):
        """Add masking rules from configuration"""
        for rule_config in rules_config:
            rule = MaskingRule(
                field_name=rule_config['field_name'],
                sensitivity_level=SensitivityLevel(rule_config['sensitivity_level']),
                strategy=MaskingStrategy(rule_config['strategy']),
                strategy_params=rule_config.get('strategy_params', {}),
                condition=rule_config.get('condition'),
                preserve_format=rule_config.get('preserve_format', True),
                deterministic=rule_config.get('deterministic', True)
            )
            self.add_masking_rule(rule)

    def mask_value(self, field_name: str, value: Any, row_context: Dict[str, Any] = None) -> Any:
        """Mask a single value based on field rules"""
        if field_name not in self.rules:
            return value

        rule = self.rules[field_name]

        # Check condition if specified
        if rule.condition and row_context:
            if not self._evaluate_condition(rule.condition, row_context):
                return value  # Don't mask if condition not met

        # Find appropriate provider
        provider = self._get_provider_for_rule(rule)
        if not provider:
            logger.warning(f"No provider found for masking strategy: {rule.strategy}")
            return value

        try:
            # Use cache for deterministic masking
            if rule.deterministic:
                cache_key = f"{field_name}:{str(value)}"
                if cache_key in self.field_cache:
                    return self.field_cache[cache_key]

                masked_value = provider.mask(value, rule)
                self.field_cache[cache_key] = masked_value
                return masked_value
            else:
                return provider.mask(value, rule)

        except Exception as e:
            logger.error(f"Error masking field '{field_name}' with value '{value}': {str(e)}")
            return value  # Return original value on error

    def mask_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Mask all applicable fields in a row"""
        masked_row = row.copy()

        for field_name in row.keys():
            if field_name in self.rules:
                masked_row[field_name] = self.mask_value(field_name, row[field_name], row)

        return masked_row

    def mask_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mask an entire dataset"""
        logger.info(f"Starting to mask dataset with {len(dataset)} rows")

        masked_dataset = []
        for i, row in enumerate(dataset):
            try:
                masked_row = self.mask_row(row)
                masked_dataset.append(masked_row)

                if i % 1000 == 0:
                    logger.debug(f"Masked {i} rows")

            except Exception as e:
                logger.error(f"Error masking row {i}: {str(e)}")
                masked_dataset.append(row)  # Keep original row on error

        logger.info(f"Completed masking dataset with {len(masked_dataset)} rows")
        return masked_dataset

    def _get_provider_for_rule(self, rule: MaskingRule) -> Optional[BaseMaskingProvider]:
        """Get the appropriate provider for a masking rule"""
        for provider in self.providers:
            if provider.can_handle(rule):
                return provider
        return None

    def _evaluate_condition(self, condition: str, row_context: Dict[str, Any]) -> bool:
        """Evaluate a simple condition (basic implementation)"""
        # This is a basic implementation - you might want to use a more sophisticated
        # expression evaluator like `eval` with safety measures or a dedicated library
        try:
            # Simple conditions like "age > 18" or "status == 'active'"
            # This is a simplified version - enhance as needed
            for field, value in row_context.items():
                condition = condition.replace(field, f"'{value}'")

            # Basic safety check - only allow simple comparison operators
            allowed_ops = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not in']
            if not any(op in condition for op in allowed_ops):
                return True

            return eval(condition)
        except Exception as e:
            logger.warning(f"Error evaluating condition '{condition}': {str(e)}")
            return True  # Default to masking if condition fails

    def get_masking_stats(self) -> Dict[str, Any]:
        """Get statistics about masking operations"""
        return {
            'total_rules': len(self.rules),
            'rules_by_strategy': {
                strategy.value: len([r for r in self.rules.values() if r.strategy == strategy])
                for strategy in MaskingStrategy
            },
            'rules_by_sensitivity': {
                level.value: len([r for r in self.rules.values() if r.sensitivity_level == level])
                for level in SensitivityLevel
            },
            'cache_size': len(self.field_cache)
        }

    def clear_cache(self):
        """Clear the deterministic masking cache"""
        self.field_cache.clear()
        logger.info("Cleared masking cache")


# Utility functions for common masking scenarios
def create_pii_masking_rules() -> List[MaskingRule]:
    """Create common PII masking rules"""
    return [
        MaskingRule(
            field_name="email",
            sensitivity_level=SensitivityLevel.PII,
            strategy=MaskingStrategy.FAKE_REPLACE,
            strategy_params={"fake_type": "email"}
        ),
        MaskingRule(
            field_name="phone",
            sensitivity_level=SensitivityLevel.PII,
            strategy=MaskingStrategy.PARTIAL_MASK,
            strategy_params={"preserve_start": 3, "preserve_end": 2}
        ),
        MaskingRule(
            field_name="ssn",
            sensitivity_level=SensitivityLevel.PII,
            strategy=MaskingStrategy.PARTIAL_MASK,
            strategy_params={"preserve_start": 0, "preserve_end": 4}
        ),
        MaskingRule(
            field_name="credit_card",
            sensitivity_level=SensitivityLevel.FINANCIAL,
            strategy=MaskingStrategy.PARTIAL_MASK,
            strategy_params={"preserve_start": 0, "preserve_end": 4}
        ),
    ]


def create_financial_masking_rules() -> List[MaskingRule]:
    """Create common financial data masking rules"""
    return [
        MaskingRule(
            field_name="income",
            sensitivity_level=SensitivityLevel.FINANCIAL,
            strategy=MaskingStrategy.RANGE_SHIFT,
            strategy_params={"shift_range": 0.15}  # 15% shift
        ),
        MaskingRule(
            field_name="account_balance",
            sensitivity_level=SensitivityLevel.FINANCIAL,
            strategy=MaskingStrategy.RANGE_SHIFT,
            strategy_params={"shift_range": 0.2}  # 20% shift
        ),
    ]