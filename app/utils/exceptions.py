# app/utils/exceptions.py
"""
Custom Exceptions for Synthetic Data Platform
"""


class SyntheticDataError(Exception):
    """Base exception for synthetic data platform"""
    pass


class ConfigurationError(SyntheticDataError):
    """Configuration validation or loading errors"""
    pass


class GenerationError(SyntheticDataError):
    """Data generation errors"""
    pass


class ValidationError(SyntheticDataError):
    """Data validation errors"""
    pass


class RelationshipError(SyntheticDataError):
    """Foreign key relationship errors"""
    pass


class MaskingError(SyntheticDataError):
    """Data masking errors"""
    pass


class SchemaParsingError(SyntheticDataError):
    """Schema parsing errors"""
    pass
