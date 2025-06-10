"""
Enhanced Configuration Manager with Schema Validation
Supports YAML/JSON configs with Pydantic validation
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, ValidationError, validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "str"
    BOOLEAN = "bool"
    DATE = "date"
    DATETIME = "datetime"
    DECIMAL = "decimal"


class SensitivityLevel(str, Enum):
    NONE = "NONE"
    PII = "PII"
    FINANCIAL = "FINANCIAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    PUBLIC = "PUBLIC"


class ConstraintType(str, Enum):
    PRIMARY_KEY = "PK"
    FOREIGN_KEY = "FK"
    UNIQUE = "UNIQUE"
    NOT_NULL = "NOT_NULL"
    CHECK = "CHECK"


class RuleConfig(BaseModel):
    type: str
    regex: Optional[str] = None
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    values: Optional[List[Any]] = None
    format: Optional[str] = None
    locale: Optional[str] = None

    @validator('type')
    def validate_rule_type(cls, v):
        valid_types = ['email', 'phone_number', 'address', 'first_name', 'last_name',
                       'range', 'choice', 'date_range', 'custom']
        if v not in valid_types:
            raise ValueError(f'Rule type must be one of {valid_types}')
        return v


class ColumnConfig(BaseModel):
    name: str
    type: DataType
    constraint: Optional[List[ConstraintType]] = []
    rule: Union[str, RuleConfig, None] = None
    sensitivity: Optional[SensitivityLevel] = SensitivityLevel.NONE
    nullable: bool = True
    unique: bool = False
    default: Optional[Any] = None

    @validator('constraint')
    def validate_constraints(cls, v):
        if v is None:
            return []
        return [ConstraintType(c) if isinstance(c, str) else c for c in v]


class ForeignKeyConfig(BaseModel):
    parent_table: str
    parent_column: str
    child_column: str
    cascade_delete: bool = False


class TableConfig(BaseModel):
    table_name: str
    columns: List[ColumnConfig]
    foreign_keys: Optional[List[ForeignKeyConfig]] = []
    rows: Optional[int] = None

    @validator('columns')
    def validate_columns(cls, v):
        if not v:
            raise ValueError("Table must have at least one column")

        # Check for duplicate column names
        names = [col.name for col in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate column names found")

        # Ensure at least one primary key exists
        pk_columns = [col for col in v if ConstraintType.PRIMARY_KEY in (col.constraint or [])]
        if not pk_columns:
            logger.warning(f"No primary key defined for table")

        return v


class ExportConfig(BaseModel):
    format: str = "csv"
    destination: str = "local"
    path: str = "./output"
    compression: Optional[str] = None
    batch_size: int = 10000

    @validator('format')
    def validate_format(cls, v):
        valid_formats = ['csv', 'json', 'parquet', 'sql', 'excel']
        if v not in valid_formats:
            raise ValueError(f'Export format must be one of {valid_formats}')
        return v


class DatabaseConfig(BaseModel):
    host: str
    port: int = 5432
    database: str
    username: str
    password: str
    schema: Optional[str] = "public"


class GenerationConfig(BaseModel):
    tables: List[TableConfig]
    locale: str = "en_US"
    rows: int = 1000
    seed: Optional[int] = None
    export: Optional[ExportConfig] = ExportConfig()
    database: Optional[DatabaseConfig] = None

    @validator('tables')
    def validate_tables(cls, v):
        if not v:
            raise ValueError("At least one table must be defined")

        # Check for duplicate table names
        names = [table.table_name for table in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate table names found")

        # Validate foreign key relationships
        cls._validate_foreign_keys(v)
        return v

    @staticmethod
    def _validate_foreign_keys(tables: List[TableConfig]):
        """Validate foreign key relationships across tables"""
        table_columns = {}
        for table in tables:
            table_columns[table.table_name] = [col.name for col in table.columns]

        for table in tables:
            for fk in table.foreign_keys or []:
                # Check parent table exists
                if fk.parent_table not in table_columns:
                    raise ValueError(f"Parent table '{fk.parent_table}' not found")

                # Check parent column exists
                if fk.parent_column not in table_columns[fk.parent_table]:
                    raise ValueError(f"Parent column '{fk.parent_column}' not found in table '{fk.parent_table}'")

                # Check child column exists
                if fk.child_column not in table_columns[table.table_name]:
                    raise ValueError(f"Child column '{fk.child_column}' not found in table '{table.table_name}'")


class ConfigManager:
    """Enhanced configuration manager with validation and schema support"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[GenerationConfig] = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def load_config(self, config_path: Union[str, Path]) -> GenerationConfig:
        """Load and validate configuration from file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

            self.config = GenerationConfig(**config_data)
            self.config_path = config_path
            logger.info(f"Configuration loaded successfully from {config_path}")
            return self.config

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def save_config(self, config: GenerationConfig, output_path: Union[str, Path]):
        """Save configuration to file"""
        output_path = Path(output_path)

        try:
            config_dict = config.dict()

            if output_path.suffix.lower() in ['.yaml', '.yml']:
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")

            logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def validate_config(self, config_data: Dict) -> bool:
        """Validate configuration data against schema"""
        try:
            GenerationConfig(**config_data)
            return True
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_table_config(self, table_name: str) -> Optional[TableConfig]:
        """Get configuration for a specific table"""
        if not self.config:
            raise ValueError("No configuration loaded")

        for table in self.config.tables:
            if table.table_name == table_name:
                return table
        return None

    def get_sensitive_columns(self, table_name: str) -> List[str]:
        """Get list of sensitive columns for a table"""
        table_config = self.get_table_config(table_name)
        if not table_config:
            return []

        return [
            col.name for col in table_config.columns
            if col.sensitivity and col.sensitivity != SensitivityLevel.NONE
        ]

    def get_foreign_key_dependencies(self) -> Dict[str, List[str]]:
        """Get foreign key dependency graph"""
        if not self.config:
            raise ValueError("No configuration loaded")

        dependencies = {}
        for table in self.config.tables:
            deps = []
            for fk in table.foreign_keys or []:
                deps.append(fk.parent_table)
            dependencies[table.table_name] = deps

        return dependencies

    def get_generation_order(self) -> List[str]:
        """Get table generation order respecting foreign key dependencies"""
        dependencies = self.get_foreign_key_dependencies()

        # Topological sort to determine generation order
        visited = set()
        temp_visited = set()
        result = []

        def visit(table: str):
            if table in temp_visited:
                raise ValueError(f"Circular dependency detected involving table: {table}")
            if table in visited:
                return

            temp_visited.add(table)
            for dependency in dependencies.get(table, []):
                visit(dependency)
            temp_visited.remove(table)
            visited.add(table)
            result.append(table)

        for table in dependencies:
            if table not in visited:
                visit(table)

        return result


# Example usage and factory functions
def create_config_from_sample() -> GenerationConfig:
    """Create a sample configuration"""
    return GenerationConfig(
        tables=[
            TableConfig(
                table_name="address",
                columns=[
                    ColumnConfig(
                        name="address_id",
                        type=DataType.INT,
                        constraint=[ConstraintType.PRIMARY_KEY]
                    ),
                    ColumnConfig(
                        name="address",
                        type=DataType.STRING,
                        rule="address"
                    )
                ]
            ),
            TableConfig(
                table_name="customer",
                columns=[
                    ColumnConfig(
                        name="customer_id",
                        type=DataType.INT,
                        constraint=[ConstraintType.PRIMARY_KEY]
                    ),
                    ColumnConfig(
                        name="first_name",
                        type=DataType.STRING,
                        rule="first_name"
                    ),
                    ColumnConfig(
                        name="last_name",
                        type=DataType.STRING,
                        rule="last_name"
                    ),
                    ColumnConfig(
                        name="email",
                        type=DataType.STRING,
                        rule=RuleConfig(
                            type="email",
                            regex=r"^[\w.-]+@[\w.-]+\.\w{2,4}$"
                        ),
                        sensitivity=SensitivityLevel.PII
                    ),
                    ColumnConfig(
                        name="address_id",
                        type=DataType.INT,
                        constraint=[ConstraintType.FOREIGN_KEY]
                    )
                ],
                foreign_keys=[
                    ForeignKeyConfig(
                        parent_table="address",
                        parent_column="address_id",
                        child_column="address_id"
                    )
                ]
            )
        ],
        locale="en_GB",
        rows=20000
    )


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()

    # Create sample config
    sample_config = create_config_from_sample()

    # Save sample config
    config_manager.save_config(sample_config, "sample_config.yaml")

    # Load and validate config
    loaded_config = config_manager.load_config("sample_config.yaml")

    # Get generation order
    order = config_manager.get_generation_order()
    print(f"Generation order: {order}")

    # Get sensitive columns
    sensitive = config_manager.get_sensitive_columns("customer")
    print(f"Sensitive columns in customer: {sensitive}")