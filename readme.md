## 4. Integration with Existing Code

Your existing `main.py` and `data_generator.py` will be enhanced to work with the new components:

### Enhanced main.py integration points:
- Import schema_parser for DDL parsing
- Use rule_engine for template management
- Integrate masking_engine for data anonymization
- Use enhanced exporter for multi-destination output
- Add CLI interface through click

### OptimizedDataGenerator enhancements:
- Integration with relationship_preserver
- Masking engine callbacks
- Rule engine integration
- Enhanced export capabilities

## 5. Key Features Added:

### Input Layer:
- **Schema Parser**: Parse DDL files, database schemas
- **Data Profiler**: Analyze existing data to infer generation rules

### Core Engine:
- **Rule Engine**: Template-based rule management
- **Relationship Preserver**: Enhanced FK/PK handling
- **Masking Engine**: Data anonymization capabilities

### Output Layer:
- **Multi-format Export**: CSV, JSON, Parquet, SQL with cloud support
- **Audit Logging**: Comprehensive generation tracking

### API & Orchestration:
- **CLI Interface**: Command-line tool with rich options
- **REST API**: Web service for remote job management
- **Job Scheduler**: Automated generation scheduling

## 6. Usage Examples:

### CLI Usage:
```bash
# Generate from DDL file
python -m test_data_generator --schema schema.sql --rows 10000 --output ./data

# Generate from existing data profile
python -m test_data_generator --profile existing_data.csv --rules custom_rules.yaml

# Generate with masking
python -m test_data_generator --config config.yaml --mask-sensitive --export-to s3
```

### Programmatic Usage:
```python
from test_data_generator import TestDataGenerator

generator = TestDataGenerator(config_path="config.yaml")
generator.load_schema("schema.sql")
generator.set_rules("custom_rules.yaml")
generator.generate(rows=10000)
generator.export(format="parquet", destination="s3://my-bucket/test-data/")
```

This structure maintains your existing optimized code while adding the enterprise features from your architecture diagram.