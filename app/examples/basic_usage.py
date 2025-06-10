# examples/basic_usage.py
"""
Basic usage examples for Synthetic Data Platform
"""
import json
from pathlib import Path
from app.core.data_generator import EnhancedDataGenerator
from app.input.sample_data_profiler import SampleDataProfiler
from app.main import EnhancedDataGenerationOrchestrator


def basic_generation_example():
    """Basic data generation example"""
    print("=== Basic Data Generation Example ===")

    # Define simple configuration
    config = {
        "tables": [
            {
                "table_name": "customers",
                "columns": [
                    {
                        "name": "customer_id",
                        "type": "int",
                        "constraints": ["PK"],
                        "rule": {"type": "range", "min": 1, "max": 10000}
                    },
                    {
                        "name": "first_name",
                        "type": "str",
                        "rule": "first_name"
                    },
                    {
                        "name": "last_name",
                        "type": "str",
                        "rule": "last_name"
                    },
                    {
                        "name": "email",
                        "type": "str",
                        "rule": {"type": "email"}
                    },
                    {
                        "name": "age",
                        "type": "int",
                        "rule": {"type": "range", "min": 18, "max": 80}
                    },
                    {
                        "name": "registration_date",
                        "type": "date",
                        "rule": {
                            "type": "date_range",
                            "start": "2020-01-01",
                            "end": "2024-12-31"
                        }
                    }
                ]
            }
        ]
    }

    # Initialize generator
    generator = EnhancedDataGenerator(config=config, locale="en_US")

    # Generate batch of data
    table_metadata = config["tables"][0]
    data = generator.generate_batch_enhanced(
        table_metadata=table_metadata,
        batch_size=100
    )

    print(f"Generated {len(data)} records")
    print("Sample record:")
    print(json.dumps(data[0], indent=2, default=str))


def profiling_example():
    """Data profiling example"""
    print("\n=== Data Profiling Example ===")

    import pandas as pd

    # Create sample data
    sample_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'] * 200,
        'last_name': ['Smith', 'Doe', 'Johnson', 'Brown', 'Wilson'] * 200,
        'email': [f'customer{i}@example.com' for i in range(1, 1001)],
        'age': [20 + (i % 60) for i in range(1000)],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'] * 200,
        'income': [30000 + (i * 100) for i in range(1000)]
    })

    # Initialize profiler
    profiler = SampleDataProfiler()

    # Profile the dataset
    profile = profiler.profile_dataset(sample_data, "customers")

    print(f"Profiled table: {profile['table_name']}")
    print(f"Rows: {profile['row_count']}, Columns: {profile['column_count']}")
    print(f"Data quality score: {100 - profile['data_quality']['missing_data_percentage']:.1f}%")

    # Generate configuration from profiling
    config = profiler.generate_test_data_config("customers")

    print("\nGenerated configuration:")
    print(json.dumps(config, indent=2, default=str))


def orchestrator_example():
    """Complete orchestrator example"""
    print("\n=== Complete Orchestrator Example ===")

    # Define configuration with relationships
    config = {
        "tables": [
            {
                "table_name": "departments",
                "columns": [
                    {
                        "name": "dept_id",
                        "type": "int",
                        "constraints": ["PK"],
                        "rule": {"type": "range", "min": 1, "max": 100}
                    },
                    {
                        "name": "dept_name",
                        "type": "str",
                        "rule": {
                            "type": "choice",
                            "value": ["Engineering", "Marketing", "Sales", "HR", "Finance"]
                        }
                    },
                    {
                        "name": "location",
                        "type": "str",
                        "rule": "city"
                    }
                ]
            },
            {
                "table_name": "employees",
                "columns": [
                    {
                        "name": "emp_id",
                        "type": "int",
                        "constraints": ["PK"],
                        "rule": {"type": "range", "min": 1000, "max": 9999}
                    },
                    {
                        "name": "first_name",
                        "type": "str",
                        "rule": "first_name"
                    },
                    {
                        "name": "last_name",
                        "type": "str",
                        "rule": "last_name"
                    },
                    {
                        "name": "email",
                        "type": "str",
                        "rule": {"type": "email"}
                    },
                    {
                        "name": "dept_id",
                        "type": "int",
                        "constraints": ["FK"]
                    },
                    {
                        "name": "salary",
                        "type": "float",
                        "rule": {"type": "range", "min": 30000, "max": 150000}
                    },
                    {
                        "name": "hire_date",
                        "type": "date",
                        "rule": {
                            "type": "date_range",
                            "start": "2020-01-01",
                            "end": "2024-12-31"
                        }
                    }
                ],
                "foreign_keys": [
                    {
                        "parent_table": "departments",
                        "parent_column": "dept_id",
                        "child_column": "dept_id"
                    }
                ]
            }
        ]
    }

    # Create output directory
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)

    # Initialize orchestrator
    orchestrator = EnhancedDataGenerationOrchestrator(
        config=config,
        output_dir=str(output_dir),
        locale="en_US"
    )

    # Setup generator
    orchestrator.setup_generator(use_profiling=False)

    # Generate data
    generated_data, stats = orchestrator.generate_all_tables(
        total_records_per_table=50,
        output_format="csv",
        batch_size=25
    )

    print(f"Generated data for {len(generated_data)} tables")
    print("Generation statistics:")
    for table_name, table_stats in stats.items():
        if isinstance(table_stats, dict) and "status" in table_stats:
            print(f"  {table_name}: {table_stats['status']} - {table_stats.get('records_generated', 0)} records")

    print(f"Output saved to: {output_dir}")


def conditional_rules_example():
    """Example with conditional rules"""
    print("\n=== Conditional Rules Example ===")

    config = {
        "tables": [
            {
                "table_name": "customers",
                "columns": [
                    {
                        "name": "customer_id",
                        "type": "int",
                        "constraints": ["PK"],
                        "rule": {"type": "range", "min": 1, "max": 1000}
                    },
                    {
                        "name": "age",
                        "type": "int",
                        "rule": {"type": "range", "min": 18, "max": 80}
                    },
                    {
                        "name": "income",
                        "type": "float",
                        "conditional_rules": [
                            {
                                "condition": {
                                    "column": "age",
                                    "operator": "<",
                                    "value": 25
                                },
                                "rule": {"type": "range", "min": 20000, "max": 40000},
                                "type": "float"
                            },
                            {
                                "condition": {
                                    "column": "age",
                                    "operator": ">=",
                                    "value": 25
                                },
                                "rule": {"type": "range", "min": 40000, "max": 120000},
                                "type": "float"
                            }
                        ]
                    },
                    {
                        "name": "credit_score",
                        "type": "int",
                        "conditional_rules": [
                            {
                                "condition": {
                                    "column": "income",
                                    "operator": ">",
                                    "value": 80000
                                },
                                "rule": {"type": "range", "min": 700, "max": 850},
                                "type": "int"
                            },
                            {
                                "condition": {
                                    "column": "income",
                                    "operator": "<=",
                                    "value": 80000
                                },
                                "rule": {"type": "range", "min": 600, "max": 750},
                                "type": "int"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    generator = EnhancedDataGenerator(config=config, locale="en_US")

    table_metadata = config["tables"][0]
    data = generator.generate_batch_enhanced(
        table_metadata=table_metadata,
        batch_size=10
    )

    print("Sample records with conditional rules:")
    for i, record in enumerate(data[:3]):
        print(
            f"Record {i + 1}: Age={record['age']}, Income=${record['income']:.0f}, Credit Score={record['credit_score']}")


def main():
    """Run all examples"""
    basic_generation_example()
    profiling_example()
    orchestrator_example()
    conditional_rules_example()

    print("\n=== All Examples Completed ===")
    print("Check the './example_output' directory for generated files.")


if __name__ == "__main__":
    main()