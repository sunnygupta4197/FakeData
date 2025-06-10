# tests/test_generator.py
"""
Tests for data generator
"""
import pytest
import pandas as pd
from app.core.data_generator import EnhancedDataGenerator
from app.input.sample_data_profiler import SampleDataProfiler


class TestEnhancedDataGenerator:

    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
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
                        }
                    ]
                }
            ]
        }

        self.generator = EnhancedDataGenerator(
            config=self.config,
            locale="en_US"
        )

    def test_generator_initialization(self):
        """Test generator initializes correctly"""
        assert self.generator is not None
        assert self.generator.config == self.config
        assert self.generator.faker is not None

    def test_basic_value_generation(self):
        """Test basic value generation"""
        # Test email generation
        email = self.generator.generate_value_advanced("email", "str")
        assert "@" in email
        assert "." in email

        # Test range generation
        age = self.generator.generate_value_advanced(
            {"type": "range", "min": 18, "max": 80},
            "int"
        )
        assert 18 <= age <= 80

        # Test name generation
        first_name = self.generator.generate_value_advanced("first_name", "str")
        assert isinstance(first_name, str)
        assert len(first_name) > 0

    def test_batch_generation(self):
        """Test batch data generation"""
        table_metadata = self.config["tables"][0]
        batch_size = 100

        batch_data = self.generator.generate_batch_enhanced(
            table_metadata=table_metadata,
            batch_size=batch_size
        )

        assert len(batch_data) == batch_size
        assert all(isinstance(row, dict) for row in batch_data)

        # Check all columns are present
        expected_columns = {col["name"] for col in table_metadata["columns"]}
        for row in batch_data:
            assert set(row.keys()) == expected_columns

    def test_unique_constraints(self):
        """Test unique constraint enforcement"""
        # Add unique constraint
        config_with_unique = self.config.copy()
        config_with_unique["tables"][0]["columns"][0]["constraints"] = ["PK", "unique"]

        generator = EnhancedDataGenerator(
            config=config_with_unique,
            locale="en_US"
        )

        table_metadata = config_with_unique["tables"][0]
        batch_data = generator.generate_batch_enhanced(
            table_metadata=table_metadata,
            batch_size=50
        )

        # Check uniqueness of customer_id
        customer_ids = [row["customer_id"] for row in batch_data]
        assert len(customer_ids) == len(set(customer_ids))

    def test_data_type_conversion(self):
        """Test data type conversion"""
        value = self.generator._convert_value_to_type("123", "int")
        assert value == 123
        assert isinstance(value, int)

        value = self.generator._convert_value_to_type("123.45", "float")
        assert value == 123.45
        assert isinstance(value, float)

        value = self.generator._convert_value_to_type("true", "bool")
        assert value is True

        value = self.generator._convert_value_to_type("2023-01-01", "date")
        assert value is not None

    def test_conditional_rules(self):
        """Test conditional rule application"""
        # Test with mock row data
        row_data = {"age": 25}

        column_def = {
            "name": "income",
            "type": "int",
            "conditional_rules": [
                {
                    "condition": {
                        "column": "age",
                        "operator": ">",
                        "value": 18
                    },
                    "rule": {"type": "range", "min": 30000, "max": 100000},
                    "type": "int"
                }
            ]
        }

        value = self.generator.apply_conditional_rules(row_data, column_def)

        if value is not None:
            assert 30000 <= value <= 100000

    def test_statistics_tracking(self):
        """Test statistics collection"""
        stats = self.generator.get_enhanced_statistics()

        assert isinstance(stats, dict)
        assert "libraries_available" in stats
        assert "unique_constraints_tracked" in stats
        assert "generated_tables" in stats


class TestSampleDataProfiler:

    def setup_method(self):
        """Setup test fixtures"""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'first_name': ['John', 'Jane', 'Bob', 'Alice'] * 25,
            'last_name': ['Smith', 'Doe', 'Johnson', 'Brown'] * 25,
            'email': [f'user{i}@example.com' for i in range(1, 101)],
            'age': [20, 25, 30, 35, 40] * 20,
            'income': [30000, 40000, 50000, 60000, 70000] * 20,
            'city': ['New York', 'London', 'Paris', 'Tokyo'] * 25
        })

        self.profiler = SampleDataProfiler()

    def test_profiler_initialization(self):
        """Test profiler initializes correctly"""
        assert self.profiler is not None
        assert hasattr(self.profiler, 'profile_results')

    def test_data_loading(self):
        """Test data loading from DataFrame"""
        df = self.profiler.load_data(self.sample_data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert len(df.columns) == 7

    def test_dataset_profiling(self):
        """Test dataset profiling"""
        profile = self.profiler.profile_dataset(
            df=self.sample_data,
            table_name="test_table"
        )

        assert isinstance(profile, dict)
        assert "table_name" in profile
        assert "columns" in profile
        assert "data_quality" in profile
        assert "relationships" in profile

        # Check column profiles
        assert len(profile["columns"]) == 7

        # Check specific column profile
        email_profile = profile["columns"]["email"]
        assert email_profile["inferred_type"] == "str"
        assert email_profile["unique_count"] == 100
        assert email_profile["null_count"] == 0

    def test_config_generation(self):
        """Test configuration generation from profile"""
        # First profile the data
        self.profiler.profile_dataset(
            df=self.sample_data,
            table_name="test_table"
        )

        # Generate config
        config = self.profiler.generate_test_data_config("test_table")

        assert isinstance(config, dict)
        assert "tables" in config
        assert len(config["tables"]) == 1

        table_config = config["tables"][0]
        assert table_config["table_name"] == "test_table"
        assert len(table_config["columns"]) == 7

        # Check email column rule
        email_column = next(
            col for col in table_config["columns"]
            if col["name"] == "email"
        )
        assert email_column["rule"]["type"] == "email"

    def test_data_quality_analysis(self):
        """Test data quality analysis"""
        # Add some nulls and duplicates
        data_with_issues = self.sample_data.copy()
        data_with_issues.loc[0:4, 'email'] = None
        data_with_issues = pd.concat([data_with_issues, data_with_issues.head(5)])

        profile = self.profiler.profile_dataset(
            df=data_with_issues,
            table_name="test_table_issues"
        )

        quality = profile["data_quality"]

        assert quality["missing_data_percentage"] > 0
        assert quality["duplicate_rows"] > 0
        assert "email" in quality["columns_with_nulls"]

    def test_pattern_detection(self):
        """Test string pattern detection"""
        email_series = self.sample_data["email"]
        patterns = self.profiler._detect_string_patterns(email_series)

        assert patterns["email"] > 90  # Should detect most as emails
        assert patterns["numeric"] == 0  # No numeric strings


class TestIntegration:
    """Integration tests"""

    def test_profiler_to_generator_integration(self):
        """Test integration between profiler and generator"""
        # Create sample data
        sample_data = pd.DataFrame({
            'id': range(1, 51),
            'name': ['Person ' + str(i) for i in range(1, 51)],
            'email': [f'person{i}@test.com' for i in range(1, 51)],
            'age': [20 + (i % 50) for i in range(50)]
        })

        # Profile the data
        profiler = SampleDataProfiler()
        profile = profiler.profile_dataset(sample_data, "test_table")

        # Generate config
        config = profiler.generate_test_data_config("test_table")

        # Use config with generator
        generator = EnhancedDataGenerator(config=config, locale="en_US")

        # Integrate profiling results
        generator.integrate_profiling_results({"test_table": profile})

        # Generate data
        table_metadata = config["tables"][0]
        generated_data = generator.generate_batch_enhanced(
            table_metadata=table_metadata,
            batch_size=25,
            profiling_results={"test_table": profile}
        )

        assert len(generated_data) == 25
        assert all("email" in row for row in generated_data)
        assert all("@" in row["email"] for row in generated_data)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        from app.main import EnhancedDataGenerationOrchestrator
        import tempfile
        import json

        # Create test config
        config = {
            "tables": [
                {
                    "table_name": "users",
                    "columns": [
                        {
                            "name": "user_id",
                            "type": "int",
                            "constraints": ["PK"],
                            "rule": {"type": "range", "min": 1, "max": 1000}
                        },
                        {
                            "name": "username",
                            "type": "str",
                            "rule": "user_name"
                        },
                        {
                            "name": "email",
                            "type": "str",
                            "rule": {"type": "email"}
                        }
                    ]
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create orchestrator
            orchestrator = EnhancedDataGenerationOrchestrator(
                config=config,
                output_dir=temp_dir,
                locale="en_US"
            )

            # Setup generator
            orchestrator.setup_generator(use_profiling=False)

            # Generate data
            generated_data, stats = orchestrator.generate_all_tables(
                total_records_per_table=10,
                output_format="csv",
                batch_size=5
            )

            assert "users" in generated_data
            assert len(generated_data["users"]) == 10
            assert "users" in stats
            assert stats["users"]["status"] == "success"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
