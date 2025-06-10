import os
import logging
import argparse
from datetime import datetime
import json

import pandas as pd
from traceback import print_exc

from json_reader import JSONConfigReader
from core.data_generator import EnhancedDataGenerator, ProfilerIntegratedGenerator
from validator import DataValidator
from core.relationship_preserver import RelationshipPreserver
from core.rule_engine import RuleEngine
from input.sample_data_profiler import SampleDataProfiler
from input.schema_parser import SchemaParser
from writer import CSVWriter, JsonWriter, ParquetWriter, SQLQueryWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedDataGenerationOrchestrator:
    """Orchestrates the enhanced data generation process with all new features"""

    def __init__(self, config, output_dir="./output", locale="en_GB"):
        self.config = config
        self.output_dir = output_dir
        self.locale = locale
        self.logger = logger

        # Initialize enhanced components
        self.relationship_preserver = RelationshipPreserver(logger=self.logger)
        self.rule_engine = RuleEngine(logger=self.logger)
        self.sample_profiler = SampleDataProfiler(logger=self.logger)
        self.data_validator = DataValidator(logger=self.logger)

        # Initialize generator (will be set based on whether profiling is used)
        self.generator = None
        self.profiling_results = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def setup_generator(self, use_profiling=False, sample_data_path=None):
        """Setup the appropriate data generator based on requirements"""
        if use_profiling and sample_data_path:
            self.logger.info("Setting up ProfilerIntegratedGenerator with sample data profiling")
            self.generator = ProfilerIntegratedGenerator(
                config=self.config,
                locale=self.locale,
                logger=self.logger
            )
            self.generator.set_profiler(self.sample_profiler)

            # Load and profile sample data
            self._profile_sample_data(sample_data_path)
        else:
            self.logger.info("Setting up EnhancedDataGenerator")
            self.generator = EnhancedDataGenerator(
                config=self.config,
                locale=self.locale,
                logger=self.logger
            )

    def _profile_sample_data(self, sample_data_path):
        """Profile sample data to improve generation quality"""
        try:
            self.logger.info(f"Profiling sample data from: {sample_data_path}")

            # Load sample data
            sample_df = self.sample_profiler.load_data(sample_data_path)

            # Profile the dataset
            self.profiling_results = self.sample_profiler.profile_dataset(
                df=sample_df,
                table_name="sample_data",
                sample_size=10000  # Limit for performance
            )

            # Integrate profiling results into generator
            if hasattr(self.generator, 'integrate_profiling_results'):
                self.generator.integrate_profiling_results(self.profiling_results)

            self.logger.info("Sample data profiling completed successfully")

        except Exception as e:
            self.logger.error(f"Error profiling sample data: {e}")
            self.profiling_results = {}

    def load_templates(self, template_file_path=None):
        """Load rule templates from file or create from schema"""
        try:
            if template_file_path and os.path.exists(template_file_path):
                self.logger.info(f"Loading templates from file: {template_file_path}")
                self.rule_engine.load_templates_from_file(template_file_path)
            else:
                self.logger.info("Creating templates from configuration schema")
                # Create templates from config schema
                for table_config in self.config.get("tables", []):
                    template = self.rule_engine.create_template_from_schema(table_config)
                    self.rule_engine.add_template(template)

            self.logger.info(f"Loaded {len(self.rule_engine.list_templates())} templates")

        except Exception as e:
            self.logger.error(f"Error loading templates: {e}")

    def parse_ddl_schema(self, ddl_file_path):
        """Parse DDL file to extract schema information"""
        try:
            self.logger.info(f"Parsing DDL schema from: {ddl_file_path}")
            schema_parser = SchemaParser(logger=self.logger)

            schema_metadata = schema_parser.parse_ddl_file(ddl_file_path)

            # Update config with parsed schema
            if "tables" not in self.config:
                self.config["tables"] = []

            # Convert schema metadata to config format
            for table_name, table_info in schema_metadata.get("tables", {}).items():
                table_config = {
                    "table_name": table_name,
                    "columns": table_info.get("columns", []),
                    "foreign_keys": table_info.get("foreign_keys", []),
                    "primary_keys": table_info.get("primary_keys", [])
                }
                self.config["tables"].append(table_config)

            self.logger.info(f"Parsed {len(schema_metadata.get('tables', {}))} tables from DDL")
            return schema_metadata

        except Exception as e:
            self.logger.error(f"Error parsing DDL schema: {e}")
            return {}

    def prepare_generation_order(self):
        """Determine the correct order for table generation based on dependencies"""
        try:
            # Extract table metadata
            tables_metadata = self.config.get("tables", [])

            # Build dependency graph
            self.relationship_preserver.build_dependency_graph(tables_metadata)

            # Get generation order
            table_names = [table["table_name"] for table in tables_metadata]
            generation_order = self.relationship_preserver.get_generation_order(table_names)

            self.logger.info(f"Table generation order: {' -> '.join(generation_order)}")
            return generation_order

        except Exception as e:
            self.logger.error(f"Error determining generation order: {e}")
            # Fallback to original order
            return [table["table_name"] for table in self.config.get("tables", [])]

    def generate_table_data(self, table_metadata, total_records, batch_size=10000):
        """Generate data for a single table with enhanced features"""
        table_name = table_metadata["table_name"]
        self.logger.info(f"Generating {total_records} records for table: {table_name}")

        all_generated_records = []
        records_generated = 0

        try:
            # Get foreign key values for this table
            fk_values = {}
            for fk in table_metadata.get("foreign_keys", []):
                parent_table = fk["parent_table"]
                parent_column = fk["parent_column"]

                available_values = self.relationship_preserver.get_fk_values(
                    parent_table=parent_table,
                    parent_column=parent_column,
                    expected_data_type=fk.get("data_type"),
                    sample_size=min(total_records * 2, 10000)  # Get enough values
                )

                if available_values:
                    fk_values[f"{parent_table}.{parent_column}"] = available_values
                    self.logger.info(f"Found {len(available_values)} FK values for {parent_table}.{parent_column}")

            # Generate data in batches
            while records_generated < total_records:
                current_batch_size = min(batch_size, total_records - records_generated)

                # Generate batch using enhanced generator
                if hasattr(self.generator, 'generate_batch_enhanced'):
                    batch_data = self.generator.generate_batch_enhanced(
                        table_metadata=table_metadata,
                        batch_size=current_batch_size,
                        foreign_key_data=fk_values,
                        profiling_results=self.profiling_results.get(table_name)
                    )
                else:
                    # Fallback to basic batch generation
                    batch_data = self.generator.generate_batch(
                        table_metadata,
                        current_batch_size,
                        fk_values
                    )

                # Validate generated records
                valid_records = []
                for record in batch_data:
                    is_valid, validation_message = self.data_validator.validate_constraints(
                        record, table_metadata
                    )
                    if is_valid:
                        valid_records.append(record)
                    else:
                        self.logger.warning(f"Invalid record: {validation_message}")

                if valid_records:
                    all_generated_records.extend(valid_records)
                    records_generated += len(valid_records)

                    # Refresh FK pools with new data
                    generated_data_map = {table_name: pd.DataFrame(valid_records)}
                    self.relationship_preserver.refresh_fk_pools(
                        generated_data=generated_data_map,
                        tables_metadata=[table_metadata],
                        table_name=table_name
                    )

                # Progress logging
                if records_generated % 50000 == 0:
                    self.logger.info(f"Generated {records_generated}/{total_records} records for {table_name}")

            self.logger.info(f"Completed generation for {table_name}: {len(all_generated_records)} records")
            return all_generated_records

        except Exception as e:
            self.logger.error(f"Error generating data for {table_name}: {e}")
            raise

    def export_table_data(self, table_data, table_metadata, output_format):
        """Export generated table data to specified format"""
        if not table_data:
            self.logger.warning(f"No data to export for table {table_metadata['table_name']}")
            return

        try:
            # Convert to DataFrame
            df = pd.DataFrame(table_data)

            # Apply data type conversions
            self._apply_data_type_conversions(df, table_metadata)

            # Choose appropriate writer
            writer = None
            if output_format == "csv":
                writer = CSVWriter(df, table_metadata, self.output_dir, len(table_data), self.logger)
            elif output_format == "parquet":
                writer = ParquetWriter(df, table_metadata, self.output_dir, len(table_data), self.logger)
            elif output_format == "json":
                writer = JsonWriter(df, table_metadata, self.output_dir, len(table_data), self.logger)
            elif output_format == "sql_query":
                writer = SQLQueryWriter(df, table_metadata, self.output_dir, len(table_data), self.logger)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            writer.write_data()
            self.logger.info(f"Exported {len(table_data)} records for {table_metadata['table_name']}")

        except Exception as e:
            self.logger.error(f"Error exporting data for {table_metadata['table_name']}: {e}")
            raise

    def _apply_data_type_conversions(self, df, table_metadata):
        """Apply proper data type conversions to DataFrame"""
        for column in table_metadata.get("columns", []):
            column_name = column["name"]
            column_type = column["type"]

            if column_name not in df.columns:
                continue

            try:
                if column_type in ["boolean", "bool"]:
                    df[column_name] = df[column_name].astype(bool)
                elif column_type in ["int", "integer"]:
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0).astype(int)
                elif column_type in ["float", "double", "decimal"]:
                    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                elif column_type in ["date"]:
                    df[column_name] = pd.to_datetime(df[column_name], errors='coerce').dt.date
                elif column_type in ["datetime", "timestamp"]:
                    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
            except Exception as e:
                self.logger.warning(f"Error converting column {column_name} to {column_type}: {e}")

    def validate_referential_integrity(self, generated_data_map):
        """Validate referential integrity across all generated tables"""
        try:
            self.logger.info("Validating referential integrity...")

            tables_metadata = self.config.get("tables", [])
            validation_report = self.relationship_preserver.validate_relationships(
                df_map=generated_data_map,
                tables_metadata=tables_metadata
            )

            # Log validation results
            for table_name, results in validation_report.items():
                if isinstance(results, dict):
                    if results.get("valid", True):
                        self.logger.info(f"✓ {table_name}: Referential integrity validated")
                    else:
                        issues = results.get("issues", [])
                        self.logger.warning(f"✗ {table_name}: {len(issues)} integrity issues found")
                        for issue in issues[:5]:  # Show first 5 issues
                            self.logger.warning(f"  - {issue}")

            return validation_report

        except Exception as e:
            self.logger.error(f"Error validating referential integrity: {e}")
            return {}

    def generate_all_tables(self, total_records_per_table, output_format="csv", batch_size=10000):
        """Generate data for all tables in the correct order"""
        self.logger.info("Starting enhanced data generation process")

        # Get generation order
        generation_order = self.prepare_generation_order()

        # Track all generated data for validation
        all_generated_data = {}
        generation_stats = {}

        try:
            for i, table_name in enumerate(generation_order, 1):
                self.logger.info(f"Processing table {i}/{len(generation_order)}: {table_name}")

                # Find table metadata
                table_metadata = None
                for table in self.config.get("tables", []):
                    if table["table_name"] == table_name:
                        table_metadata = table
                        break

                if not table_metadata:
                    self.logger.error(f"Table metadata not found for {table_name}")
                    continue

                start_time = pd.Timestamp.now()

                try:
                    # Generate data for this table
                    table_data = self.generate_table_data(
                        table_metadata=table_metadata,
                        total_records=total_records_per_table,
                        batch_size=batch_size
                    )

                    # Store for validation
                    all_generated_data[table_name] = pd.DataFrame(table_data)

                    # Export data
                    self.export_table_data(table_data, table_metadata, output_format)

                    # Record statistics
                    end_time = pd.Timestamp.now()
                    duration = (end_time - start_time).total_seconds()

                    generation_stats[table_name] = {
                        "records_generated": len(table_data),
                        "duration_seconds": duration,
                        "records_per_second": len(table_data) / duration if duration > 0 else 0,
                        "status": "success"
                    }

                    self.logger.info(f"✓ Completed {table_name} in {duration:.2f}s "
                                     f"({generation_stats[table_name]['records_per_second']:.0f} records/sec)")

                except Exception as e:
                    self.logger.error(f"✗ Failed to process {table_name}: {e}")
                    generation_stats[table_name] = {
                        "status": "failed",
                        "error": str(e)
                    }
                    continue

            # Validate referential integrity
            if all_generated_data:
                validation_report = self.validate_referential_integrity(all_generated_data)
                generation_stats["validation_report"] = validation_report

            # Generate comprehensive report
            self.generate_enhanced_report(generation_stats, output_format)

            return all_generated_data, generation_stats

        except Exception as e:
            self.logger.error(f"Error in data generation process: {e}")
            raise

    def generate_enhanced_report(self, generation_stats, output_format):
        """Generate comprehensive generation report with enhanced metrics"""
        try:
            # Calculate summary statistics
            successful_tables = [name for name, stats in generation_stats.items()
                                 if isinstance(stats, dict) and stats.get("status") == "success"]
            failed_tables = [name for name, stats in generation_stats.items()
                             if isinstance(stats, dict) and stats.get("status") == "failed"]

            total_records = sum(stats.get("records_generated", 0)
                                for stats in generation_stats.values()
                                if isinstance(stats, dict) and "records_generated" in stats)

            total_duration = sum(stats.get("duration_seconds", 0)
                                 for stats in generation_stats.values()
                                 if isinstance(stats, dict) and "duration_seconds" in stats)

            # Get enhanced statistics from generator
            generator_stats = {}
            if hasattr(self.generator, 'get_enhanced_statistics'):
                generator_stats = self.generator.get_enhanced_statistics()

            # Get relationship preserver status
            fk_pools_status = self.relationship_preserver.get_fk_pools_status()

            # Get rule engine statistics
            rule_engine_stats = self.rule_engine.get_statistics()

            # Compile comprehensive report
            report = {
                "generation_timestamp": pd.Timestamp.now().isoformat(),
                "generator_type": "EnhancedDataGenerator",
                "configuration": {
                    "total_tables": len(self.config.get("tables", [])),
                    "records_per_table": self.config.get("row_count", self.config.get("total_records", 100)),
                    "output_format": output_format,
                    "output_directory": self.output_dir,
                    "locale": self.locale,
                    "profiling_enabled": bool(self.profiling_results)
                },
                "summary_statistics": {
                    "successful_tables": len(successful_tables),
                    "failed_tables": len(failed_tables),
                    "total_records_generated": total_records,
                    "total_duration_seconds": total_duration,
                    "average_records_per_second": total_records / total_duration if total_duration > 0 else 0
                },
                "enhanced_features": {
                    "generator_statistics": generator_stats,
                    "fk_pools_status": fk_pools_status,
                    "rule_engine_statistics": rule_engine_stats,
                    "profiling_results_available": bool(self.profiling_results)
                },
                "table_details": generation_stats
            }

            # Save report
            report_path = os.path.join(self.output_dir, "enhanced_generation_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"Enhanced generation report saved to: {report_path}")

            # Log summary
            self.logger.info("=== ENHANCED GENERATION SUMMARY ===")
            self.logger.info(f"Generator: {report['generator_type']}")
            self.logger.info(f"Successful tables: {len(successful_tables)}/{len(self.config.get('tables', []))}")
            self.logger.info(f"Total records: {total_records:,}")
            self.logger.info(f"Total duration: {total_duration:.2f}s")
            self.logger.info(
                f"Average speed: {report['summary_statistics']['average_records_per_second']:.0f} records/sec")

            if self.profiling_results:
                self.logger.info("✓ Sample data profiling was used")

            if fk_pools_status:
                self.logger.info(f"✓ FK pools maintained: {len(fk_pools_status)}")

            if failed_tables:
                self.logger.warning(f"✗ Failed tables: {failed_tables}")

        except Exception as e:
            self.logger.error(f"Error generating enhanced report: {e}")


def main(config_path,
         total_records_per_table,
         output_dir="./output",
         output_format="csv",
         batch_size=10000,
         template_file=None,
         sample_data_path=None,
         ddl_schema_path=None,
         locale="en_GB"):
    """Main function orchestrating the enhanced data generation process"""

    start_time = datetime.now()

    try:
        # Load configuration
        logger.info(f"Loading configuration from: {config_path}")
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Use JSONConfigReader for other formats
            config_reader = JSONConfigReader(config_path)
            config = config_reader.load_config()

        # Create orchestrator
        orchestrator = EnhancedDataGenerationOrchestrator(
            config=config,
            output_dir=output_dir,
            locale=locale
        )

        # Parse DDL schema if provided
        if ddl_schema_path and os.path.exists(ddl_schema_path):
            orchestrator.parse_ddl_schema(ddl_schema_path)

        # Load rule templates
        orchestrator.load_templates(template_file)

        # Setup generator with or without profiling
        use_profiling = sample_data_path and os.path.exists(sample_data_path)
        orchestrator.setup_generator(use_profiling=use_profiling, sample_data_path=sample_data_path)

        # Generate all table data
        logger.info("Starting enhanced data generation...")
        generated_data, stats = orchestrator.generate_all_tables(
            total_records_per_table=total_records_per_table,
            output_format=output_format,
            batch_size=batch_size
        )

        end_time = datetime.now()
        logger.info(f"Enhanced data generation completed in: {end_time - start_time}")

        return generated_data, stats

    except Exception as e:
        logger.error(f"Fatal error in main process: {e}")
        print_exc()
        raise


def parse_arguments():
    """Parse command line arguments with enhanced options"""
    parser = argparse.ArgumentParser(description='Enhanced Data Generator with Profiling and Advanced Features')

    # Required arguments
    parser.add_argument('--config', '-c', required=True,
                        help='Path to JSON/YAML config file')

    # Output configuration
    parser.add_argument('--output_dir', '-od', default="./output",
                        help='Directory to save generated files')
    parser.add_argument('--output_format', '-f',
                        choices=['csv', 'json', 'sql_query', 'parquet'],
                        default='csv', help='Output file format')

    # Generation parameters
    parser.add_argument('--rows', '-r', type=int, default=1000,
                        help='Number of rows to generate per table')
    parser.add_argument('--batch_size', '-bs', type=int, default=10000,
                        help='Batch size for processing')
    parser.add_argument('--locale', '-l', default="en_GB",
                        help='Locale for data generation')

    # Enhanced features
    parser.add_argument('--template_file', '-t',
                        help='Path to rule template file (YAML/JSON)')
    parser.add_argument('--sample_data', '-sd',
                        help='Path to sample data file for profiling')
    parser.add_argument('--ddl_schema', '-ddl',
                        help='Path to DDL file for schema parsing')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(args.output_dir, f"enhanced_generation_{timestamp}")

    try:
        # Run enhanced data generation
        generated_data, generation_stats = main(
            config_path=args.config,
            total_records_per_table=args.rows,
            output_dir=output_directory,
            output_format=args.output_format,
            batch_size=args.batch_size,
            template_file=args.template_file,
            sample_data_path=args.sample_data,
            ddl_schema_path=args.ddl_schema,
            locale=args.locale
        )

        logger.info("=== ENHANCED DATA GENERATION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Output location: {output_directory}")
        logger.info(f"Tables processed: {len(generated_data)}")

        successful_tables = [name for name, stats in generation_stats.items()
                             if isinstance(stats, dict) and stats.get("status") == "success"]
        logger.info(f"Successfully generated: {len(successful_tables)} tables")

    except Exception as e:
        logger.error(f"Enhanced data generation failed: {e}")
        print_exc()
        exit(1)