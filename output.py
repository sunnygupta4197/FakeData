import json
import sqlite3
import pandas as pd
from typing import List, Dict, Any
import logging


class DataSaver:
    def __init__(self, logger=None, config=None, table_name=None):
        if logger is not None:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.config = config
        self.table_name = table_name if table_name else self.config.get("table_name", "generated_table")
        self.columns = self.config.get("columns", {})

    def _get_sql_type(self, column_type: str) -> str:
        """Convert column type to SQL type"""
        type_mapping = {
            'int': 'INT',
            'string': 'VARCHAR(255)',
            'text': 'TEXT',
            'float': 'DECIMAL(10,2)',
            'double': 'DOUBLE',
            'date': 'DATE',
            'datetime': 'DATETIME',
            'timestamp': 'TIMESTAMP',
            'boolean': 'BOOLEAN',
            'bool': 'BOOLEAN'
        }
        return type_mapping.get(column_type.lower(), 'VARCHAR(255)')

    def save_to_csv(self, data: List[Dict[str, Any]], filename: str = None):
        """Save data to CSV file"""
        if not data:
            self.logger.warning("No data to save")
            return

        if filename is None:
            filename = f"{self.table_name}_data.csv"

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        self.logger.info(f"Data saved to {filename}")

    def save_to_json(self, data: List[Dict[str, Any]], filename: str = None):
        """Save data to JSON file"""
        if filename is None:
            filename = f"{self.table_name}_data.json"

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.info(f"Data saved to {filename}")

    def save_to_sql(self, data: List[Dict[str, Any]], filename: str = None):
        """Save data as SQL INSERT statements"""
        if not data:
            return

        if filename is None:
            filename = f"{self.table_name}_data.sql"

        with open(filename, 'w') as f:
            # Create table statement
            f.write(f"CREATE TABLE {self.table_name} (\n")

            column_definitions = []
            primary_keys = self.config.get("primary_key", [])
            required_columns = self.config.get("rules", {}).get("required_columns", [])
            unique_constraints = self.config.get("unique_constraints", [])

            for column_name, column_config in self.columns.items():
                column_type = self._get_sql_type(column_config.get("type", "string"))
                definition = f"  {column_name} {column_type}"

                if column_name in primary_keys:
                    definition += " PRIMARY KEY"
                elif column_name in required_columns:
                    definition += " NOT NULL"

                if column_name in unique_constraints:
                    definition += " UNIQUE"

                column_definitions.append(definition)

            f.write(",\n".join(column_definitions))
            f.write("\n);\n\n")

            # Insert statements
            for record in data:
                values = []
                for column_name in self.columns.keys():
                    value = record.get(column_name)
                    if value is None:
                        values.append("NULL")
                    elif isinstance(value, str):
                        values.append(f"'{value.replace(chr(39), chr(39) + chr(39))}'")  # Escape single quotes
                    else:
                        values.append(str(value))

                column_names = ", ".join(self.columns.keys())
                f.write(f"INSERT INTO {self.table_name} ({column_names}) VALUES ({', '.join(values)});\n")

        self.logger.info(f"SQL statements saved to {filename}")

    def save_to_sqlite(self, data: List[Dict[str, Any]], filename: str = None):
        """Save data to SQLite database"""
        if filename is None:
            filename = f"{self.table_name}_data.db"

        conn = sqlite3.connect(filename)
        df = pd.DataFrame(data)
        df.to_sql(self.table_name, conn, if_exists='replace', index=False)
        conn.close()
        self.logger.info(f"Data saved to SQLite database {filename}")
