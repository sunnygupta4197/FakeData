# test_data_generator.py
# execution syntex - python test_data_generator.py --metadata customer_orders_metadata_extended.json --rows 10

import json
import random
import string
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import re

fake = Faker()

class TestDataGenerator:
    def __init__(self, metadata_file):
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.tables = self.metadata['tables']
        self.unique_values = {}

    def apply_rule(self, rule, col_name=None):
        if isinstance(rule, str):
            if rule == 'email':
                return fake.email()
            elif rule == 'phone_number':
                return fake.phone_number()
            elif rule == 'name':
                return fake.name()
            elif rule == 'date_of_birth':
                return fake.date_of_birth().isoformat()
            elif rule == 'bool':
                return random.choice([True, False])
            elif rule == 'word':
                return fake.word()
            else:
                return fake.word()

        if isinstance(rule, dict):
            if rule.get("type") == "range":
                return random.randint(rule["min"], rule["max"]) if isinstance(rule["min"], int) else round(random.uniform(rule["min"], rule["max"]), 2)
            elif rule.get("type") == "date_range":
                start = datetime.strptime(rule['start'], "%Y-%m-%d")
                end = datetime.strptime(rule['end'], "%Y-%m-%d")
                delta = end - start
                rand_days = random.randint(0, delta.days)
                return (start + timedelta(days=rand_days)).date().isoformat()
            elif rule.get("type") == "regex":
                pattern = rule.get("pattern")
                return self.generate_from_regex(pattern)
            elif rule.get("type") == "list":
                return random.choice(rule.get("values", []))

        return fake.word()

    def generate_from_regex(self, pattern):
        # Simple pattern generator for demonstration (not full regex support)
        if pattern == "^[A-Z]{3}\d{3}$":
            return ''.join(random.choices(string.ascii_uppercase, k=3)) + ''.join(random.choices(string.digits, k=3))
        return fake.word()

    def evaluate_conditions(self, conditions, row):
        for cond in conditions:
            col = cond['column']
            op = cond['operator']
            val = cond['value']
            if col not in row:
                return False
            if op == 'equals' and row[col] != val:
                return False
        return True

    def generate_column_value(self, column, row_context=None):
        col_type = column['type']
        name = column['name']
        rule = column.get('rule')
        conditional_rules = column.get('conditional_rules', [])
        allow_null = column.get('nullable', False)
        unique = 'unique' in column.get('constraints', [])

        # Handle nullability
        if allow_null and random.random() < 0.1:
            return None

        # Conditional logic
        if conditional_rules:
            for cond_rule in conditional_rules:
                if self.evaluate_conditions(cond_rule['when'], row_context):
                    return self.apply_rule(cond_rule['then']['rule'], name)

        # Normal rule
        value = self.apply_rule(rule, name) if rule else self.default_value(col_type)

        # Ensure uniqueness
        if unique:
            seen = self.unique_values.setdefault(name, set())
            while value in seen:
                value = self.apply_rule(rule, name) if rule else self.default_value(col_type)
            seen.add(value)

        return value

    def default_value(self, col_type):
        if col_type == 'int':
            return random.randint(1, 10000)
        elif col_type == 'float':
            return round(random.uniform(100.0, 10000.0), 2)
        elif col_type == 'str':
            return fake.word()
        elif col_type == 'email':
            return fake.email()
        elif col_type == 'date':
            return fake.date()
        elif col_type == 'bool':
            return random.choice([True, False])
        return fake.word()

    def maintain_fk_relationships(self, df_map):
        for table in self.tables:
            fk_list = table.get('foreign_keys', [])
            for fk in fk_list:
                parent_table = fk['parent_table']
                parent_column = fk['parent_column']
                child_column = fk['child_column']
                if parent_table in df_map:
                    df_map[table['table_name']][child_column] = df_map[parent_table][parent_column].sample(n=len(df_map[table['table_name']]), replace=True).values
        return df_map

    def generate_table_data(self, table, num_rows):
        data = []
        for _ in range(num_rows):
            row = {}
            for col in table['columns']:
                row[col['name']] = self.generate_column_value(col, row_context=row)
            data.append(row)
        return pd.DataFrame(data)

    def generate_data(self, num_rows):
        df_map = {}
        for table in self.tables:
            df_map[table['table_name']] = self.generate_table_data(table, num_rows)
        df_map = self.maintain_fk_relationships(df_map)
        return df_map

    def export_to_csv(self, df_map):
        for table_name, df in df_map.items():
            df.to_csv(f"{table_name}.csv", index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test Data Generator Utility')
    parser.add_argument('--metadata', required=True, help='Path to metadata JSON file')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows to generate per table')

    args = parser.parse_args()

    generator = TestDataGenerator(args.metadata)
    df_map = generator.generate_data(args.rows)
    generator.export_to_csv(df_map)
    print(f"Generated test data for {len(df_map)} tables.")