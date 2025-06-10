"""
Rule Engine Module - Template-based rule management for data generation
Integrates with OptimizedDataGenerator for enhanced rule processing
"""

import yaml
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import copy


@dataclass
class ColumnRule:
    """Data class for column generation rules"""
    name: str
    type: str
    rule: Dict[str, Any]
    constraints: List[str] = None
    nullable: bool = True
    default: Any = None
    conditional_rules: List[Dict[str, Any]] = None
    sensitivity: str = None


@dataclass
class TableTemplate:
    """Data class for table templates"""
    name: str
    description: str
    columns: List[ColumnRule]
    relationships: List[Dict[str, Any]] = None
    row_count: int = 1000


class RuleEngine:
    """Manage and apply rule templates for data generation"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.templates = {}
        self.global_rules = {}
        self.column_patterns = {}
        self.load_default_patterns()
    
    def load_default_patterns(self):
        """Load default column name patterns and rules"""
        self.column_patterns = {
            # Personal Information
            r'.*email.*': {"type": "email"},
            r'.*phone.*|.*mobile.*|.*tel.*': {"type": "phone_number"},
            r'.*first.*name.*|.*fname.*': "first_name",
            r'.*last.*name.*|.*lname.*|.*surname.*': "last_name",
            r'^name$|.*full.*name.*': "name",
            r'.*age.*': {"type": "range", "min": 18, "max": 80},
            r'.*birth.*|.*dob.*': {
                "type": "date_range", 
                "start": "1950-01-01", 
                "end": "2005-12-31"
            },
            
            # Address Information
            r'.*address.*': "address",
            r'.*street.*': "street_address",
            r'.*city.*': "city",
            r'.*state.*|.*province.*': "state",
            r'.*country.*': "country",
            r'.*zip.*|.*postal.*': "postcode",
            
            # Business Information
            r'.*company.*|.*org.*': "company",
            r'.*job.*|.*position.*|.*title.*': "job",
            r'.*department.*': {"type": "choice", "value": ["IT", "HR", "Finance", "Marketing", "Operations"]},
            
            # Financial
            r'.*price.*|.*amount.*|.*cost.*|.*salary.*': {
                "type": "range", "min": 100, "max": 10000
            },
            r'.*credit.*card.*': "credit_card_number",
            r'.*account.*number.*': {"type": "range", "min": 1000000000, "max": 9999999999},
            
            # Status and Categories
            r'.*status.*': {
                "type": "choice",
                "value": ["active", "inactive", "pending", "completed"],
                "probabilities": {"active": 0.6, "inactive": 0.2, "pending": 0.15, "completed": 0.05}
            },
            r'.*type.*|.*category.*': {
                "type": "choice",
                "value": ["type_a", "type_b", "type_c"],
                "probabilities": {"type_a": 0.5, "type_b": 0.3, "type_c": 0.2}
            },
            
            # Timestamps
            r'.*created.*|.*insert.*': {
                "type": "timestamp_range",
                "start": "2020-01-01 00:00:00",
                "end": "2024-12-31 23:59:59"
            },
            r'.*updated.*|.*modified.*': {
                "type": "timestamp_range",
                "start": "2021-01-01 00:00:00",
                "end": "2024-12-31 23:59:59"
            },
            
            # Technical
            r'.*url.*|.*website.*': "url",
            r'.*uuid.*|.*guid.*': "uuid4",
            r'.*ip.*address.*': "ipv4",
            r'.*mac.*address.*': "mac_address",
            
            # Quantities and Measurements
            r'.*quantity.*|.*count.*|.*num.*': {"type": "range", "min": 1, "max": 100},
            r'.*weight.*': {"type": "range", "min": 0.1, "max": 1000.0},
            r'.*height.*': {"type": "range", "min": 100, "max": 200},
            r'.*distance.*': {"type": "range", "min": 1, "max": 1000},
        }
    
    def load_templates_from_file(self, file_path: str):
        """Load rule templates from YAML or JSON file"""
        try:
            file_path = Path(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._process_template_data(data)
            self.logger.info(f"Loaded templates from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading templates from {file_path}: {e}")
            raise
    
    def _process_template_data(self, data: Dict[str, Any]):
        """Process template data from file"""
        # Load global rules
        if 'global_rules' in data:
            self.global_rules.update(data['global_rules'])
        
        # Load column patterns
        if 'column_patterns' in data:
            self.column_patterns.update(data['column_patterns'])
        
        # Load table templates
        if 'templates' in data:
            for template_name, template_data in data['templates'].items():
                self.templates[template_name] = self._create_table_template(template_name, template_data)
    
    def _create_table_template(self, name: str, data: Dict[str, Any]) -> TableTemplate:
        """Create TableTemplate from dictionary data"""
        columns = []
        
        for col_data in data.get('columns', []):
            column_rule = ColumnRule(
                name=col_data['name'],
                type=col_data['type'],
                rule=col_data.get('rule', {}),
                constraints=col_data.get('constraints', []),
                nullable=col_data.get('nullable', True),
                default=col_data.get('default'),
                conditional_rules=col_data.get('conditional_rules', []),
                sensitivity=col_data.get('sensitivity')
            )
            columns.append(column_rule)
        
        return TableTemplate(
            name=name,
            description=data.get('description', ''),
            columns=columns,
            relationships=data.get('relationships', []),
            row_count=data.get('row_count', 1000)
        )
    
    def create_template_from_schema(self, schema: Dict[str, Any]) -> TableTemplate:
        """Create a template from parsed schema information"""
        table_name = schema.get('table_name', 'generated_table')
        columns = []
        
        for col_name, col_info in schema.get('columns', {}).items():
            # Infer rule based on column name and type
            inferred_rule = self._infer_rule_from_column(col_name, col_info)
            
            column_rule = ColumnRule(
                name=col_name,
                type=col_info.get('type', 'string'),
                rule=inferred_rule,
                constraints=col_info.get('constraints', []),
                nullable=col_info.get('nullable', True),
                default=col_info.get('default'),
                sensitivity=self._determine_sensitivity(col_name)
            )
            columns.append(column_rule)
        
        # Extract relationships if present
        relationships = []
        if 'foreign_keys' in schema:
            for fk in schema['foreign_keys']:
                relationships.append({
                    'type': 'foreign_key',
                    'column': fk['column'],
                    'references': {
                        'table': fk['references']['table'],
                        'column': fk['references']['column']
                    }
                })
        
        return TableTemplate(
            name=table_name,
            description=f"Auto-generated template for {table_name}",
            columns=columns,
            relationships=relationships,
            row_count=1000
        )
    
    def _infer_rule_from_column(self, col_name: str, col_info: Dict[str, Any]) -> Dict[str, Any]:
        """Infer generation rule based on column name and metadata"""
        col_name_lower = col_name.lower()
        
        # Check column name patterns
        for pattern, rule in self.column_patterns.items():
            if re.match(pattern, col_name_lower):
                if isinstance(rule, str):
                    return {"type": rule}
                else:
                    return rule
        
        # Fallback based on data type
        data_type = col_info.get('type', '').lower()
        
        if data_type in ['int', 'integer', 'bigint']:
            return {"type": "range", "min": 1, "max": 1000000}
        elif data_type in ['float', 'double', 'decimal', 'numeric']:
            return {"type": "range", "min": 0.0, "max": 10000.0}
        elif data_type in ['varchar', 'text', 'string']:
            max_length = col_info.get('max_length', 50)
            return {"type": "text", "max_nb_chars": min(max_length, 100)}
        elif data_type in ['date']:
            return {
                "type": "date_range",
                "start": "2020-01-01",
                "end": "2024-12-31"
            }
        elif data_type in ['datetime', 'timestamp']:
            return {
                "type": "timestamp_range",
                "start": "2020-01-01 00:00:00",
                "end": "2024-12-31 23:59:59"
            }
        elif data_type in ['boolean', 'bool']:
            return {"type": "boolean"}
        else:
            return {"type": "text", "max_nb_chars": 50}
    
    def _determine_sensitivity(self, col_name: str) -> Optional[str]:
        """Determine data sensitivity level based on column name"""
        col_name_lower = col_name.lower()
        
        # Highly sensitive patterns
        high_sensitivity_patterns = [
            r'.*ssn.*', r'.*social.*security.*', r'.*tax.*id.*',
            r'.*credit.*card.*', r'.*password.*', r'.*secret.*'
        ]
        
        # Medium sensitivity patterns  
        medium_sensitivity_patterns = [
            r'.*email.*', r'.*phone.*', r'.*mobile.*',
            r'.*address.*', r'.*salary.*', r'.*income.*'
        ]
        
        for pattern in high_sensitivity_patterns:
            if re.match(pattern, col_name_lower):
                return 'high'
        
        for pattern in medium_sensitivity_patterns:
            if re.match(pattern, col_name_lower):
                return 'medium'
        
        return 'low'
    
    def get_template(self, name: str) -> Optional[TableTemplate]:
        """Get a specific template by name"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())
    
    def add_template(self, template: TableTemplate):
        """Add a new template"""
        self.templates[template.name] = template
        self.logger.info(f"Added template: {template.name}")
    
    def remove_template(self, name: str) -> bool:
        """Remove a template by name"""
        if name in self.templates:
            del self.templates[name]
            self.logger.info(f"Removed template: {name}")
            return True
        return False
    
    def update_template(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing template"""
        if name not in self.templates:
            return False
        
        template = self.templates[name]
        
        # Update basic properties
        if 'description' in updates:
            template.description = updates['description']
        if 'row_count' in updates:
            template.row_count = updates['row_count']
        
        # Update columns
        if 'columns' in updates:
            updated_columns = []
            for col_data in updates['columns']:
                column_rule = ColumnRule(
                    name=col_data['name'],
                    type=col_data['type'],
                    rule=col_data.get('rule', {}),
                    constraints=col_data.get('constraints', []),
                    nullable=col_data.get('nullable', True),
                    default=col_data.get('default'),
                    conditional_rules=col_data.get('conditional_rules', []),
                    sensitivity=col_data.get('sensitivity')
                )
                updated_columns.append(column_rule)
            template.columns = updated_columns
        
        # Update relationships
        if 'relationships' in updates:
            template.relationships = updates['relationships']
        
        self.logger.info(f"Updated template: {name}")
        return True
    
    def apply_conditional_rules(self, template: TableTemplate, row_data: Dict[str, Any]) -> TableTemplate:
        """Apply conditional rules based on current row data"""
        modified_template = copy.deepcopy(template)
        
        for column in modified_template.columns:
            if column.conditional_rules:
                for condition in column.conditional_rules:
                    if self._evaluate_condition(condition.get('condition', {}), row_data):
                        # Apply the conditional rule
                        column.rule.update(condition.get('rule', {}))
                        if 'type' in condition:
                            column.type = condition['type']
        
        return modified_template
    
    def _evaluate_condition(self, condition: Dict[str, Any], row_data: Dict[str, Any]) -> bool:
        """Evaluate a conditional rule condition"""
        if not condition:
            return True
        
        column = condition.get('column')
        operator = condition.get('operator', '==')
        value = condition.get('value')
        
        if column not in row_data:
            return False
        
        row_value = row_data[column]
        
        if operator == '==':
            return row_value == value
        elif operator == '!=':
            return row_value != value
        elif operator == '>':
            return row_value > value
        elif operator == '<':
            return row_value < value
        elif operator == '>=':
            return row_value >= value
        elif operator == '<=':
            return row_value <= value
        elif operator == 'in':
            return row_value in value
        elif operator == 'not_in':
            return row_value not in value
        elif operator == 'contains':
            return value in str(row_value)
        
        return False
    
    def export_template(self, name: str, file_path: str, format: str = 'yaml'):
        """Export a template to file"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        template = self.templates[name]
        data = {
            'templates': {
                name: {
                    'description': template.description,
                    'row_count': template.row_count,
                    'columns': [asdict(col) for col in template.columns],
                    'relationships': template.relationships or []
                }
            }
        }
        
        file_path = Path(file_path)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)
        
        self.logger.info(f"Exported template '{name}' to {file_path}")
    
    def validate_template(self, template: TableTemplate) -> List[str]:
        """Validate a template and return list of issues"""
        issues = []
        
        if not template.name:
            issues.append("Template name cannot be empty")
        
        if not template.columns:
            issues.append("Template must have at least one column")
        
        column_names = set()
        for i, column in enumerate(template.columns):
            if not column.name:
                issues.append(f"Column {i} has no name")
            elif column.name in column_names:
                issues.append(f"Duplicate column name: {column.name}")
            else:
                column_names.add(column.name)
            
            if not column.type:
                issues.append(f"Column '{column.name}' has no type specified")
            
            # Validate rule structure
            if column.rule:
                rule_issues = self._validate_rule(column.rule, column.name)
                issues.extend(rule_issues)
        
        # Validate relationships
        if template.relationships:
            for rel in template.relationships:
                if 'column' not in rel:
                    issues.append("Relationship missing 'column' field")
                elif rel['column'] not in column_names:
                    issues.append(f"Relationship references unknown column: {rel['column']}")
        
        return issues
    
    def _validate_rule(self, rule: Dict[str, Any], column_name: str) -> List[str]:
        """Validate a specific rule"""
        issues = []
        rule_type = rule.get('type')
        
        if rule_type == 'range':
            if 'min' not in rule or 'max' not in rule:
                issues.append(f"Range rule for '{column_name}' missing min or max")
            elif rule.get('min', 0) >= rule.get('max', 1):
                issues.append(f"Range rule for '{column_name}' has min >= max")
        
        elif rule_type == 'choice':
            if 'value' not in rule or not rule['value']:
                issues.append(f"Choice rule for '{column_name}' missing or empty value list")
            
            if 'probabilities' in rule:
                probs = rule['probabilities']
                values = rule['value']
                
                if set(probs.keys()) != set(values):
                    issues.append(f"Choice rule for '{column_name}' probability keys don't match values")
                
                if abs(sum(probs.values()) - 1.0) > 0.01:
                    issues.append(f"Choice rule for '{column_name}' probabilities don't sum to 1.0")
        
        elif rule_type in ['date_range', 'timestamp_range']:
            if 'start' not in rule or 'end' not in rule:
                issues.append(f"Date range rule for '{column_name}' missing start or end")
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded templates"""
        stats = {
            'template_count': len(self.templates),
            'total_columns': sum(len(t.columns) for t in self.templates.values()),
            'templates_with_relationships': sum(1 for t in self.templates.values() if t.relationships),
            'sensitivity_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'column_types': {}
        }
        
        for template in self.templates.values():
            for column in template.columns:
                # Count sensitivity levels
                sensitivity = column.sensitivity or 'low'
                stats['sensitivity_distribution'][sensitivity] += 1
                
                # Count column types
                col_type = column.type
                stats['column_types'][col_type] = stats['column_types'].get(col_type, 0) + 1
        
        return stats