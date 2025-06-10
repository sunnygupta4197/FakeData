# Synthetic Data Platform Documentation

## Overview

The Synthetic Data Platform is an enterprise-grade solution for generating high-quality synthetic data that maintains the statistical properties and relationships of your original data while ensuring privacy and compliance.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-data-platform.git
cd synthetic-data-platform

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Generate data from configuration
syndata generate --config examples/sample_configs/customer_config.json --rows 1000

# Profile existing data
syndata profile --data sample_data.csv --output profile_report.json

# Validate configuration
syndata validate --config my_config.yaml

# Monitor jobs
syndata status --all
```

### Docker Usage

```bash
# Start the full platform
docker-compose up -d

# Access the web interface
open http://localhost:8001

# Access the API
curl http://localhost:8000/health
```

## Features

### 🔄 Data Generation
- **Multi-format support**: CSV, JSON, Parquet, SQL
- **Relationship preservation**: Maintains foreign key relationships
- **Smart rules engine**: Configurable generation rules and templates
- **Conditional logic**: Context-aware data generation
- **Performance optimized**: Batch processing and memory management

### 📊 Data Profiling
- **Automatic analysis**: Statistical profiling of existing data
- **Pattern detection**: Identifies emails, phones, names, etc.
- **Distribution learning**: Learns data distributions for realistic generation
- **Quality assessment**: Data quality metrics and recommendations

### 🔒 Privacy & Security
- **Data masking**: Static and dynamic masking options
- **Anonymization**: PII detection and anonymization
- **Audit logging**: Complete audit trail for compliance
- **Role-based access**: User authentication and authorization

### 🚀 Enterprise Features
- **REST API**: Full programmatic access
- **Job scheduling**: Automated data generation workflows
- **Web interface**: User-friendly dashboard
- **Monitoring**: Real-time job monitoring and metrics
- **Scalability**: Distributed processing support

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │   Core Engine   │    │  Output Layer   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ Schema Parser   │────│ Rule Engine     │────│ Export Manager  │
│ Data Profiler   │    │ Data Generator  │    │ Audit Logger    │
│ Sample Importer │    │ Relationship    │    │ Format Writers  │
│                 │    │ Preserver       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Orchestration  │
                    ├─────────────────┤
                    │ REST API        │
                    │ CLI Interface   │
                    │ Web Dashboard   │
                    │ Job Scheduler   │
                    └─────────────────┘
```

## Configuration

### Basic Configuration

```yaml
tables:
  - table_name: customers
    columns:
      - name: customer_id
        type: int
        constraints: [PK]
        rule:
          type: range
          min: 1
          max: 10000
      
      - name: first_name
        type: str
        rule: first_name
      
      - name: email
        type: str
        rule:
          type: email
          regex: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        sensitivity: PII
      
      - name: age
        type: int
        rule:
          type: range
          min: 18
          max: 80

locale: en_US
rows: 10000
output_format: csv
```

### Advanced Configuration with Relationships

```yaml
tables:
  - table_name: departments
    columns:
      - name: dept_id
        type: int
        constraints: [PK]
        rule:
          type: range
          min: 1
          max: 10
      
      - name: dept_name
        type: str
        rule:
          type: choice
          value: ["Engineering", "Marketing", "Sales", "HR"]

  - table_name: employees
    columns:
      - name: emp_id
        type: int
        constraints: [PK]
        rule:
          type: range
          min: 1000
          max: 9999
      
      - name: dept_id
        type: int
        constraints: [FK]
      
      - name: salary
        type: float
        conditional_rules:
          - condition:
              column: dept_id
              operator: "=="
              value: 1  # Engineering
            rule:
              type: range
              min: 80000
              max: 150000
    
    foreign_keys:
      - parent_table: departments
        parent_column: dept_id
        child_column: dept_id
```

## API Reference

### Authentication

```bash
# Set API key (if authentication is enabled)
export SYNDATA_API_KEY="your-api-key-here"
```

### Core Endpoints

#### Jobs

```bash
# Create a new job
POST /jobs
Content-Type: application/json

{
  "job_type": "generate",
  "config": {
    "tables": [...],
    "rows": 1000,
    "output_format": "csv"
  },
  "priority": "normal"
}

# Get job status
GET /jobs/{job_id}

# List all jobs
GET /jobs?page=1&page_size=20&status_filter=completed

# Cancel a job
DELETE /jobs/{job_id}

# Get job logs
GET /jobs/{job_id}/logs?lines=100
```

#### Configuration

```bash
# Validate configuration
POST /config/validate
Content-Type: application/json

{
  "tables": [...],
  "locale": "en_US"
}

# Upload configuration file
POST /config/upload
Content-Type: multipart/form-data

# Get schema templates
GET /schemas
GET /schemas/{schema_name}
```

#### System

```bash
# Health check
GET /health

# System statistics
GET /stats

# Metrics (Prometheus format)
GET /metrics
```

## CLI Reference

### Data Generation

```bash
# Basic generation
syndata generate \
  --config config.yaml \
  --rows 10000 \
  --output-format csv \
  --output-dir ./output

# With profiling
syndata generate \
  --config config.yaml \
  --sample-data existing_data.csv \
  --rows 50000 \
  --wait

# Using templates
syndata generate \
  --template customer_template.yaml \
  --rows 25000 \
  --priority high
```

### Data Profiling

```bash
# Profile data and generate config
syndata profile \
  --data sample_data.csv \
  --output generated_config.yaml \
  --format yaml

# Generate HTML report
syndata profile \
  --data large_dataset.parquet \
  --output profile_report.html \
  --format html
```

### Job Management

```bash
# List all jobs
syndata status --all

# Filter by status
syndata status --status running --limit 5

# Monitor specific job
syndata status job_12345 --monitor

# View job logs
syndata logs job_12345 --lines 200

# Cancel running job
syndata cancel job_12345
```

### Configuration Management

```bash
# Validate configuration
syndata validate --config my_config.yaml

# Start scheduler
syndata scheduler \
  --config-file jobs_config.yaml \
  --max-workers 8

# Show dashboard
syndata dashboard
```

## Web Interface

The web interface provides a user-friendly way to interact with the platform:

- **Dashboard**: System overview and metrics
- **Data Generation**: Interactive configuration builder
- **Job Monitor**: Real-time job tracking
- **Configuration**: Template management and validation
- **Data Profiling**: Upload and analyze existing data

Access the web interface at `http://localhost:8001` when running with Docker.

## Advanced Features

### Custom Rules

Create custom generation rules:

```python
# app/core/custom_rules.py
from app.core.data_generator import EnhancedDataGenerator

class CustomGenerator(EnhancedDataGenerator):
    def generate_custom_rule(self, rule_config):
        # Your custom logic here
        return generated_value

# Use in configuration
{
  "name": "special_field",
  "type": "str",
  "rule": {
    "type": "custom",
    "custom_class": "CustomGenerator",
    "custom_method": "generate_custom_rule",
    "parameters": {...}
  }
}
```

### Machine Learning Integration

Use ML models for data generation:

```python
from app.core.data_generator import ProfilerIntegratedGenerator
from app.input.sample_data_profiler import SampleDataProfiler

# Create profiler-integrated generator
profiler = SampleDataProfiler()
generator = ProfilerIntegratedGenerator(config, locale="en_US")
generator.set_profiler(profiler)

# Generate data based on learned patterns
data = generator.generate_from_sample_data(
    sample_data_source="training_data.csv",
    table_name="customers",
    batch_size=10000
)
```

### Distributed Processing

Scale horizontally using Celery:

```python
# app/jobs/celery_tasks.py
from celery import Celery

app = Celery('syndata')

@app.task
def generate_data_task(config, table_name, batch_size):
    # Distributed generation logic
    pass

# Scale workers
celery -A app.jobs.celery_tasks worker --loglevel=info --concurrency=4
```

## Monitoring and Observability

### Metrics

The platform exposes Prometheus metrics:

```
# HELP syndata_jobs_total Total number of jobs
# TYPE syndata_jobs_total counter
syndata_jobs_total{status="completed"} 150
syndata_jobs_total{status="failed"} 5

# HELP syndata_generation_duration_seconds Time taken to generate data
# TYPE syndata_generation_duration_seconds histogram
syndata_generation_duration_seconds_bucket{le="10"} 45
syndata_generation_duration_seconds_bucket{le="30"} 89
syndata_generation_duration_seconds_bucket{le="60"} 120

# HELP syndata_rows_generated_total Total rows generated
# TYPE syndata_rows_generated_total counter
syndata_rows_generated_total 1500000
```

### Logging

Structured logging with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "app.core.data_generator",
  "message": "Generated batch completed",
  "job_id": "job_12345",
  "table_name": "customers",
  "batch_size": 1000,
  "duration_ms": 1500,
  "rows_generated": 1000
}
```

### Alerts

Set up alerts for:
- Job failures
- Long-running jobs
- High memory usage
- API errors

## Performance Tuning

### Memory Optimization

```python
# Tune batch sizes based on available memory
BATCH_SIZE_CONFIG = {
    "small_tables": 50000,   # < 10 columns
    "medium_tables": 25000,  # 10-50 columns
    "large_tables": 10000,   # > 50 columns
}

# Enable memory monitoring
generator = EnhancedDataGenerator(
    config=config,
    memory_limit_mb=8192,
    batch_size=BATCH_SIZE_CONFIG["medium_tables"]
)
```

### Database Performance

```python
# Optimize database connections
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 60,
    "pool_recycle": 3600
}

# Use bulk operations
writer = SQLWriter(
    connection=db_connection,
    bulk_insert_size=10000,
    use_copy=True  # PostgreSQL COPY for faster inserts
)
```

## Security Best Practices

### Data Protection

1. **Encrypt sensitive data** at rest and in transit
2. **Use secure randomization** for sensitive fields
3. **Implement proper access controls** for data access
4. **Audit all operations** for compliance

### API Security

```python
# Enable authentication
from fastapi.security import HTTPBearer
from app.auth import verify_token

security = HTTPBearer()

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Implement your authentication logic
    return await call_next(request)
```

### Environment Configuration

```bash
# Use environment variables for secrets
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export SECRET_KEY="your-secret-key"
export JWT_SECRET="your-jwt-secret"

# Enable SSL
export SSL_CERT_PATH="/path/to/cert.pem"
export SSL_KEY_PATH="/path/to/key.pem"
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Monitor memory usage
syndata dashboard

# Reduce batch size
syndata generate --config config.yaml --batch-size 5000

# Enable memory profiling
export SYNDATA_MEMORY_PROFILING=true
```

#### Performance Issues
```bash
# Check system stats
curl http://localhost:8000/stats

# Profile generation
syndata generate --config config.yaml --profile-performance

# Use multiple workers
syndata scheduler --max-workers 8
```

#### Relationship Issues
```bash
# Validate relationships
syndata validate --config config.yaml --check-relationships

# Debug foreign key issues
syndata generate --config config.yaml --debug-relationships
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Invalid configuration | Check config syntax and validation |
| 404 | Job not found | Verify job ID |
| 429 | Rate limit exceeded | Reduce request frequency |
| 500 | Internal server error | Check logs and system resources |

### Logs Location

```bash
# Application logs
tail -f logs/syndata.log

# Job-specific logs
tail -f logs/jobs/job_12345.log

# System logs
tail -f /var/log/syndata/system.log
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/synthetic-data-platform.git
cd synthetic-data-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Code formatting
black app/ tests/
flake8 app/ tests/

# Type checking
mypy app/
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_generator.py -v
pytest tests/test_api.py::test_job_creation

# Run with coverage
pytest --cov=app --cov-report=html

# Run integration tests
pytest tests/integration/ -v
```

### Building Documentation

```bash
# Install docs dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# Serve documentation
python -m http.server 8080 -d _build/html/
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t syndata-platform .

# Run with Docker Compose
docker-compose up -d

# Scale services
docker-compose up -d --scale app=3 --scale scheduler=2
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: syndata-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: syndata-platform
  template:
    metadata:
      labels:
        app: syndata-platform
    spec:
      containers:
      - name: syndata
        image: syndata-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: syndata-secrets
              key: database-url
```

### Production Configuration

```bash
# Environment variables
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO
export API_WORKERS=4
export DATABASE_POOL_SIZE=20
export REDIS_MAX_CONNECTIONS=100

# SSL Configuration
export SSL_ENABLED=true
export SSL_CERT_PATH=/etc/ssl/certs/syndata.pem
export SSL_KEY_PATH=/etc/ssl/private/syndata.key

# Monitoring
export PROMETHEUS_ENABLED=true
export SENTRY_DSN=https://your-sentry-dsn
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://syndata-platform.readthedocs.io
- **Issues**: https://github.com/yourusername/synthetic-data-platform/issues
- **Discussions**: https://github.com/yourusername/synthetic-data-platform/discussions
- **Email**: support@syndata-platform.com

## Changelog

### v1.0.0 (Latest)
- Initial release
- Complete data generation engine
- REST API and CLI
- Web interface
- Docker support
- Comprehensive documentation

### Roadmap

- **v1.1.0**: ML-based data generation
- **v1.2.0**: Real-time streaming support
- **v1.3.0**: Advanced privacy features
- **v2.0.0**: Multi-tenant architecture