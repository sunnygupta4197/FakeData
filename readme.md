
# 🔬 Synthetic Data Platform

[![CI/CD Pipeline](https://github.com/yourusername/synthetic-data-platform/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/synthetic-data-platform/actions)
[![Coverage](https://codecov.io/gh/yourusername/synthetic-data-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/synthetic-data-platform)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://hub.docker.com/r/syndata/platform)

An enterprise-grade synthetic data generation platform that creates realistic, privacy-safe datasets while preserving statistical properties and relationships of your original data.

## ✨ Features

- 🎯 **Intelligent Data Generation**: ML-driven synthetic data that preserves statistical properties
- 🔗 **Relationship Preservation**: Maintains foreign key relationships and data dependencies
- 🛡️ **Privacy-First**: Built-in PII detection, masking, and anonymization
- 📊 **Auto-Profiling**: Learns from existing data to generate realistic synthetic datasets
- 🚀 **Production-Ready**: REST API, job scheduling, monitoring, and audit logging
- 🎨 **Multi-Format Support**: CSV, JSON, Parquet, SQL, and direct database export
- 🌐 **Web Interface**: User-friendly dashboard for configuration and monitoring
- 📝 **Template System**: Reusable configuration templates for common use cases

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-data-platform.git
cd synthetic-data-platform

# Start the platform
docker-compose up -d

# Access the web interface
open http://localhost:8001

# Generate your first dataset
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d @examples/sample_configs/customer_config.json
```

### Local Installation

```bash
# Setup the project
make setup

# Activate environment
source venv/bin/activate

# Generate sample data
syndata generate --config config/sample_config.yaml --rows 10000

# Start the API server
syndata-api

# Start the web interface
streamlit run app/web/app.py --server.port 8001
```

## 📖 Documentation

- **[User Guide](docs/user_guide.md)** - Complete usage documentation
- **[API Reference](docs/api_reference.md)** - REST API documentation
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions
- **[Examples](examples/)** - Sample configurations and use cases

## 🎯 Use Cases

### Data Privacy & Compliance
```yaml
# Anonymize production data for development
tables:
  - table_name: customers
    columns:
      - name: email
        type: str
        rule: email
        sensitivity: PII
      - name: phone
        type: str
        rule: phone_number
        sensitivity: PII
        masking:
          method: hash
          preserve_format: true
```

### Testing & QA
```bash
# Generate test data for different scenarios
syndata generate \
  --config test_scenarios/edge_cases.yaml \
  --rows 50000 \
  --output-format sql
```

### ML Model Training
```python
from app.core.data_generator import ProfilerIntegratedGenerator

# Generate training data based on production patterns
generator = ProfilerIntegratedGenerator(config)
training_data = generator.generate_from_sample_data(
    sample_data_source="production_sample.csv",
    batch_size=100000
)
```

### Data Sharing
```bash
# Create shareable datasets without sensitive information
syndata generate \
  --config sharing_config.yaml \
  --sample-data protected_data.csv \
  --rows 1000000 \
  --output-format parquet
```

## 🔧 Configuration

### Basic Configuration
```yaml
tables:
  - table_name: users
    columns:
      - name: user_id
        type: int
        constraints: [PK]
        rule:
          type: range
          min: 1
          max: 1000000
      
      - name: email
        type: str
        rule: email
        
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
      - name: name
        type: str
        rule:
          type: choice
          value: ["Engineering", "Marketing", "Sales"]

  - table_name: employees
    columns:
      - name: emp_id
        type: int
        constraints: [PK]
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

## 🛠️ CLI Usage

```bash
# Generate data
syndata generate --config config.yaml --rows 100000

# Profile existing data
syndata profile --data sample.csv --output rules.yaml

# Validate configuration
syndata validate --config config.yaml

# Monitor jobs
syndata status --all

# View job logs
syndata logs job_12345

# Start scheduler
syndata scheduler --config jobs.yaml
```

## 🌐 REST API

```bash
# Create generation job
POST /jobs
{
  "job_type": "generate",
  "config": {...},
  "priority": "high"
}

# Check job status
GET /jobs/{job_id}

# Download results
GET /jobs/{job_id}/download

# System health
GET /health

# Metrics
GET /metrics
```

## 🏗️ Architecture

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

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/test_generator.py -v
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=app --cov-report=html

# Performance tests
make test-performance
```

## 📊 Monitoring

The platform includes comprehensive monitoring and observability:

- **Prometheus Metrics**: Performance and usage metrics
- **Structured Logging**: JSON logs with correlation IDs
- **Health Checks**: Service health monitoring
- **Audit Trail**: Complete audit logging for compliance

```bash
# View metrics
curl http://localhost:8000/metrics

# Check system status
curl http://localhost:8000/stats

# Monitor jobs in real-time
syndata status job_12345 --monitor
```

## 🔒 Security

- **Data Encryption**: At-rest and in-transit encryption
- **Access Control**: Role-based access control
- **Audit Logging**: Complete audit trail
- **Privacy Controls**: PII detection and anonymization
- **Secure Defaults**: Security-first configuration

## 🚀 Performance

- **Batch Processing**: Efficient memory usage for large datasets
- **Parallel Generation**: Multi-threaded data generation
- **Optimized Exports**: Fast data export in multiple formats
- **Caching**: Intelligent caching for repeated operations

### Benchmarks

| Dataset Size | Generation Time | Memory Usage |
|--------------|----------------|--------------|
| 100K rows    | 15 seconds     | 250 MB       |
| 1M rows      | 2.5 minutes    | 1.2 GB       |
| 10M rows     | 25 minutes     | 4.5 GB       |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development setup
git clone https://github.com/yourusername/synthetic-data-platform.git
cd synthetic-data-platform
make setup

# Run tests
make test

# Format code
make format

# Submit PR
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

## 📋 Requirements

- Python 3.8+
- PostgreSQL 12+ (optional, for job persistence)
- Redis 6+ (optional, for caching)
- Docker & Docker Compose (for containerized deployment)

## 🗂️ Project Structure

```
synthetic-data-platform/
├── app/                    # Main application code
│   ├── api/               # REST API and CLI
│   ├── core/              # Core generation engine
│   ├── input/             # Data input and profiling
│   ├── output/            # Export and audit logging
│   ├── jobs/              # Job management and scheduling
│   ├── utils/             # Utilities and helpers
│   └── web/               # Web interface
├── config/                # Configuration files and templates
├── docs/                  # Documentation
├── examples/              # Usage examples
├── tests/                 # Test suite
├── scripts/               # Setup and deployment scripts
├── docker-compose.yml     # Docker composition
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Faker](https://github.com/joke2k/faker) for basic data generation
- Uses [FastAPI](https://fastapi.tiangolo.com/) for the REST API
- Powered by [Streamlit](https://streamlit.io/) for the web interface
- Inspired by privacy-preserving data generation research

## 📞 Support

- **Documentation**: [Read the Docs](https://syndata-platform.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/synthetic-data-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/synthetic-data-platform/discussions)
- **Email**: support@syndata-platform.com

## 🗺️ Roadmap

### v1.1.0 (Q2 2024)
- [ ] Advanced ML-based generation models
- [ ] Real-time streaming data generation
- [ ] Enhanced privacy controls
- [ ] Cloud deployment templates

### v1.2.0 (Q3 2024)
- [ ] Multi-tenant architecture
- [ ] Advanced relationship modeling
- [ ] Custom plugin system
- [ ] Enhanced monitoring and alerting

### v2.0.0 (Q4 2024)
- [ ] Graph-based data generation
- [ ] Advanced anonymization techniques
- [ ] Federated learning integration
- [ ] Enterprise SSO integration

---

⭐ **Star this repository if you find it useful!**

Built with ❤️ by the Synthetic Data Platform team.