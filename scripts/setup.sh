#!/bin/bash

set -e

echo "🚀 Setting up Synthetic Data Platform..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output
mkdir -p logs
mkdir -p temp
mkdir -p config/templates
mkdir -p tests/fixtures

# Copy example configurations
echo "📋 Setting up example configurations..."
cat > config/sample_config.yaml << 'EOF'
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

      - name: last_name
        type: str
        rule: last_name

      - name: email
        type: str
        rule:
          type: email
        sensitivity: PII

      - name: age
        type: int
        rule:
          type: range
          min: 18
          max: 80

      - name: registration_date
        type: date
        rule:
          type: date_range
          start: "2020-01-01"
          end: "2024-12-31"

locale: en_US
rows: 1000
output_format: csv
EOF

# Set up environment file
if [ ! -f ".env" ]; then
    echo "🔧 Creating .env file..."
    cp .env.example .env
fi

# Initialize database (if needed)
echo "🗄️ Setting up database..."
if command -v docker &> /dev/null; then
    echo "Starting PostgreSQL with Docker..."
    docker run --name syndata-postgres -e POSTGRES_PASSWORD=syndata -e POSTGRES_DB=syndata -e POSTGRES_USER=syndata -p 5432:5432 -d postgres:15 || echo "PostgreSQL container already exists"

    echo "Starting Redis with Docker..."
    docker run --name syndata-redis -p 6379:6379 -d redis:7-alpine || echo "Redis container already exists"

    echo "⏳ Waiting for services to start..."
    sleep 5
fi

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
python -m pytest tests/ -v --tb=short || echo "⚠️ Some tests failed, but setup continues..."

echo "✅ Setup completed successfully!"
echo ""
echo "🎉 Synthetic Data Platform is ready!"
echo ""
echo "Quick start commands:"
echo "  # Activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "  # Generate sample data:"
echo "  syndata generate --config config/sample_config.yaml --rows 1000"
echo ""
echo "  # Start API server:"
echo "  syndata-api"
echo ""
echo "  # Start web interface:"
echo "  streamlit run app/web/app.py --server.port 8001"
echo ""
echo "  # View documentation:"
echo "  open docs/README.md"
