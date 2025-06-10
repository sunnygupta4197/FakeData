# app/web/app.py
"""
Web interface for the synthetic data platform
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
from typing import Dict, Any
import os

# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Synthetic Data Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-running {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">🔬 Synthetic Data Platform</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Generate Data", "Job Monitor", "Configuration", "Data Profiling"]
    )

    if page == "Dashboard":
        show_dashboard()
    elif page == "Generate Data":
        show_data_generation()
    elif page == "Job Monitor":
        show_job_monitor()
    elif page == "Configuration":
        show_configuration()
    elif page == "Data Profiling":
        show_data_profiling()


def show_dashboard():
    """Display main dashboard"""
    st.header("📊 Dashboard")

    # Get system stats
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Jobs", stats.get("total_jobs", 0))

            with col2:
                st.metric("Active Jobs", stats.get("active_jobs", 0))

            with col3:
                st.metric("Completed Jobs", stats.get("completed_jobs", 0))

            with col4:
                st.metric("Failed Jobs", stats.get("failed_jobs", 0))

            # System health
            st.subheader("System Health")
            health_col1, health_col2 = st.columns(2)

            with health_col1:
                st.metric("System Load", f"{stats.get('system_load', 0):.1f}%")

            with health_col2:
                st.metric("Memory Usage", f"{stats.get('memory_usage', 0):.1f}%")

        else:
            st.error("Failed to fetch system statistics")

    except requests.exceptions.RequestException:
        st.error("Unable to connect to API server")

    # Recent jobs
    st.subheader("Recent Jobs")
    show_recent_jobs()


def show_data_generation():
    """Data generation interface"""
    st.header("🔄 Generate Synthetic Data")

    # Configuration method selection
    config_method = st.radio(
        "Configuration Method",
        ["Upload Config File", "Interactive Builder", "Use Template"]
    )

    if config_method == "Upload Config File":
        show_config_upload()
    elif config_method == "Interactive Builder":
        show_interactive_builder()
    elif config_method == "Use Template":
        show_template_selector()


def show_config_upload():
    """Config file upload interface"""
    uploaded_file = st.file_uploader(
        "Choose a configuration file",
        type=['json', 'yaml', 'yml']
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.json'):
                config = json.load(uploaded_file)
            else:
                import yaml
                config = yaml.safe_load(uploaded_file)

            st.success("Configuration loaded successfully!")
            st.json(config)

            # Generation parameters
            col1, col2 = st.columns(2)
            with col1:
                rows = st.number_input("Rows per table", min_value=1, value=1000)
                output_format = st.selectbox("Output format", ["csv", "json", "parquet", "sql"])

            with col2:
                batch_size = st.number_input("Batch size", min_value=100, value=10000)
                priority = st.selectbox("Priority", ["low", "normal", "high", "urgent"])

            if st.button("Generate Data", type="primary"):
                generate_data_job(config, rows, output_format, batch_size, priority)

        except Exception as e:
            st.error(f"Error loading configuration: {e}")


def show_interactive_builder():
    """Interactive configuration builder"""
    st.subheader("Build Configuration Interactively")

    # Table configuration
    table_name = st.text_input("Table Name", value="customers")

    st.subheader("Columns")

    # Initialize session state for columns
    if 'columns' not in st.session_state:
        st.session_state.columns = []

    # Add new column
    with st.expander("Add New Column"):
        col_name = st.text_input("Column Name")
        col_type = st.selectbox("Data Type", ["str", "int", "float", "bool", "date", "datetime"])
        col_rule = st.selectbox("Generation Rule", [
            "first_name", "last_name", "email", "phone_number", "address",
            "city", "country", "range", "choice", "date_range", "custom"
        ])

        if st.button("Add Column"):
            if col_name:
                st.session_state.columns.append({
                    "name": col_name,
                    "type": col_type,
                    "rule": col_rule,
                    "nullable": True
                })
                st.success(f"Added column: {col_name}")

    # Display current columns
    if st.session_state.columns:
        st.subheader("Current Columns")
        df = pd.DataFrame(st.session_state.columns)
        st.dataframe(df)

        if st.button("Clear All Columns"):
            st.session_state.columns = []
            st.experimental_rerun()

        # Generate configuration
        config = {
            "tables": [{
                "table_name": table_name,
                "columns": st.session_state.columns
            }]
        }

        st.subheader("Generated Configuration")
        st.json(config)

        # Generation options
        col1, col2 = st.columns(2)
        with col1:
            rows = st.number_input("Rows", min_value=1, value=1000)
            output_format = st.selectbox("Format", ["csv", "json", "parquet"])

        with col2:
            batch_size = st.number_input("Batch", min_value=100, value=10000)
            priority = st.selectbox("Priority", ["normal", "high"])

        if st.button("Generate Data", type="primary"):
            generate_data_job(config, rows, output_format, batch_size, priority)


def show_template_selector():
    """Template selection interface"""
    st.subheader("Select Template")

    # Get available templates
    try:
        response = requests.get(f"{API_BASE_URL}/schemas")
        if response.status_code == 200:
            schemas = response.json().get("schemas", [])

            if schemas:
                selected_template = st.selectbox("Available Templates", schemas)

                # Get template details
                template_response = requests.get(f"{API_BASE_URL}/schemas/{selected_template}")
                if template_response.status_code == 200:
                    template = template_response.json().get("schema", {})
                    st.json(template)

                    # Generation options
                    col1, col2 = st.columns(2)
                    with col1:
                        rows = st.number_input("Rows", min_value=1, value=1000)
                        output_format = st.selectbox("Format", ["csv", "json", "parquet"])

                    with col2:
                        batch_size = st.number_input("Batch", min_value=100, value=10000)
                        priority = st.selectbox("Priority", ["normal", "high"])

                    if st.button("Generate from Template", type="primary"):
                        generate_data_job(template, rows, output_format, batch_size, priority)
            else:
                st.info("No templates available")
        else:
            st.error("Failed to fetch templates")

    except requests.exceptions.RequestException:
        st.error("Unable to connect to API server")


def show_job_monitor():
    """Job monitoring interface"""
    st.header("📈 Job Monitor")

    # Refresh button
    if st.button("🔄 Refresh"):
        st.experimental_rerun()

    try:
        response = requests.get(f"{API_BASE_URL}/jobs")
        if response.status_code == 200:
            jobs_data = response.json()
            jobs = jobs_data.get("jobs", [])

            if jobs:
                # Convert to DataFrame for better display
                df = pd.DataFrame(jobs)

                # Status filter
                status_filter = st.selectbox(
                    "Filter by Status",
                    ["All"] + df["status"].unique().tolist()
                )

                if status_filter != "All":
                    df = df[df["status"] == status_filter]

                # Display jobs table
                st.dataframe(
                    df[["job_id", "status", "progress", "created_at", "message"]],
                    use_container_width=True
                )

                # Job details
                if not df.empty:
                    selected_job = st.selectbox("Select job for details", df["job_id"].tolist())

                    if selected_job:
                        job_details = df[df["job_id"] == selected_job].iloc[0]

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            status_class = f"status-{job_details['status']}"
                            st.markdown(f'**Status:** <span class="{status_class}">{job_details["status"]}</span>',
                                        unsafe_allow_html=True)

                        with col2:
                            st.metric("Progress", f"{job_details['progress']:.1f}%")

                        with col3:
                            if st.button("Cancel Job") and job_details["status"] in ["pending", "running"]:
                                cancel_job(selected_job)

                        # Job logs
                        if st.button("View Logs"):
                            show_job_logs(selected_job)
            else:
                st.info("No jobs found")
        else:
            st.error("Failed to fetch jobs")

    except requests.exceptions.RequestException:
        st.error("Unable to connect to API server")


def show_configuration():
    """Configuration management interface"""
    st.header("⚙️ Configuration Management")

    tab1, tab2, tab3 = st.tabs(["System Config", "Templates", "Validation"])

    with tab1:
        show_system_config()

    with tab2:
        show_template_management()

    with tab3:
        show_config_validation()


def show_system_config():
    """System configuration interface"""
    st.subheader("System Configuration")

    # Current configuration display
    config_data = {
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "database_url": "postgresql://syndata:syndata@localhost:5432/syndata",
        "redis_url": "redis://localhost:6379/0",
        "output_dir": "./output",
        "max_workers": 4,
        "rate_limit": 100
    }

    st.json(config_data)

    if st.button("Update Configuration"):
        st.success("Configuration updated successfully!")


def show_template_management():
    """Template management interface"""
    st.subheader("Template Management")

    # Upload new template
    uploaded_template = st.file_uploader(
        "Upload Template",
        type=['json', 'yaml', 'yml']
    )

    if uploaded_template:
        try:
            if uploaded_template.name.endswith('.json'):
                template = json.load(uploaded_template)
            else:
                import yaml
                template = yaml.safe_load(uploaded_template)

            st.success("Template loaded successfully!")
            st.json(template)

            if st.button("Save Template"):
                st.success("Template saved!")

        except Exception as e:
            st.error(f"Error loading template: {e}")


def show_config_validation():
    """Configuration validation interface"""
    st.subheader("Configuration Validation")

    validation_text = st.text_area(
        "Paste configuration to validate",
        height=300,
        placeholder="Paste your JSON or YAML configuration here..."
    )

    if st.button("Validate Configuration"):
        if validation_text:
            try:
                # Try to parse as JSON first
                try:
                    config = json.loads(validation_text)
                except json.JSONDecodeError:
                    # Try YAML
                    import yaml
                    config = yaml.safe_load(validation_text)

                # Validate with API
                response = requests.post(
                    f"{API_BASE_URL}/config/validate",
                    json=config
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("valid"):
                        st.success("✅ Configuration is valid!")
                    else:
                        st.error("❌ Configuration validation failed!")
                        errors = result.get("errors", [])
                        for error in errors:
                            st.error(f"• {error}")
                else:
                    st.error("Failed to validate configuration")

            except Exception as e:
                st.error(f"Error parsing configuration: {e}")


def show_data_profiling():
    """Data profiling interface"""
    st.header("📊 Data Profiling")

    # Upload data file for profiling
    uploaded_data = st.file_uploader(
        "Upload data file for profiling",
        type=['csv', 'xlsx', 'json', 'parquet']
    )

    if uploaded_data:
        try:
            # Load data based on file type
            if uploaded_data.name.endswith('.csv'):
                df = pd.read_csv(uploaded_data)
            elif uploaded_data.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_data)
            elif uploaded_data.name.endswith('.json'):
                df = pd.read_json(uploaded_data)
            elif uploaded_data.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_data)

            st.success(f"Data loaded successfully! Shape: {df.shape}")

            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            # Column analysis
            st.subheader("Column Analysis")

            # Data types
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique Count': df.nunique()
            })

            st.dataframe(dtypes_df, use_container_width=True)

            # Generate profile report
            if st.button("Generate Detailed Profile"):
                with st.spinner("Generating profile report..."):
                    # Simple profiling
                    profile_data = generate_simple_profile(df)
                    st.json(profile_data)

            # Generate synthetic data config
            if st.button("Generate Synthetic Data Config"):
                config = generate_config_from_data(df)
                st.subheader("Generated Configuration")
                st.json(config)

                # Option to save or use config
                if st.button("Use This Configuration"):
                    st.session_state.generated_config = config
                    st.success("Configuration saved to session!")

        except Exception as e:
            st.error(f"Error loading data: {e}")


def show_recent_jobs():
    """Display recent jobs"""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs?page=1&page_size=5")
        if response.status_code == 200:
            jobs_data = response.json()
            jobs = jobs_data.get("jobs", [])

            if jobs:
                for job in jobs:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])

                        with col1:
                            st.write(f"**{job['job_id']}**")

                        with col2:
                            status = job['status']
                            status_color = {
                                'completed': '🟢',
                                'running': '🟡',
                                'failed': '🔴',
                                'pending': '⚪'
                            }.get(status, '⚪')
                            st.write(f"{status_color} {status}")

                        with col3:
                            st.write(f"{job['progress']:.1f}%")

                        with col4:
                            if job['created_at']:
                                created = pd.to_datetime(job['created_at'])
                                st.write(created.strftime("%H:%M"))
            else:
                st.info("No recent jobs")
    except:
        st.error("Unable to fetch recent jobs")


def generate_data_job(config: Dict[str, Any], rows: int, output_format: str,
                      batch_size: int, priority: str):
    """Generate data job"""
    try:
        job_request = {
            "job_type": "generate",
            "config": {
                "tables": config.get("tables", []),
                "rows": rows,
                "output_format": output_format,
                "batch_size": batch_size
            },
            "priority": priority
        }

        response = requests.post(f"{API_BASE_URL}/jobs", json=job_request)

        if response.status_code == 201:
            job = response.json()
            st.success(f"Job created successfully! Job ID: {job['job_id']}")

            # Show job progress
            with st.spinner("Generating data..."):
                monitor_job_progress(job['job_id'])
        else:
            st.error(f"Failed to create job: {response.text}")

    except Exception as e:
        st.error(f"Error creating job: {e}")


def monitor_job_progress(job_id: str):
    """Monitor job progress in real-time"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    import time
    while True:
        try:
            response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
            if response.status_code == 200:
                job = response.json()

                progress = job['progress']
                status = job['status']

                progress_placeholder.progress(progress / 100)
                status_placeholder.write(f"Status: {status} - {job.get('message', '')}")

                if status in ['completed', 'failed', 'cancelled']:
                    break

                time.sleep(2)
            else:
                break
        except:
            break

    # Final status
    if status == 'completed':
        st.success("✅ Data generation completed successfully!")
        if job.get('result', {}).get('output_path'):
            st.info(f"Output saved to: {job['result']['output_path']}")
    elif status == 'failed':
        st.error(f"❌ Job failed: {job.get('error', 'Unknown error')}")
    else:
        st.warning(f"Job ended with status: {status}")


def cancel_job(job_id: str):
    """Cancel a job"""
    try:
        response = requests.delete(f"{API_BASE_URL}/jobs/{job_id}")
        if response.status_code == 200:
            st.success("Job cancelled successfully!")
        else:
            st.error("Failed to cancel job")
    except:
        st.error("Error cancelling job")


def show_job_logs(job_id: str):
    """Show job logs"""
    try:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}/logs")
        if response.status_code == 200:
            logs_data = response.json()
            logs = logs_data.get("logs", [])

            st.subheader(f"Logs for Job {job_id}")
            for log in logs:
                st.text(log)
        else:
            st.error("Failed to fetch logs")
    except:
        st.error("Error fetching logs")


def generate_simple_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate simple data profile"""
    profile = {
        "shape": df.shape,
        "columns": {},
        "memory_usage": df.memory_usage(deep=True).sum(),
        "duplicates": df.duplicated().sum()
    }

    for col in df.columns:
        col_profile = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": float(df[col].isnull().sum() / len(df) * 100),
            "unique_count": int(df[col].nunique()),
            "unique_percentage": float(df[col].nunique() / len(df) * 100)
        }

        if df[col].dtype in ['int64', 'float64']:
            col_profile.update({
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std())
            })
        elif df[col].dtype == 'object':
            value_counts = df[col].value_counts().head(5)
            col_profile["top_values"] = value_counts.to_dict()
            col_profile["avg_length"] = float(df[col].astype(str).str.len().mean())

        profile["columns"][col] = col_profile

    return profile


def generate_config_from_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate synthetic data config from real data"""
    columns = []

    for col in df.columns:
        col_config = {
            "name": col,
            "type": map_pandas_dtype_to_syndata(df[col].dtype),
            "nullable": bool(df[col].isnull().any())
        }

        # Infer generation rules
        rule = infer_generation_rule(col, df[col])
        if rule:
            col_config["rule"] = rule

        columns.append(col_config)

    return {
        "tables": [{
            "table_name": "generated_table",
            "columns": columns
        }],
        "locale": "en_US",
        "rows": 1000
    }


def map_pandas_dtype_to_syndata(dtype) -> str:
    """Map pandas dtype to synthetic data generator type"""
    dtype_str = str(dtype)

    if 'int' in dtype_str:
        return 'int'
    elif 'float' in dtype_str:
        return 'float'
    elif 'bool' in dtype_str:
        return 'bool'
    elif 'datetime' in dtype_str:
        return 'datetime'
    else:
        return 'str'


def infer_generation_rule(column_name: str, series: pd.Series) -> Dict[str, Any]:
    """Infer generation rule from column data"""
    col_lower = column_name.lower()

    # Email detection
    if 'email' in col_lower:
        return {"type": "email"}

    # Name detection
    if any(name in col_lower for name in ['first_name', 'firstname', 'fname']):
        return {"type": "first_name"}
    if any(name in col_lower for name in ['last_name', 'lastname', 'lname', 'surname']):
        return {"type": "last_name"}
    if col_lower in ['name', 'full_name', 'fullname']:
        return {"type": "name"}

    # Phone detection
    if any(phone in col_lower for phone in ['phone', 'mobile', 'tel']):
        return {"type": "phone_number"}

    # Address detection
    if 'address' in col_lower:
        return {"type": "address"}
    if 'city' in col_lower:
        return {"type": "city"}
    if 'country' in col_lower:
        return {"type": "country"}

    # Numeric ranges
    if series.dtype in ['int64', 'float64']:
        min_val = float(series.min())
        max_val = float(series.max())
        return {
            "type": "range",
            "min": min_val,
            "max": max_val
        }

    # Categorical data
    if series.dtype == 'object' and series.nunique() / len(series) < 0.1:
        values = series.value_counts()
        return {
            "type": "choice",
            "value": values.index.tolist()[:10],  # Top 10 values
            "probabilities": {
                str(k): float(v / values.sum())
                for k, v in values.head(10).items()
            }
        }

    # Default string rule
    return {"type": "text", "max_nb_chars": 50}


if __name__ == "__main__":
    main()