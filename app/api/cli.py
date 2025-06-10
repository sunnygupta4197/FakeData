# app/api/cli.py
"""
Command Line Interface for Synthetic Data Platform
"""
import click
import json
import yaml
import sys
from pathlib import Path
from typing import Optional
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

console = Console()

# API Base URL
API_BASE_URL = "http://localhost:8000"


@click.group()
@click.option('--api-url', default=API_BASE_URL, help='API base URL')
@click.pass_context
def cli(ctx, api_url):
    """Synthetic Data Platform CLI"""
    ctx.ensure_object(dict)
    ctx.obj['API_URL'] = api_url


@cli.command()
@click.option('--config', '-c', required=True, help='Configuration file path')
@click.option('--rows', '-r', default=1000, help='Number of rows to generate')
@click.option('--output-format', '-f', default='csv',
              type=click.Choice(['csv', 'json', 'parquet', 'sql']),
              help='Output format')
@click.option('--output-dir', '-o', default='./output', help='Output directory')
@click.option('--batch-size', '-b', default=10000, help='Batch size')
@click.option('--priority', '-p', default='normal',
              type=click.Choice(['low', 'normal', 'high', 'urgent']),
              help='Job priority')
@click.option('--wait', '-w', is_flag=True, help='Wait for job completion')
@click.option('--sample-data', '-s', help='Sample data file for profiling')
@click.option('--template', '-t', help='Template file')
@click.pass_context
def generate(ctx, config, rows, output_format, output_dir, batch_size,
             priority, wait, sample_data, template):
    """Generate synthetic data"""
    try:
        # Load configuration
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]Configuration file not found: {config}[/red]")
            sys.exit(1)

        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        # Prepare job request
        job_request = {
            "job_type": "generate",
            "config": {
                **config_data,
                "rows": rows,
                "output_format": output_format,
                "output_path": output_dir,
                "batch_size": batch_size
            },
            "priority": priority,
            "metadata": {
                "cli_initiated": True,
                "sample_data_path": sample_data,
                "template_path": template
            }
        }

        # Submit job
        console.print("[blue]Submitting data generation job...[/blue]")

        response = requests.post(
            f"{ctx.obj['API_URL']}/jobs",
            json=job_request
        )

        if response.status_code == 201:
            job = response.json()
            job_id = job['job_id']

            console.print(f"[green]Job created successfully![/green]")
            console.print(f"Job ID: {job_id}")

            if wait:
                monitor_job(ctx, job_id)
            else:
                console.print(f"Use 'syndata status {job_id}' to check progress")
        else:
            console.print(f"[red]Failed to create job: {response.text}[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('job_id', required=False)
@click.option('--all', '-a', is_flag=True, help='Show all jobs')
@click.option('--status', '-s', help='Filter by status')
@click.option('--limit', '-l', default=10, help='Number of jobs to show')
@click.pass_context
def status(ctx, job_id, all, status, limit):
    """Check job status"""
    try:
        if job_id:
            # Show specific job
            show_job_status(ctx, job_id)
        else:
            # Show job list
            show_job_list(ctx, all, status, limit)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument('job_id')
@click.option('--lines', '-n', default=100, help='Number of log lines')
@click.pass_context
def logs(ctx, job_id, lines):
    """Show job logs"""
    try:
        response = requests.get(
            f"{ctx.obj['API_URL']}/jobs/{job_id}/logs",
            params={'lines': lines}
        )

        if response.status_code == 200:
            logs_data = response.json()
            logs = logs_data.get('logs', [])

            console.print(f"[blue]Logs for job {job_id}:[/blue]")
            for log in logs:
                console.print(log)
        else:
            console.print(f"[red]Failed to fetch logs: {response.text}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument('job_id')
@click.pass_context
def cancel(ctx, job_id):
    """Cancel a job"""
    try:
        response = requests.delete(f"{ctx.obj['API_URL']}/jobs/{job_id}")

        if response.status_code == 200:
            console.print(f"[green]Job {job_id} cancelled successfully[/green]")
        else:
            console.print(f"[red]Failed to cancel job: {response.text}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--config', '-c', required=True, help='Configuration file to validate')
@click.pass_context
def validate(ctx, config):
    """Validate configuration file"""
    try:
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]Configuration file not found: {config}[/red]")
            sys.exit(1)

        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        response = requests.post(
            f"{ctx.obj['API_URL']}/config/validate",
            json=config_data
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('valid'):
                console.print("[green]✅ Configuration is valid[/green]")
            else:
                console.print("[red]❌ Configuration validation failed[/red]")
                errors = result.get('errors', [])
                for error in errors:
                    console.print(f"[red]• {error}[/red]")
        else:
            console.print(f"[red]Validation request failed: {response.text}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--data', '-d', required=True, help='Data file to profile')
@click.option('--output', '-o', help='Output file for profile report')
@click.option('--format', '-f', default='json',
              type=click.Choice(['json', 'yaml', 'html']),
              help='Output format')
@click.pass_context
def profile(ctx, data, output, format):
    """Profile existing data to generate rules"""
    try:
        from app.input.sample_data_profiler import SampleDataProfiler

        console.print(f"[blue]Profiling data file: {data}[/blue]")

        profiler = SampleDataProfiler()

        # Load data
        df = profiler.load_data(data)
        console.print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Profile dataset
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
        ) as progress:
            task = progress.add_task("Profiling data...", total=None)

            profile_results = profiler.profile_dataset(df, "profiled_data")

        # Generate configuration
        config = profiler.generate_test_data_config()

        # Save results
        if output:
            output_path = Path(output)
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2, default=str)
            elif format == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif format == 'html':
                profiler.save_profile_report(output_path, format='html')

            console.print(f"[green]Profile saved to: {output_path}[/green]")
        else:
            # Print to console
            rprint(Panel(json.dumps(config, indent=2, default=str), title="Generated Configuration"))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.pass_context
def dashboard(ctx):
    """Show system dashboard"""
    try:
        # Get system stats
        response = requests.get(f"{ctx.obj['API_URL']}/stats")

        if response.status_code == 200:
            stats = response.json()

            # Create dashboard table
            table = Table(title="System Dashboard")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Jobs", str(stats.get('total_jobs', 0)))
            table.add_row("Active Jobs", str(stats.get('active_jobs', 0)))
            table.add_row("Completed Jobs", str(stats.get('completed_jobs', 0)))
            table.add_row("Failed Jobs", str(stats.get('failed_jobs', 0)))
            table.add_row("System Load", f"{stats.get('system_load', 0):.1f}%")
            table.add_row("Memory Usage", f"{stats.get('memory_usage', 0):.1f}%")

            console.print(table)
        else:
            console.print(f"[red]Failed to fetch stats: {response.text}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def show_job_status(ctx, job_id):
    """Show status of specific job"""
    response = requests.get(f"{ctx.obj['API_URL']}/jobs/{job_id}")

    if response.status_code == 200:
        job = response.json()

        # Create job details table
        table = Table(title=f"Job {job_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Status", job['status'])
        table.add_row("Progress", f"{job['progress']:.1f}%")
        table.add_row("Created", job['created_at'])
        table.add_row("Started", job.get('started_at', 'N/A'))
        table.add_row("Completed", job.get('completed_at', 'N/A'))
        table.add_row("Message", job.get('message', 'N/A'))

        if job.get('error'):
            table.add_row("Error", job['error'])

        if job.get('result'):
            result = job['result']
            if 'output_path' in result:
                table.add_row("Output Path", result['output_path'])
            if 'rows_generated' in result:
                table.add_row("Rows Generated", str(result['rows_generated']))

        console.print(table)
    else:
        console.print(f"[red]Job not found: {job_id}[/red]")


def show_job_list(ctx, show_all, status_filter, limit):
    """Show list of jobs"""
    params = {}
    if status_filter:
        params['status_filter'] = status_filter
    if not show_all:
        params['page_size'] = limit

    response = requests.get(f"{ctx.obj['API_URL']}/jobs", params=params)

    if response.status_code == 200:
        jobs_data = response.json()
        jobs = jobs_data.get('jobs', [])

        if jobs:
            table = Table(title="Jobs")
            table.add_column("Job ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Progress", style="yellow")
            table.add_column("Created", style="blue")
            table.add_column("Message", style="white")

            for job in jobs:
                status_style = {
                    'completed': 'green',
                    'running': 'yellow',
                    'failed': 'red',
                    'pending': 'white'
                }.get(job['status'], 'white')

                table.add_row(
                    job['job_id'][:8] + "...",
                    f"[{status_style}]{job['status']}[/{status_style}]",
                    f"{job['progress']:.1f}%",
                    job['created_at'].split('T')[0],
                    job.get('message', '')[:30] + "..." if job.get('message', '') else ""
                )

            console.print(table)
        else:
            console.print("[yellow]No jobs found[/yellow]")
    else:
        console.print(f"[red]Failed to fetch jobs: {response.text}[/red]")


def monitor_job(ctx, job_id):
    """Monitor job progress in real-time"""
    import time

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
    ) as progress:

        task = progress.add_task("Monitoring job...", total=100)

        while True:
            try:
                response = requests.get(f"{ctx.obj['API_URL']}/jobs/{job_id}")

                if response.status_code == 200:
                    job = response.json()
                    job_progress = job['progress']
                    status = job['status']
                    message = job.get('message', '')

                    progress.update(
                        task,
                        completed=job_progress,
                        description=f"Job {status}: {message}"
                    )

                    if status in ['completed', 'failed', 'cancelled']:
                        break

                    time.sleep(2)
                else:
                    break

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring interrupted[/yellow]")
                break
            except Exception:
                break

    # Final status
    if status == 'completed':
        console.print("[green]✅ Job completed successfully![/green]")
    elif status == 'failed':
        console.print(f"[red]❌ Job failed: {job.get('error', 'Unknown error')}[/red]")
    else:
        console.print(f"[yellow]Job ended with status: {status}[/yellow]")


@cli.command()
@click.option('--config-file', '-c', help='Jobs configuration file')
@click.option('--job-store-url', help='Database URL for job store')
@click.option('--max-workers', default=4, help='Maximum worker threads')
@click.pass_context
def scheduler(ctx, config_file, job_store_url, max_workers):
    """Start the job scheduler"""
    try:
        from app.jobs.scheduler import SyntheticDataScheduler, SchedulerMode

        console.print("[blue]Starting job scheduler...[/blue]")

        scheduler = SyntheticDataScheduler(
            mode=SchedulerMode.BLOCKING,
            job_store_url=job_store_url,
            max_workers=max_workers
        )

        # Load jobs from config file if provided
        if config_file:
            scheduler.load_jobs_from_config(config_file)

        # Start scheduler
        scheduler.start()

    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Scheduler error: {e}[/red]")


def main():
    """Main entry point"""
    cli()


if __name__ == '__main__':
    main()

