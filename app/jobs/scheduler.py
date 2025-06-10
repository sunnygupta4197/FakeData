"""
Production-Grade Scheduler for Synthetic Data Generation
Supports APScheduler with multiple job stores and executors
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.job import Job
import pytz

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SchedulerMode(str, Enum):
    BLOCKING = "blocking"
    BACKGROUND = "background"

@dataclass
class JobConfig:
    """Configuration for a scheduled job"""
    job_id: str
    name: str
    config_path: str
    schedule_type: str  # 'cron', 'interval', 'date'
    schedule_params: Dict[str, Any]
    timezone: str = "UTC"
    max_instances: int = 1
    coalesce: bool = True
    misfire_grace_time: int = 30
    enabled: bool = True
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class JobExecution:
    """Job execution record"""
    job_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: JobStatus = JobStatus.PENDING
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    rows_generated: int = 0
    execution_time: float = 0.0
    config_used: Optional[Dict] = None

class JobManager:
    """Manages job execution history and metadata"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./job_history")
        self.storage_path.mkdir(exist_ok=True)
        self.executions: Dict[str, JobExecution] = {}
        self._load_history()

    def _load_history(self):
        """Load job execution history from storage"""
        history_file = self.storage_path / "job_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for exec_data in data:
                        execution = JobExecution(**exec_data)
                        self.executions[execution.execution_id] = execution
            except Exception as e:
                logger.error(f"Failed to load job history: {e}")

    def _save_history(self):
        """Save job execution history to storage"""
        history_file = self.storage_path / "job_history.json"
        try:
            with open(history_file, 'w') as f:
                data = [asdict(exec) for exec in self.executions.values()]
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save job history: {e}")

    def start_execution(self, job_id: str, config: Dict) -> str:
        """Start a new job execution"""
        execution_id = f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution = JobExecution(
            job_id=job_id,
            execution_id=execution_id,
            start_time=datetime.now(),
            config_used=config
        )
        self.executions[execution_id] = execution
        self._save_history()
        return execution_id

    def complete_execution(self, execution_id: str, status: JobStatus,
                         output_path: str = None, rows_generated: int = 0,
                         error_message: str = None):
        """Complete a job execution"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            execution.end_time = datetime.now()
            execution.status = status
            execution.output_path = output_path
            execution.rows_generated = rows_generated
            execution.error_message = error_message
            if execution.start_time:
                execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
            self._save_history()

    def get_job_history(self, job_id: str, limit: int = 10) -> List[JobExecution]:
        """Get execution history for a job"""
        executions = [e for e in self.executions.values() if e.job_id == job_id]
        executions.sort(key=lambda x: x.start_time, reverse=True)
        return executions[:limit]

    def get_execution_stats(self, job_id: str) -> Dict[str, Any]:
        """Get execution statistics for a job"""
        executions = [e for e in self.executions.values() if e.job_id == job_id]
        if not executions:
            return {}

        total_runs = len(executions)
        successful_runs = len([e for e in executions if e.status == JobStatus.SUCCESS])
        failed_runs = len([e for e in executions if e.status == JobStatus.FAILED])
        avg_execution_time = sum(e.execution_time for e in executions) / total_runs
        total_rows = sum(e.rows_generated for e in executions)

        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
            "avg_execution_time": avg_execution_time,
            "total_rows_generated": total_rows
        }

class SyntheticDataScheduler:
    """Production-grade scheduler for synthetic data generation jobs"""

    def __init__(self,
                 mode: SchedulerMode = SchedulerMode.BACKGROUND,
                 job_store_url: Optional[str] = None,
                 max_workers: int = 4):
        self.mode = mode
        self.job_manager = JobManager()
        self.data_generator_func: Optional[Callable] = None

        # Configure job stores
        jobstores = {
            'default': SQLAlchemyJobStore(url=job_store_url) if job_store_url else MemoryJobStore()
        }

        # Configure executors
        executors = {
            'default': ThreadPoolExecutor(max_workers),
            'processpool': ProcessPoolExecutor(max_workers)
        }

        # Job defaults
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 30
        }

        # Initialize scheduler
        if mode == SchedulerMode.BLOCKING:
            self.scheduler = BlockingScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=pytz.UTC
            )
        else:
            self.scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=pytz.UTC
            )

        # Add event listeners
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        self.scheduler.add_listener(self._job_missed, EVENT_JOB_MISSED)

        self.jobs: Dict[str, JobConfig] = {}
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for scheduler"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def set_data_generator(self, generator_func: Callable):
        """Set the data generator function to be called by jobs"""
        self.data_generator_func = generator_func

    def add_job(self, job_config: JobConfig) -> bool:
        """Add a new scheduled job"""
        try:
            if not job_config.enabled:
                logger.info(f"Job {job_config.job_id} is disabled, skipping")
                return False

            # Validate schedule parameters
            if not self._validate_schedule_params(job_config):
                logger.error(f"Invalid schedule parameters for job {job_config.job_id}")
                return False

            # Create timezone object
            tz = pytz.timezone(job_config.timezone)

            # Add job based on schedule type
            if job_config.schedule_type == 'cron':
                self.scheduler.add_job(
                    func=self._execute_job,
                    trigger='cron',
                    args=[job_config.job_id],
                    id=job_config.job_id,
                    name=job_config.name,
                    timezone=tz,
                    max_instances=job_config.max_instances,
                    coalesce=job_config.coalesce,
                    misfire_grace_time=job_config.misfire_grace_time,
                    **job_config.schedule_params
                )
            elif job_config.schedule_type == 'interval':
                self.scheduler.add_job(
                    func=self._execute_job,
                    trigger='interval',
                    args=[job_config.job_id],
                    id=job_config.job_id,
                    name=job_config.name,
                    timezone=tz,
                    max_instances=job_config.max_instances,
                    coalesce=job_config.coalesce,
                    misfire_grace_time=job_config.misfire_grace_time,
                    **job_config.schedule_params
                )
            elif job_config.schedule_type == 'date':
                run_date = datetime.fromisoformat(job_config.schedule_params['run_date'])
                self.scheduler.add_job(
                    func=self._execute_job,
                    trigger='date',
                    args=[job_config.job_id],
                    id=job_config.job_id,
                    name=job_config.name,
                    run_date=run_date,
                    timezone=tz,
                    max_instances=job_config.max_instances
                )

            self.jobs[job_config.job_id] = job_config
            logger.info(f"Successfully added job: {job_config.job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add job {job_config.job_id}: {e}")
            return False

    def _validate_schedule_params(self, job_config: JobConfig) -> bool:
        """Validate schedule parameters"""
        if job_config.schedule_type == 'cron':
            required_params = ['minute', 'hour', 'day', 'month', 'day_of_week']
            # At least one cron parameter should be provided
            return any(param in job_config.schedule_params for param in required_params)
        elif job_config.schedule_type == 'interval':
            valid_params = ['weeks', 'days', 'hours', 'minutes', 'seconds']
            return any(param in job_config.schedule_params for param in valid_params)
        elif job_config.schedule_type == 'date':
            return 'run_date' in job_config.schedule_params
        return False

    def _execute_job(self, job_id: str):
        """Execute a scheduled job"""
        if not self.data_generator_func:
            logger.error("No data generator function set")
            return

        job_config = self.jobs.get(job_id)
        if not job_config:
            logger.error(f"Job configuration not found: {job_id}")
            return

        # Load job configuration
        config_path = Path(job_config.config_path)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return

        try:
            # Start execution tracking
            with open(config_path, 'r') as f:
                config = json.load(f)

            execution_id = self.job_manager.start_execution(job_id, config)
            logger.info(f"Starting job execution: {execution_id}")

            # Execute data generation
            result = self.data_generator_func(config_path)

            # Complete execution tracking
            self.job_manager.complete_execution(
                execution_id=execution_id,
                status=JobStatus.SUCCESS,
                output_path=result.get('output_path'),
                rows_generated=result.get('rows_generated', 0)
            )

            logger.info(f"Job execution completed successfully: {execution_id}")

        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            self.job_manager.complete_execution(
                execution_id=execution_id,
                status=JobStatus.FAILED,
                error_message=str(e)
            )

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self.jobs:
                del self.jobs[job_id]
            logger.info(f"Successfully removed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job"""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Successfully paused job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Successfully resumed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False

    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a scheduled job"""
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job_config = self.jobs.get(job_id)
                stats = self.job_manager.get_execution_stats(job_id)
                return {
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time,
                    'trigger': str(job.trigger),
                    'config': asdict(job_config) if job_config else None,
                    'stats': stats
                }
        except Exception as e:
            logger.error(f"Failed to get job info {job_id}: {e}")
        return None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            job_info = self.get_job_info(job.id)
            if job_info:
                jobs.append(job_info)
        return jobs

    def run_job_now(self, job_id: str) -> bool:
        """Execute a job immediately"""
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                self.scheduler.modify_job(job_id, next_run_time=datetime.now())
                logger.info(f"Job {job_id} scheduled for immediate execution")
                return True
            else:
                logger.error(f"Job {job_id} not found")
                return False
        except Exception as e:
            logger.error(f"Failed to run job {job_id} immediately: {e}")
            return False

    def start(self):
        """Start the scheduler"""
        try:
            if not self.scheduler.running:
                self.scheduler.start()
                logger.info(f"Scheduler started in {self.mode.value} mode")
            else:
                logger.warning("Scheduler is already running")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler"""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=wait)
                logger.info("Scheduler shut down successfully")
            else:
                logger.warning("Scheduler is not running")
        except Exception as e:
            logger.error(f"Failed to shutdown scheduler: {e}")

    def _job_executed(self, event):
        """Handle job execution event"""
        logger.info(f"Job {event.job_id} executed successfully")

    def _job_error(self, event):
        """Handle job error event"""
        logger.error(f"Job {event.job_id} failed: {event.exception}")

    def _job_missed(self, event):
        """Handle missed job event"""
        logger.warning(f"Job {event.job_id} missed its scheduled run time")

    def load_jobs_from_config(self, config_file: str):
        """Load multiple jobs from a configuration file"""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Jobs configuration file not found: {config_file}")

        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    jobs_config = yaml.safe_load(f)
                else:
                    jobs_config = json.load(f)

            for job_data in jobs_config.get('jobs', []):
                job_config = JobConfig(**job_data)
                self.add_job(job_config)

            logger.info(f"Loaded {len(jobs_config.get('jobs', []))} jobs from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load jobs from config: {e}")
            raise

# CLI Interface for Scheduler Management
def create_scheduler_cli():
    """Create CLI interface for scheduler management"""
    import argparse

    parser = argparse.ArgumentParser(description='Synthetic Data Scheduler')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Start scheduler
    start_parser = subparsers.add_parser('start', help='Start the scheduler')
    start_parser.add_argument('--config', required=True, help='Jobs configuration file')
    start_parser.add_argument('--mode', choices=['blocking', 'background'],
                            default='background', help='Scheduler mode')
    start_parser.add_argument('--job-store-url', help='Database URL for job store')
    start_parser.add_argument('--max-workers', type=int, default=4,
                            help='Maximum number of worker threads')

    # Add job
    add_parser = subparsers.add_parser('add-job', help='Add a new job')
    add_parser.add_argument('--job-id', required=True, help='Unique job ID')
    add_parser.add_argument('--name', required=True, help='Job name')
    add_parser.add_argument('--config-path', required=True, help='Path to generation config')
    add_parser.add_argument('--schedule-type', choices=['cron', 'interval', 'date'],
                          required=True, help='Schedule type')
    add_parser.add_argument('--schedule', required=True,
                          help='Schedule parameters (JSON format)')
    add_parser.add_argument('--timezone', default='UTC', help='Timezone')

    # List jobs
    list_parser = subparsers.add_parser('list', help='List all jobs')

    # Run job now
    run_parser = subparsers.add_parser('run-now', help='Run job immediately')
    run_parser.add_argument('--job-id', required=True, help='Job ID to run')

    # Remove job
    remove_parser = subparsers.add_parser('remove', help='Remove a job')
    remove_parser.add_argument('--job-id', required=True, help='Job ID to remove')

    # Job history
    history_parser = subparsers.add_parser('history', help='Show job execution history')
    history_parser.add_argument('--job-id', required=True, help='Job ID')
    history_parser.add_argument('--limit', type=int, default=10, help='Number of records')

    return parser

def main():
    """Main entry point for scheduler CLI"""
    parser = create_scheduler_cli()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize scheduler
    mode = SchedulerMode.BLOCKING if args.command == 'start' and args.mode == 'blocking' else SchedulerMode.BACKGROUND
    scheduler = SyntheticDataScheduler(
        mode=mode,
        job_store_url=getattr(args, 'job_store_url', None),
        max_workers=getattr(args, 'max_workers', 4)
    )

    # Set dummy data generator function (replace with actual implementation)
    def dummy_generator(config_path):
        logger.info(f"Generating data with config: {config_path}")
        return {"output_path": "/tmp/output", "rows_generated": 1000}

    scheduler.set_data_generator(dummy_generator)

    try:
        if args.command == 'start':
            # Load jobs from config
            scheduler.load_jobs_from_config(args.config)
            scheduler.start()

            if mode == SchedulerMode.BLOCKING:
                # Keep running until interrupted
                try:
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal")
            else:
                logger.info("Scheduler started in background mode")

        elif args.command == 'add-job':
            schedule_params = json.loads(args.schedule)
            job_config = JobConfig(
                job_id=args.job_id,
                name=args.name,
                config_path=args.config_path,
                schedule_type=args.schedule_type,
                schedule_params=schedule_params,
                timezone=args.timezone
            )
            scheduler.add_job(job_config)

        elif args.command == 'list':
            jobs = scheduler.list_jobs()
            for job in jobs:
                print(f"ID: {job['id']}, Name: {job['name']}, Next Run: {job['next_run_time']}")

        elif args.command == 'run-now':
            scheduler.run_job_now(args.job_id)

        elif args.command == 'remove':
            scheduler.remove_job(args.job_id)

        elif args.command == 'history':
            history = scheduler.job_manager.get_job_history(args.job_id, args.limit)
            for execution in history:
                print(f"Execution: {execution.execution_id}, Status: {execution.status}, "
                      f"Start: {execution.start_time}, Rows: {execution.rows_generated}")

    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1

    finally:
        if scheduler.scheduler.running:
            scheduler.shutdown()

    return 0

# Example job configurations
EXAMPLE_JOBS_CONFIG = {
    "jobs": [
        {
            "job_id": "daily_customer_data",
            "name": "Daily Customer Data Generation",
            "config_path": "./configs/customer_config.json",
            "schedule_type": "cron",
            "schedule_params": {
                "hour": 2,
                "minute": 0
            },
            "timezone": "UTC",
            "enabled": True,
            "tags": ["daily", "customer"]
        },
        {
            "job_id": "weekly_full_dataset",
            "name": "Weekly Full Dataset Generation",
            "config_path": "./configs/full_dataset_config.json",
            "schedule_type": "cron",
            "schedule_params": {
                "day_of_week": 0,  # Monday
                "hour": 1,
                "minute": 0
            },
            "timezone": "UTC",
            "enabled": True,
            "tags": ["weekly", "full-dataset"]
        },
        {
            "job_id": "hourly_incremental",
            "name": "Hourly Incremental Data",
            "config_path": "./configs/incremental_config.json",
            "schedule_type": "interval",
            "schedule_params": {
                "hours": 1
            },
            "timezone": "UTC",
            "enabled": False,
            "tags": ["hourly", "incremental"]
        }
    ]
}

if __name__ == "__main__":
    exit(main())