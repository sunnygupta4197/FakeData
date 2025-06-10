"""
Job Scheduler for Synthetic Data Generation Platform
Supports cron-like scheduling, recurring jobs, and integration with APScheduler.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

from .jobs.job_manager import JobManager, JobStatus
from .core.data_generator import OptimizedDataGenerator
from .utils.config_manager import ConfigManager
from .utils.exceptions import SchedulerError
from .output.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Types of scheduling."""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ScheduleStatus(str, Enum):
    """Status of scheduled jobs."""
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ScheduledJob:
    """Configuration for a scheduled job."""
    schedule_id: str
    name: str
    description: Optional[str] = None
    schedule_type: ScheduleType = ScheduleType.ONCE
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    job_config: Dict[str, Any] = field(default_factory=dict)
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None


class DataGenerationScheduler:
    """Main scheduler for data generation jobs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scheduled_jobs: Dict[str, ScheduledJob] = {}
        self.job_manager = JobManager()
        self.config_manager = ConfigManager()
        self.audit_logger = AuditLogger()

        # Configure APScheduler
        self.scheduler = self._create_scheduler()
        self.is_running = False

    def _create_scheduler(self) -> AsyncIOScheduler:
        """Create and configure APScheduler instance."""
        jobstores = {
            'default': SQLAlchemyJobStore(
                url=self.config.get('database_url', 'sqlite:///scheduler.db')
            )
        }

        executors = {
            'default': AsyncIOExecutor()
        }

        job_defaults = {
            'coalesce': True,
            'max_instances': self.config.get('max_concurrent_jobs', 3),
            'misfire_grace_time': self.config.get('misfire_grace_time', 300)  # 5 minutes
        }

        scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )

        # Add event listeners
        scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        scheduler.add_listener(self._job_missed, EVENT_JOB_MISSED)

        return scheduler

    async def start(self):
        """Start the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        try:
            self.scheduler.start()
            self.is_running = True
            logger.info("Scheduler started successfully")

            # Log startup
            await self.audit_logger.log_system_event(
                event_type="SCHEDULER_STARTED",
                metadata={"scheduler_id": id(self)}
            )

        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
            raise SchedulerError(f"Failed to start scheduler: {str(e)}")

    async def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return

        try:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("Scheduler stopped successfully")

            # Log shutdown
            await self.audit_logger.log_system_event(
                event_type="SCHEDULER_STOPPED",
                metadata={"scheduler_id": id(self)}
            )

        except Exception as e:
            logger.error(f"Error stopping scheduler: {str(e)}")
            raise SchedulerError(f"Error stopping scheduler: {str(e)}")

    async def schedule_job(
        self,
        name: str,
        job_config: Dict[str, Any],
        schedule_type: ScheduleType,
        trigger_config: Dict[str, Any],
        description: Optional[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        created_by: Optional[str] = None,
        max_runs: Optional[int] = None
    ) -> str:
        """Schedule a new data generation job."""
        schedule_id = str(uuid.uuid4())

        try:
            # Create scheduled job record
            scheduled_job = ScheduledJob(
                schedule_id=schedule_id,
                name=name,
                description=description,
                schedule_type=schedule_type,
                trigger_config=trigger_config,
                job_config=job_config,
                tags=tags or [],
                metadata=metadata or {},
                created_by=created_by,
                max_runs=max_runs
            )

            # Validate job configuration
            self.config_manager.validate_config(job_config)

            # Create APScheduler trigger
            trigger = self._create_trigger(schedule_type, trigger_config)

            # Schedule the job
            self.scheduler.add_job(
                func=self._execute_scheduled_job,
                trigger=trigger,
                args=[schedule_id],
                id=schedule_id,
                name=name,
                replace_existing=True
            )

            # Store scheduled job
            self.scheduled_jobs[schedule_id] = scheduled_job

            # Update next run time
            apscheduler_job = self.scheduler.get_job(schedule_id)
            if apscheduler_job:
                scheduled_job.next_run = apscheduler_job.next_run_time

            logger.info(f"Successfully scheduled job '{name}' with ID {schedule_id}")

            # Log scheduling
            await self.audit_logger.log_system_event(
                event_type="JOB_SCHEDULED",
                metadata={
                    "schedule_id": schedule_id,
                    "name": name,
                    "schedule_type": schedule_type.value,
                    "created_by": created_by
                }
            )

            return schedule_id

        except Exception as e:
            logger.error(f"Failed to schedule job '{name}': {str(e)}")
            raise SchedulerError(f"Failed to schedule job: {str(e)}")

    def _create_trigger(self, schedule_type: ScheduleType, config: Dict[str, Any]):
        """Create APScheduler trigger based on schedule type and configuration."""
        if schedule_type == ScheduleType.ONCE:
            run_date = config.get('run_date')
            if isinstance(run_date, str):
                run_date = datetime.fromisoformat(run_date)
            return DateTrigger(run_date=run_date)

        elif schedule_type == ScheduleType.INTERVAL:
            return IntervalTrigger(
                seconds=config.get('seconds', 0),
                minutes=config.get('minutes', 0),
                hours=config.get('hours', 0),
                days=config.get('days', 0),
                weeks=config.get('weeks', 0),
                start_date=config.get('start_date'),
                end_date=config.get('end_date')
            )

        elif schedule_type == ScheduleType.CRON:
            return CronTrigger(
                year=config.get('year'),
                month=config.get('month'),
                day=config.get('day'),
                week=config.get('week'),
                day_of_week=config.get('day_of_week'),
                hour=config.get('hour'),
                minute=config.get('minute'),
                second=config.get('second'),
                start_date=config.get('start_date'),
                end_date=config.get('end_date'),
                timezone=config.get('timezone', 'UTC')
            )

        elif schedule_type == ScheduleType.DAILY:
            return CronTrigger(
                hour=config.get('hour', 0),
                minute=config.get('minute', 0),
                second=config.get('second', 0)
            )

        elif schedule_type == ScheduleType.WEEKLY:
            return CronTrigger(
                day_of_week=config.get('day_of_week', 0),
                hour=config.get('hour', 0),
                minute=config.get('minute', 0),
                second=config.get('second', 0)
            )

        elif schedule_type == ScheduleType.MONTHLY:
            return CronTrigger(
                day=config.get('day', 1),
                hour=config.get('hour', 0),
                minute=config.get('minute', 0),
                second=config.get('second', 0)
            )

        else:
            raise SchedulerError(f"Unsupported schedule type: {schedule_type}")

    async def _execute_scheduled_job(self, schedule_id: str):
        """Execute a scheduled data generation job."""
        scheduled_job = self.scheduled_jobs.get(schedule_id)
        if not scheduled_job:
            logger.error(f"Scheduled job {schedule_id} not found")
            return

        try:
            logger.info(f"Executing scheduled job: {scheduled_job.name}")

            # Check if job should still run
            if scheduled_job.status != ScheduleStatus.ACTIVE:
                logger.info(f"Skipping inactive scheduled job: {scheduled_job.name}")
                return

            # Check max runs limit
            if scheduled_job.max_runs and scheduled_job.run_count >= scheduled_job.max_runs:
                logger.info(f"Scheduled job {scheduled_job.name} has reached max runs limit")
                await self.pause_scheduled_job(schedule_id)
                return

            # Create execution job
            job_id = str(uuid.uuid4())
            await self.job_manager.create_job(
                job_id=job_id,
                job_type="generate",
                config=scheduled_job.job_config,
                priority="normal",
                tags=scheduled_job.tags + ["scheduled"],
                metadata={
                    "schedule_id": schedule_id,
                    "scheduled_job_name": scheduled_job.name,
                    "run_number": scheduled_job.run_count + 1
                }
            )

            # Execute the job
            generator = OptimizedDataGenerator(
                config=scheduled_job.job_config,
                audit_logger=self.audit_logger
            )

            result = await generator.generate_async()

            # Update scheduled job statistics
            scheduled_job.last_run = datetime.utcnow()
            scheduled_job.run_count += 1
            scheduled_job.updated_at = datetime.utcnow()

            # Update next run time
            apscheduler_job = self.scheduler.get_job(schedule_id)
            if apscheduler_job:
                scheduled_job.next_run = apscheduler_job.next_run_time

            # Complete the job
            await self.job_manager.complete_job(job_id, result)

            logger.info(f"Successfully executed scheduled job: {scheduled_job.name}")

            # Log execution
            await self.audit_logger.log_system_event(
                event_type="SCHEDULED_JOB_EXECUTED",
                metadata={
                    "schedule_id": schedule_id,
                    "job_id": job_id,
                    "run_count": scheduled_job.run_count,
                    "rows_generated": result.get("total_rows", 0)
                }
            )

        except Exception as e:
            logger.error(f"Error executing scheduled job {scheduled_job.name}: {str(e)}")

            # Log error
            await self.audit_logger.log_system_event(
                event_type="SCHEDULED_JOB_ERROR",
                metadata={
                    "schedule_id": schedule_id,
                    "error": str(e)
                }
            )

            raise

    async def get_scheduled_job(self, schedule_id: str) -> Optional[ScheduledJob]:
        """Get a scheduled job by ID."""
        return self.scheduled_jobs.get(schedule_id)

    async def list_scheduled_jobs(
        self,
        status_filter: Optional[ScheduleStatus] = None,
        tags_filter: List[str] = None
    ) -> List[ScheduledJob]:
        """List all scheduled jobs with optional filtering."""
        jobs = list(self.scheduled_jobs.values())

        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]

        if tags_filter:
            jobs = [job for job in jobs if any(tag in job.tags for tag in tags_filter)]

        return jobs

    async def pause_scheduled_job(self, schedule_id: str) -> bool:
        """Pause a scheduled job."""
        scheduled_job = self.scheduled_jobs.get(schedule_id)
        if not scheduled_job:
            return False

        try:
            self.scheduler.pause_job(schedule_id)
            scheduled_job.status = ScheduleStatus.PAUSED
            scheduled_job.updated_at = datetime.utcnow()

            logger.info(f"Paused scheduled job: {scheduled_job.name}")

            # Log pause
            await self.audit_logger.log_system_event(
                event_type="SCHEDULED_JOB_PAUSED",
                metadata={"schedule_id": schedule_id}
            )

            return True

        except Exception as e:
            logger.error(f"Error pausing scheduled job {schedule_id}: {str(e)}")
            return False

    async def resume_scheduled_job(self, schedule_id: str) -> bool:
        """Resume a paused scheduled job."""
        scheduled_job = self.scheduled_jobs.get(schedule_id)
        if not scheduled_job:
            return False

        try:
            self.scheduler.resume_job(schedule_id)
            scheduled_job.status = ScheduleStatus.ACTIVE
            scheduled_job.updated_at = datetime.utcnow()

            # Update next run time
            apscheduler_job = self.scheduler.get_job(schedule_id)
            if apscheduler_job:
                scheduled_job.next_run = apscheduler_job.next_run_time

            logger.info(f"Resumed scheduled job: {scheduled_job.name}")

            # Log resume
            await self.audit_logger.log_system_event(
                event_type="SCHEDULED_JOB_RESUMED",
                metadata={"schedule_id": schedule_id}
            )

            return True

        except Exception as e:
            logger.error(f"Error resuming scheduled job {schedule_id}: {str(e)}")
            return False

    async def cancel_scheduled_job(self, schedule_id: str) -> bool:
        """Cancel a scheduled job."""
        scheduled_job = self.scheduled_jobs.get(schedule_id)
        if not scheduled_job:
            return False

        try:
            self.scheduler.remove_job(schedule_id)
            scheduled_job.status = ScheduleStatus.CANCELLED
            scheduled_job.updated_at = datetime.utcnow()

            logger.info(f"Cancelled scheduled job: {scheduled_job.name}")

            # Log cancellation
            await self.audit_logger.log_system_event(
                event_type="SCHEDULED_JOB_CANCELLED",
                metadata={"schedule_id": schedule_id}
            )

            return True

        except Exception as e:
            logger.error(f"Error cancelling scheduled job {schedule_id}: {str(e)}")
            return False

    async def update_scheduled_job(
        self,
        schedule_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a scheduled job configuration."""
        scheduled_job = self.scheduled_jobs.get(schedule_id)
        if not scheduled_job:
            return False

        try:
            # Update job configuration if provided
            if 'job_config' in updates:
                self.config_manager.validate_config(updates['job_config'])
                scheduled_job.job_config = updates['job_config']

            # Update schedule configuration if provided
            if 'trigger_config' in updates:
                trigger = self._create_trigger(scheduled_job.schedule_type, updates['trigger_config'])
                self.scheduler.reschedule_job(schedule_id, trigger=trigger)
                scheduled_job.trigger_config = updates['trigger_config']

                # Update next run time
                apscheduler_job = self.scheduler.get_job(schedule_id)
                if apscheduler_job:
                    scheduled_job.next_run = apscheduler_job.next_run_time

            # Update other fields
            for field in ['name', 'description', 'tags', 'metadata', 'max_runs']:
                if field in updates:
                    setattr(scheduled_job, field, updates[field])

            scheduled_job.updated_at = datetime.utcnow()

            logger.info(f"Updated scheduled job: {scheduled_job.name}")

            # Log update
            await self.audit_logger.log_system_event(
                event_type="SCHEDULED_JOB_UPDATED",
                metadata={"schedule_id": schedule_id, "updates": list(updates.keys())}
            )

            return True

        except Exception as e:
            logger.error(f"Error updating scheduled job {schedule_id}: {str(e)}")
            return False

    async def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "is_running": self.is_running,
            "total_scheduled_jobs": len(self.scheduled_jobs),
            "active_jobs": len([j for j in self.scheduled_jobs.values() if j.status == ScheduleStatus.ACTIVE]),
            "paused_jobs": len([j for j in self.scheduled_jobs.values() if j.status == ScheduleStatus.PAUSED]),
            "cancelled_jobs": len([j for j in self.scheduled_jobs.values() if j.status == ScheduleStatus.CANCELLED]),
            "total_executions": sum(j.run_count for j in self.scheduled_jobs.values()),
            "next_jobs": []
        }

        # Get next scheduled jobs
        for job in self.scheduled_jobs.values():
            if job.status == ScheduleStatus.ACTIVE and job.next_run:
                stats["next_jobs"].append({
                    "schedule_id": job.schedule_id,
                    "name": job.name,
                    "next_run": job.next_run.isoformat(),
                    "schedule_type": job.schedule_type.value
                })

        # Sort by next run time
        stats["next_jobs"].sort(key=lambda x: x["next_run"])
        stats["next_jobs"] = stats["next_jobs"][:10]  # Limit to next 10 jobs

        return stats

    # Event handlers
    async def _job_executed(self, event):
        """Handle job execution events."""
        logger.debug(f"Job {event.job_id} executed successfully")

    async def _job_error(self, event):
        """Handle job error events."""
        logger.error(f"Job {event.job_id} failed: {event.exception}")

    async def _job_missed(self, event):
        """Handle missed job events."""
        logger.warning(f"Job {event.job_id} was missed")


# Utility functions for common scheduling patterns
def create_daily_schedule(hour: int = 0, minute: int = 0) -> Dict[str, Any]:
    """Create a daily schedule configuration."""
    return {
        "hour": hour,
        "minute": minute,
        "second": 0
    }


def create_weekly_schedule(day_of_week: int = 0, hour: int = 0, minute: int = 0) -> Dict[str, Any]:
    """Create a weekly schedule configuration."""