"""
Job Scheduler for Synthetic Data Generation Platform

Supports cron-based scheduling, job queuing, and automated generation workflows.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import uuid

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.job import Job
from apscheduler.events import (
    EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED,
    EVENT_JOB_ADDED, EVENT_JOB_REMOVED, JobExecutionEvent
)

from .job_manager import JobManager, JobStatus, JobType
from ..core.data_generator import OptimizedDataGenerator
from ..core.masking_engine import MaskingEngine
from ..output.exporter import DataExporter
from ..utils.config_manager import ConfigManager
from ..utils.exceptions import SchedulerError

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduling triggers"""
    CRON = "cron"
    INTERVAL = "interval"
    ONE_TIME = "one_time"


class ScheduleStatus(Enum):
    """Schedule status"""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    EXPIRED = "expired"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled job"""
    schedule_id: str
    name: str
    description: str
    schedule_type: ScheduleType
    trigger_config: Dict[str, Any]  # Cron expression, interval config, etc.
    generation_config: Dict[str, Any]  # Data generation parameters
    status: ScheduleStatus
    created_at: datetime
    created_by: str
    next_run_time: Optional[datetime] = None
    last_run_time: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    retry_config: Optional[Dict[str, Any]] = None
    notification_config: Optional[Dict[str, Any]] = None


class DataGenerationScheduler:
    """Main scheduler class for managing automated data generation jobs"""

    def __init__(self, job_manager: JobManager, config: Dict[str, Any] = None):
        self.job_manager = job_manager
        self.config = config or {}
        self.scheduler = AsyncIOScheduler(
            timezone=self.config.get('timezone', 'UTC'),
            job_defaults={
                'coalesce': True,
                'max_instances': self.config.get('max_concurrent_jobs', 3),
                'misfire_grace_time': self.config.get('misfire_grace_time', 300)  # 5 minutes
            }
        )

        self.schedules: Dict[str, ScheduleConfig] = {}
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_worker_threads', 4)
        )

        # Set up event listeners
        self._setup_event_listeners()

        self.is_running = False

    def _setup_event_listeners(self):
        """Set up APScheduler event listeners"""
        self.scheduler.add_listener(
            self._on_job_executed,
            EVENT_JOB_EXECUTED
        )

        self.scheduler.add_listener(
            self._on_job_error,
            EVENT_JOB_ERROR
        )

        self.scheduler.add_listener(
            self._on_job_missed,
            EVENT_JOB_MISSED
        )

        self.scheduler.add_listener(
            self._on_job_added,
            EVENT_JOB_ADDED
        )

        self.scheduler.add_listener(
            self._on_job_removed,
            EVENT_JOB_REMOVED
        )

    async def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        try:
            self.scheduler.start()
            self.is_running = True
            logger.info("Data generation scheduler started successfully")

            # Load existing schedules from storage
            await self._load_schedules()

        except Exception as e:
            logger.error(f"Failed to start scheduler: {str(e)}")
            raise SchedulerError(f"Failed to start scheduler: {str(e)}")

    async def shutdown(self):
        """Shutdown the scheduler gracefully"""
        if not self.is_running:
            return

        try:
            logger.info("Shutting down data generation scheduler")

            # Save current schedules
            await self._save_schedules()

            # Shutdown scheduler
            self.scheduler.shutdown(wait=True)

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)

            self.is_running = False
            logger.info("Scheduler shutdown completed")

        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {str(e)}")

    async def create_schedule(
            self,
            name: str,
            description: str,
            schedule_type: ScheduleType,
            trigger_config: Dict[str, Any],
            generation_config: Dict[str, Any],
            created_by: str = "system",
            max_runs: Optional[int] = None,
            retry_config: Optional[Dict[str, Any]] = None,
            notification_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new scheduled job"""
        try:
            schedule_id = str(uuid.uuid4())

            # Validate trigger configuration
            trigger = self._create_trigger(schedule_type, trigger_config)

            # Create schedule configuration
            schedule_config = ScheduleConfig(
                schedule_id=schedule_id,
                name=name,
                description=description,
                schedule_type=schedule_type,
                trigger_config=trigger_config,
                generation_config=generation_config,
                status=ScheduleStatus.ACTIVE,
                created_at=datetime.utcnow(),
                created_by=created_by,
                max_runs=max_runs,
                retry_config=retry_config or {"max_retries": 3, "retry_delay": 60},
                notification_config=notification_config
            )

            # Add job to scheduler
            job = self.scheduler.add_job(
                func=self._execute_scheduled_generation,
                trigger=trigger,
                args=[schedule_id],
                id=schedule_id,
                name=name,
                replace_existing=True
            )

            # Store schedule configuration
            self.schedules[schedule_id] = schedule_config
            schedule_config.next_run_time = job.next_run_time

            logger.info(f"Created schedule '{name}' with ID: {schedule_id}")

            # Save to persistent storage
            await self._save_schedules()

            return schedule_id

        except Exception as e:
            logger.error(f"Failed to create schedule: {str(e)}")
            raise SchedulerError(f"Failed to create schedule: {str(e)}")

    def _create_trigger(self, schedule_type: ScheduleType, config: Dict[str, Any]):
        """Create APScheduler trigger from configuration"""
        if schedule_type == ScheduleType.CRON:
            return CronTrigger(**config)
        elif schedule_type == ScheduleType.INTERVAL:
            return IntervalTrigger(**config)
        elif schedule_type == ScheduleType.ONE_TIME:
            return DateTrigger(run_date=config.get('run_date'))
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")

    async def update_schedule(
            self,
            schedule_id: str,
            updates: Dict[str, Any]
    ) -> bool:
        """Update an existing schedule"""
        try:
            if schedule_id not in self.schedules:
                raise ValueError(f"Schedule {schedule_id} not found")

            schedule = self.schedules[schedule_id]

            # Update allowed fields
            for field, value in updates.items():
                if hasattr(schedule, field) and field not in ['schedule_id', 'created_at']:
                    setattr(schedule, field, value)

            # If trigger config changed, update the scheduled job
            if 'trigger_config' in updates:
                trigger = self._create_trigger(schedule.schedule_type, schedule.trigger_config)
                self.scheduler.modify_job(schedule_id, trigger=trigger)

                # Update next run time
                job = self.scheduler.get_job(schedule_id)
                if job:
                    schedule.next_run_time = job.next_run_time

            await self._save_schedules()
            logger.info(f"Updated schedule {schedule_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to update schedule {schedule_id}: {str(e)}")
            raise SchedulerError(f"Failed to update schedule: {str(e)}")

    async def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule"""
        try:
            if schedule_id not in self.schedules:
                return False

            # Remove from scheduler
            self.scheduler.remove_job(schedule_id)

            # Remove from memory
            del self.schedules[schedule_id]

            # Save changes
            await self._save_schedules()

            logger.info(f"Deleted schedule {schedule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete schedule {schedule_id}: {str(e)}")
            return False

    async def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule"""
        try:
            if schedule_id not in self.schedules:
                return False

            self.scheduler.pause_job(schedule_id)
            self.schedules[schedule_id].status = ScheduleStatus.PAUSED

            await self._save_schedules()
            logger.info(f"Paused schedule {schedule_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to pause schedule {schedule_id}: {str(e)}")
            return False

    async def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule"""
        try:
            if schedule_id not in self.schedules:
                return False

            self.scheduler.resume_job(schedule_id)
            self.schedules[schedule_id].status = ScheduleStatus.ACTIVE

            # Update next run time
            job = self.scheduler.get_job(schedule_id)
            if job:
                self.schedules[schedule_id].next_run_time = job.next_run_time

            await self._save_schedules()
            logger.info(f"Resumed schedule {schedule_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to resume schedule {schedule_id}: {str(e)}")
            return False

    async def trigger_schedule_now(self, schedule_id: str) -> str:
        """Trigger a schedule immediately (one-time execution)"""
        try:
            if schedule_id not in self.schedules:
                raise ValueError(f"Schedule {schedule_id} not found")

            # Execute the generation job directly
            job_id = await self._execute_scheduled_generation(schedule_id, manual_trigger=True)

            logger.info(f"Manually triggered schedule {schedule_id}, job ID: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to trigger schedule {schedule_id}: {str(e)}")
            raise SchedulerError(f"Failed to trigger schedule: {str(e)}")

    async def get_schedule(self, schedule_id: str) -> Optional[ScheduleConfig]:
        """Get schedule configuration"""
        return self.schedules.get(schedule_id)

    async def list_schedules(
            self,
            status_filter: Optional[ScheduleStatus] = None,
            created_by: Optional[str] = None
    ) -> List[ScheduleConfig]:
        """List all schedules with optional filtering"""
        schedules = list(self.schedules.values())

        if status_filter:
            schedules = [s for s in schedules if s.status == status_filter]

        if created_by:
            schedules = [s for s in schedules if s.created_by == created_by]

        return schedules

    async def get_schedule_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        total_schedules = len(self.schedules)
        active_schedules = len([s for s in self.schedules.values() if s.status == ScheduleStatus.ACTIVE])
        paused_schedules = len([s for s in self.schedules.values() if s.status == ScheduleStatus.PAUSED])

        # Get next scheduled job
        next_job = None
        next_run_time = None
        for job in self.scheduler.get_jobs():
            if not next_run_time or (job.next_run_time and job.next_run_time < next_run_time):
                next_run_time = job.next_run_time
                next_job = job.name

        return {
            "total_schedules": total_schedules,
            "active_schedules": active_schedules,
            "paused_schedules": paused_schedules,
            "disabled_schedules": total_schedules - active_schedules - paused_schedules,
            "next_job": next_job,
            "next_run_time": next_run_time.isoformat() if next_run_time else None,
            "scheduler_running": self.is_running
        }

    async def _execute_scheduled_generation(
            self,
            schedule_id: str,
            manual_trigger: bool = False
    ) -> str:
        """Execute the actual data generation for a scheduled job"""
        try:
            if schedule_id not in self.schedules:
                raise ValueError(f"Schedule {schedule_id} not found")

            schedule = self.schedules[schedule_id]

            logger.info(f"Executing scheduled generation: {schedule.name} (ID: {schedule_id})")

            # Check if schedule has reached max runs
            if schedule.max_runs and schedule.run_count >= schedule.max_runs:
                logger.info(f"Schedule {schedule_id} has reached max runs limit")
                schedule.status = ScheduleStatus.EXPIRED
                return None

            # Create a job in the job manager
            job_id = str(uuid.uuid4())
            job_name = f"{schedule.name}_scheduled_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            job = await self.job_manager.create_job(
                job_id=job_id,
                job_type=JobType.SCHEDULED_GENERATION,
                name=job_name,
                config=schedule.generation_config,
                created_by=schedule.created_by,
                parent_schedule_id=schedule_id
            )

            # Execute generation in thread pool to avoid blocking scheduler
            future = self.thread_pool.submit(
                self._run_generation_with_retry,
                job_id,
                schedule.generation_config,
                schedule.retry_config
            )

            # Update schedule stats
            schedule.run_count += 1
            schedule.last_run_time = datetime.utcnow()

            # Update next run time
            if not manual_trigger:
                scheduled_job = self.scheduler.get_job(schedule_id)
                if scheduled_job:
                    schedule.next_run_time = scheduled_job.next_run_time

            await self._save_schedules()

            return job_id

        except Exception as e:
            logger.error(f"Failed to execute scheduled generation {schedule_id}: {str(e)}")

            # Send notification if configured
            if schedule_id in self.schedules:
                await self._send_failure_notification(
                    self.schedules[schedule_id],
                    str(e)
                )

            raise

    def _run_generation_with_retry(
            self,
            job_id: str,
            generation_config: Dict[str, Any],
            retry_config: Dict[str, Any]
    ):
        """Run data generation with retry logic"""
        max_retries = retry_config.get('max_retries', 3)
        retry_delay = retry_config.get('retry_delay', 60)

        for attempt in range(max_retries + 1):
            try:
                # Run the actual generation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(
                        self._run_data_generation(job_id, generation_config)
                    )
                    return result
                finally:
                    loop.close()

            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed for job {job_id}: {str(e)}")

                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    asyncio.sleep(retry_delay)
                else:
                    # Mark job as failed
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        loop.run_until_complete(
                            self.job_manager.fail_job(job_id, f"Failed after {max_retries} retries: {str(e)}")
                        )
                    finally:
                        loop.close()

                    raise

    async def _run_data_generation(
            self