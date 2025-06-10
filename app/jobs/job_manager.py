# app/jobs/job_manager.py
"""
Job Manager for REST API
Handles job lifecycle management
"""
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Job model"""

    def __init__(self, job_id: str, job_type: str, config: Dict, priority: str,
                 scheduled_time: Optional[datetime] = None, tags: List[str] = None,
                 metadata: Dict[str, Any] = None, user_id: str = None):
        self.job_id = job_id
        self.job_type = job_type
        self.config = config
        self.priority = priority
        self.scheduled_time = scheduled_time
        self.tags = tags or []
        self.metadata = metadata or {}
        self.user_id = user_id
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.message = None
        self.error = None
        self.result = None

    def dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            'job_id': self.job_id,
            'job_type': self.job_type,
            'status': self.status,
            'priority': self.priority,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'progress': self.progress,
            'message': self.message,
            'error': self.error,
            'result': self.result,
            'metadata': self.metadata
        }


class JobManager:
    """Manages job lifecycle"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.job_logs: Dict[str, List[str]] = {}

    async def initialize(self):
        """Initialize job manager"""
        logger.info("Job manager initialized")

    async def cleanup(self):
        """Cleanup job manager"""
        logger.info("Job manager cleaned up")

    async def create_job(self, job_id: str, job_type: str, config: Dict,
                         priority: str, scheduled_time: Optional[datetime] = None,
                         tags: List[str] = None, metadata: Dict[str, Any] = None,
                         user_id: str = None) -> Job:
        """Create a new job"""
        job = Job(job_id, job_type, config, priority, scheduled_time,
                  tags, metadata, user_id)
        self.jobs[job_id] = job
        self.job_logs[job_id] = []
        logger.info(f"Created job: {job_id}")
        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    async def list_jobs(self, filters: Dict = None, page: int = 1,
                        page_size: int = 20) -> tuple[List[Job], int]:
        """List jobs with filtering and pagination"""
        jobs = list(self.jobs.values())

        # Apply filters
        if filters:
            if 'status' in filters:
                jobs = [j for j in jobs if j.status == filters['status']]
            if 'job_type' in filters:
                jobs = [j for j in jobs if j.job_type == filters['job_type']]
            if 'user_id' in filters:
                jobs = [j for j in jobs if j.user_id == filters['user_id']]

        total = len(jobs)

        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        jobs = jobs[start:end]

        return jobs, total

    async def update_job_status(self, job_id: str, status: JobStatus):
        """Update job status"""
        if job_id in self.jobs:
            self.jobs[job_id].status = status
            if status == JobStatus.RUNNING:
                self.jobs[job_id].started_at = datetime.utcnow()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                self.jobs[job_id].completed_at = datetime.utcnow()

    async def update_job_progress(self, job_id: str, progress: float, message: str = None):
        """Update job progress"""
        if job_id in self.jobs:
            self.jobs[job_id].progress = progress
            if message:
                self.jobs[job_id].message = message
                self.job_logs[job_id].append(f"{datetime.utcnow()}: {message}")

    async def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Complete a job with results"""
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.COMPLETED
            self.jobs[job_id].completed_at = datetime.utcnow()
            self.jobs[job_id].result = result
            self.jobs[job_id].progress = 100.0

    async def fail_job(self, job_id: str, error: str):
        """Fail a job with error message"""
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.FAILED
            self.jobs[job_id].completed_at = datetime.utcnow()
            self.jobs[job_id].error = error
            self.job_logs[job_id].append(f"{datetime.utcnow()}: ERROR: {error}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id in self.jobs and self.jobs[job_id].status in [JobStatus.PENDING, JobStatus.RUNNING]:
            self.jobs[job_id].status = JobStatus.CANCELLED
            self.jobs[job_id].completed_at = datetime.utcnow()
            return True
        return False

    async def get_job_logs(self, job_id: str, lines: int = 100) -> Optional[List[str]]:
        """Get job logs"""
        if job_id in self.job_logs:
            return self.job_logs[job_id][-lines:]
        return None

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_jobs = len(self.jobs)
        active_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING])
        completed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED])
        failed_jobs = len([j for j in self.jobs.values() if j.status == JobStatus.FAILED])

        return {
            'total_jobs': total_jobs,
            'active_jobs': active_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'system_load': 0.0,  # Placeholder
            'memory_usage': 0.0,  # Placeholder
            'uptime': 'N/A'  # Placeholder
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return self.get_system_stats()
