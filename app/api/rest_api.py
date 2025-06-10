"""
FastAPI REST API for Synthetic Data Generation Platform
Provides endpoints for job management, status checking, and configuration.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from app.jobs.job_manager import JobManager, JobStatus
from app.utils.config_manager import ConfigManager
from app.utils.exceptions import ConfigurationError, GenerationError
from app.core.data_generator import OptimizedDataGenerator
from app.output.audit_logger import AuditLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Global instances
job_manager = JobManager()
config_manager = ConfigManager()
audit_logger = AuditLogger()


class JobType(str, Enum):
    """Types of generation jobs."""
    GENERATE = "generate"
    VALIDATE = "validate"
    MASK = "mask"
    EXPORT = "export"


class Priority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Pydantic Models
class GenerationConfig(BaseModel):
    """Configuration for data generation."""
    tables: List[Dict[str, Any]]
    locale: str = "en_US"
    rows: int = Field(default=1000, ge=1, le=10000000)
    output_format: str = Field(default="csv", regex="^(csv|json|parquet|sql)$")
    output_path: Optional[str] = None
    mask_sensitive: bool = False
    export_destination: Optional[str] = None

    @validator('tables')
    def validate_tables(cls, v):
        if not v:
            raise ValueError("At least one table must be specified")
        return v


class JobRequest(BaseModel):
    """Request model for creating a new job."""
    job_type: JobType
    config: GenerationConfig
    priority: Priority = Priority.NORMAL
    scheduled_time: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    """Response model for job information."""
    job_id: str
    job_type: JobType
    status: JobStatus
    priority: Priority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobListResponse(BaseModel):
    """Response model for job listing."""
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int


class SystemStats(BaseModel):
    """System statistics and health information."""
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    system_load: float
    memory_usage: float
    uptime: str


# FastAPI App
app = FastAPI(
    title="Synthetic Data Generation Platform API",
    description="REST API for managing synthetic data generation jobs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Basic authentication dependency."""
    if not credentials:
        return None

    # Implement your authentication logic here
    # For now, just return a mock user
    return {"user_id": "anonymous", "role": "user"}


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# System statistics
@app.get("/stats", response_model=SystemStats, tags=["System"])
async def get_system_stats():
    """Get system statistics and health information."""
    stats = job_manager.get_system_stats()
    return SystemStats(**stats)


# Job Management Endpoints
@app.post("/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED, tags=["Jobs"])
async def create_job(
    job_request: JobRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a new data generation job."""
    try:
        # Validate configuration
        config_manager.validate_config(job_request.config.dict())

        # Create job
        job_id = str(uuid.uuid4())
        job = await job_manager.create_job(
            job_id=job_id,
            job_type=job_request.job_type,
            config=job_request.config.dict(),
            priority=job_request.priority,
            scheduled_time=job_request.scheduled_time,
            tags=job_request.tags,
            metadata=job_request.metadata,
            user_id=current_user.get("user_id") if current_user else None
        )

        # Log job creation
        await audit_logger.log_job_event(
            job_id=job_id,
            event_type="JOB_CREATED",
            user_id=current_user.get("user_id") if current_user else None,
            metadata={"job_type": job_request.job_type, "priority": job_request.priority}
        )

        # Schedule job execution
        background_tasks.add_task(execute_job, job_id)

        return JobResponse(**job.dict())

    except ConfigurationError as e:
        logger.error(f"Configuration error creating job: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job(job_id: str):
    """Get job details by ID."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(**job.dict())


@app.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(
    status_filter: Optional[JobStatus] = None,
    job_type_filter: Optional[JobType] = None,
    page: int = Field(default=1, ge=1),
    page_size: int = Field(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """List jobs with optional filtering and pagination."""
    try:
        filters = {}
        if status_filter:
            filters['status'] = status_filter
        if job_type_filter:
            filters['job_type'] = job_type_filter
        if current_user:
            filters['user_id'] = current_user.get("user_id")

        jobs, total = await job_manager.list_jobs(
            filters=filters,
            page=page,
            page_size=page_size
        )

        return JobListResponse(
            jobs=[JobResponse(**job.dict()) for job in jobs],
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def cancel_job(job_id: str, current_user: dict = Depends(get_current_user)):
    """Cancel a running job."""
    try:
        success = await job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

        # Log job cancellation
        await audit_logger.log_job_event(
            job_id=job_id,
            event_type="JOB_CANCELLED",
            user_id=current_user.get("user_id") if current_user else None
        )

        return {"message": "Job cancelled successfully"}

    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/jobs/{job_id}/logs", tags=["Jobs"])
async def get_job_logs(job_id: str, lines: int = Field(default=100, ge=1, le=10000)):
    """Get job execution logs."""
    logs = await job_manager.get_job_logs(job_id, lines)
    if logs is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"job_id": job_id, "logs": logs}


@app.get("/jobs/{job_id}/download", tags=["Jobs"])
async def download_job_result(job_id: str):
    """Download job result file."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job is not completed")

    if not job.result or 'output_path' not in job.result:
        raise HTTPException(status_code=404, detail="Job result file not found")

    file_path = Path(job.result['output_path'])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type='application/octet-stream'
    )


# Configuration Management Endpoints
@app.get("/config/validate", tags=["Configuration"])
async def validate_config(config: GenerationConfig):
    """Validate a generation configuration."""
    try:
        config_manager.validate_config(config.dict())
        return {"valid": True, "message": "Configuration is valid"}
    except ConfigurationError as e:
        return {"valid": False, "errors": [str(e)]}


@app.post("/config/upload", tags=["Configuration"])
async def upload_config(file: UploadFile = File(...)):
    """Upload and validate a configuration file."""
    if not file.filename.endswith(('.json', '.yaml', '.yml')):
        raise HTTPException(status_code=400, detail="Only JSON and YAML files are supported")

    try:
        content = await file.read()
        if file.filename.endswith('.json'):
            config_data = json.loads(content.decode('utf-8'))
        else:
            import yaml
            config_data = yaml.safe_load(content.decode('utf-8'))

        # Validate configuration
        config_manager.validate_config(config_data)

        return {
            "message": "Configuration uploaded and validated successfully",
            "config": config_data
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {str(e)}")
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=f"Configuration validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error uploading config: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Schema Management Endpoints
@app.get("/schemas", tags=["Schema"])
async def list_schemas():
    """List available schema templates."""
    try:
        schemas = config_manager.list_schema_templates()
        return {"schemas": schemas}
    except Exception as e:
        logger.error(f"Error listing schemas: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/schemas/{schema_name}", tags=["Schema"])
async def get_schema(schema_name: str):
    """Get a specific schema template."""
    try:
        schema = config_manager.get_schema_template(schema_name)
        if not schema:
            raise HTTPException(status_code=404, detail="Schema not found")
        return {"schema": schema}
    except Exception as e:
        logger.error(f"Error getting schema {schema_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Background task for job execution
async def execute_job(job_id: str):
    """Execute a data generation job in the background."""
    try:
        # Update job status to running
        await job_manager.update_job_status(job_id, JobStatus.RUNNING)

        # Get job details
        job = await job_manager.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        # Initialize data generator
        generator = OptimizedDataGenerator(
            config=job.config,
            audit_logger=audit_logger
        )

        # Progress callback
        async def progress_callback(progress: float, message: str = None):
            await job_manager.update_job_progress(job_id, progress, message)

        # Execute generation
        result = await generator.generate_async(
            progress_callback=progress_callback
        )

        # Update job with results
        await job_manager.complete_job(job_id, result)

        # Log completion
        await audit_logger.log_job_event(
            job_id=job_id,
            event_type="JOB_COMPLETED",
            metadata={"rows_generated": result.get("total_rows", 0)}
        )

    except Exception as e:
        logger.error(f"Error executing job {job_id}: {str(e)}")
        await job_manager.fail_job(job_id, str(e))

        # Log failure
        await audit_logger.log_job_event(
            job_id=job_id,
            event_type="JOB_FAILED",
            metadata={"error": str(e)}
        )


# WebSocket endpoint for real-time job updates
@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket, job_id: str):
    """WebSocket endpoint for real-time job status updates."""
    await websocket.accept()

    try:
        while True:
            # Get current job status
            job = await job_manager.get_job(job_id)
            if job:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "message": job.message,
                    "timestamp": datetime.utcnow().isoformat()
                })

            # Wait before next update
            await asyncio.sleep(2)

            # Break if job is completed or failed
            if job and job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                break

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}")
    finally:
        await websocket.close()


# Batch operations
@app.post("/jobs/batch", tags=["Jobs"])
async def create_batch_jobs(
    job_requests: List[JobRequest],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create multiple jobs in batch."""
    if len(job_requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 10 jobs")

    created_jobs = []

    for job_request in job_requests:
        try:
            # Validate configuration
            config_manager.validate_config(job_request.config.dict())

            # Create job
            job_id = str(uuid.uuid4())
            job = await job_manager.create_job(
                job_id=job_id,
                job_type=job_request.job_type,
                config=job_request.config.dict(),
                priority=job_request.priority,
                scheduled_time=job_request.scheduled_time,
                tags=job_request.tags,
                metadata=job_request.metadata,
                user_id=current_user.get("user_id") if current_user else None
            )

            created_jobs.append(JobResponse(**job.dict()))

            # Schedule job execution
            background_tasks.add_task(execute_job, job_id)

        except Exception as e:
            logger.error(f"Error creating batch job: {str(e)}")
            # Continue with other jobs instead of failing the entire batch

    return {"created_jobs": created_jobs, "total_created": len(created_jobs)}


# Metrics and monitoring
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get system metrics for monitoring."""
    try:
        metrics = await job_manager.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers
@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Configuration Error", "detail": str(exc)}
    )


@app.exception_handler(GenerationError)
async def generation_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Generation Error", "detail": str(exc)}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Synthetic Data Generation Platform API")
    await job_manager.initialize()
    await audit_logger.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Synthetic Data Generation Platform API")
    await job_manager.cleanup()
    await audit_logger.cleanup()


# Main function for running the server
def main():
    """Main function to run the FastAPI server."""
    uvicorn.run(
        "rest_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()