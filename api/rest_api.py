"""
FastAPI-based REST API for Synthetic Data Generation Platform

Provides endpoints for job management, generation triggering, and status monitoring.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import logging

from .core.data_generator import OptimizedDataGenerator
from .core.masking_engine import MaskingEngine, MaskingRule, SensitivityLevel, MaskingStrategy
from .output.exporter import DataExporter
from .utils.config_manager import ConfigManager
from .utils.exceptions import GenerationError, ValidationError
from .jobs.job_manager import JobManager, JobStatus, JobType

logger = logging.getLogger(__name__)

# Pydantic Models for API
class GenerationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ColumnConfig(BaseModel):
    name: str
    type: str
    constraint: Optional[List[str]] = None
    rule: Optional[Any] = None
    sensitivity: Optional[str] = None


class ForeignKeyConfig(BaseModel):
    parent_table: str
    parent_column: str
    child_column: str


class TableConfig(BaseModel):
    table_name: str
    columns: List[ColumnConfig]
    foreign_keys: Optional[List[ForeignKeyConfig]] = None
    rows: Optional[int] = None


class GenerationRequest(BaseModel):
    tables: List[TableConfig]
    locale: str = "en_US"
    rows: int = Field(gt=0, le=1000000, description="Number of rows to generate")
    export_format: str = Field(default="json", regex="^(csv|json|parquet|sql)$")
    export_destination: Optional[str] = None
    apply_masking: bool = False
    masking_rules: Optional[List[Dict[str, Any]]] = None
    job_name: Optional[str] = None

    @validator('rows')
    def validate_rows(cls, v):
        if v <= 0:
            raise ValueError('Rows must be positive')
        return v


class JobResponse(BaseModel):
    job_id: str
    status: GenerationStatus
    message: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    dependencies: Dict[str, str]


class MaskingRuleRequest(BaseModel):
    field_name: str
    sensitivity_level: str
    strategy: str
    strategy_params: Optional[Dict[str, Any]] = None
    deterministic: bool = True


# Global job manager instance
job_manager = JobManager()

# Security
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple bearer token authentication - enhance as needed"""
    if not credentials:
        return None

    # TODO: Implement proper JWT token validation
    # For now, just check if token exists and is not empty
    if credentials.credentials:
        return {"user_id": "api_user", "roles": ["user"]}

    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info("Starting Synthetic Data Generation API")
    yield
    # Shutdown
    logger.info("Shutting down Synthetic Data Generation API")
    await job_manager.shutdown()


# Initialize FastAPI app
app = FastAPI(
    title="Synthetic Data Generation Platform API",
    description="REST API for generating synthetic data with masking and relationship preservation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        dependencies={
            "faker": "available",
            "pandas": "available",
            "fastapi": "available"
        }
    )


@app.post("/generate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_data(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Start a data generation job
    """
    try:
        # Create job
        job_id = str(uuid.uuid4())
        job_name = request.job_name or f"generation_job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Register job
        job = await job_manager.create_job(
            job_id=job_id,
            job_type=JobType.GENERATION,
            name=job_name,
            config=request.dict(),
            created_by=user.get("user_id", "anonymous") if user else "anonymous"
        )

        # Start background task
        background_tasks.add_task(
            _execute_generation_job,
            job_id,
            request
        )

        return JobResponse(
            job_id=job_id,
            status=GenerationStatus.PENDING,
            message="Generation job queued successfully",
            created_at=job.created_at
        )

    except Exception as e:
        logger.error(f"Error creating generation job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create generation job: {str(e)}"
        )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, user: dict = Depends(get_current_user)):
    """Get job status and details"""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        return JobResponse(
            job_id=job.job_id,
            status=GenerationStatus(job.status.value.lower()),
            message=job.message or "",
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            progress=job.progress,
            error=job.error_message,
            result=job.result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status"
        )


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    status_filter: Optional[str] = Query(None),
    user: dict = Depends(get_current_user)
):
    """List jobs with pagination and filtering"""
    try:
        jobs = await job_manager.list_jobs(
            page=page,
            page_size=page_size,
            status_filter=status_filter,
            user_id=user.get("user_id") if user else None
        )

        job_responses = []
        for job in jobs.get("jobs", []):
            job_responses.append(JobResponse(
                job_id=job.job_id,
                status=GenerationStatus(job.status.value.lower()),
                message=job.message or "",
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                progress=job.progress,
                error=job.error_message,
                result=job.result
            ))

        return JobListResponse(
            jobs=job_responses,
            total=jobs.get("total", 0),
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list jobs"
        )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, user: dict = Depends(get_current_user)):
    """Cancel a running job"""
    try:
        success = await job_manager.cancel_job(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or cannot be cancelled"
            )

        return {"message": "Job cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job"
        )


@app.get("/jobs/{job_id}/download")
async def download_job_result(job_id: str, user: dict = Depends(get_current_user)):
    """Download generated data file"""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job not completed yet"
            )

        file_path = job.result.get("output_file")
        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No output file available"
            )

        return FileResponse(
            path=file_path,
            filename=f"synthetic_data_{job_id}.{job.result.get('format', 'json')}",
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading job result {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download job result"
        )


@app.post("/validate-config")
async def validate_config(request: GenerationRequest):
    """Validate generation configuration without running the job"""
    try:
        # Basic validation through Pydantic model
        errors = []

        # Additional custom validation
        table_names = {table.table_name for table in request.tables}

        for table in request.tables:
            # Check for duplicate column names
            column_names = [col.name for col in table.columns]
            if len(column_names) != len(set(column_names)):
                errors.append(f"Duplicate column names in table '{table.table_name}'")

            # Check foreign key references
            if table.foreign_keys:
                for fk in table.foreign_keys:
                    if fk.parent_table not in table_names:
                        errors.append(f"Foreign key references unknown table '{fk.parent_table}'")

        if errors:
            return {"valid": False, "errors": errors}

        return {"valid": True, "message": "Configuration is valid"}

    except Exception as e:
        logger.error(f"Error validating config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration: {str(e)}"
        )


@app.post("/masking/test")
async def test_masking(
    field_name: str,
    value: str,
    masking_rule: MaskingRuleRequest
):
    """Test masking rule on a sample value"""
    try:
        masking_engine = MaskingEngine()

        rule = MaskingRule(
            field_name=masking_rule.field_name,
            sensitivity_level=SensitivityLevel(masking_rule.sensitivity_level),
            strategy=MaskingStrategy(masking_rule.strategy),
            strategy_params=masking_rule.strategy_params,
            deterministic=masking_rule.deterministic
        )

        masking_engine.add_masking_rule(rule)
        masked_value = masking_engine.mask_value(field_name, value)

        return {
            "original_value": value,
            "masked_value": masked_value,
            "rule": masking_rule.dict()
        }

    except Exception as e:
        logger.error(f"Error testing masking rule: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error testing masking rule: {str(e)}"
        )


@app.get("/stats")
async def get_platform_stats(user: dict = Depends(get_current_user)):
    """Get platform usage statistics"""
    try:
        stats = await job_manager.get_stats()
        return {
            "total_jobs": stats.get("total_jobs", 0),
            "jobs_by_status": stats.get("jobs_by_status", {}),
            "jobs_today": stats.get("jobs_today", 0),
            "total_rows_generated": stats.get("total_rows_generated", 0),
            "average_job_duration": stats.get("average_job_duration", 0),
            "most_used_formats": stats.get("most_used_formats", {}),
        }

    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve platform statistics"
        )


# Background task functions

async def _execute_generation_job(job_id: str, request: GenerationRequest):
    """Execute data generation job in background"""
    try:
        # Update job status to running
        await job_manager.update_job_status(job_id, JobStatus.RUNNING, "Starting data generation")

        # Initialize components
        config_manager = ConfigManager()
        data_generator = OptimizedDataGenerator()
        masking_engine = MaskingEngine() if request.apply_masking else None
        exporter = DataExporter()

        # Set up masking rules if needed
        if masking_engine and request.masking_rules:
            masking_engine.add_masking_rules_from_config(request.masking_rules)

        # Progress tracking
        total_tables = len(request.tables)
        completed_tables = 0

        # Generate data for each table
        all_generated_data = {}

        for table_config in request.tables:
            await job_manager.update_job_progress(
                job_id,
                (completed_tables / total_tables) * 0.8,  # 80% for generation
                f"Generating data for table '{table_config.table_name}'"
            )

            # Convert table config to generator format
            table_schema = {
                "table_name": table_config.table_name,
                "columns": [col.dict() for col in table_config.columns],
                "foreign_keys": [fk.dict() for fk in table_config.foreign_keys] if table_config.foreign_keys else []
            }

            # Generate data
            rows_to_generate = table_config.rows or request.rows
            generated_data = await _generate_table_data(
                data_generator,
                table_schema,
                rows_to_generate,
                request.locale
            )

            # Apply masking if enabled
            if masking_engine:
                generated_data = masking_engine.mask_dataset(generated_data)

            all_generated_data[table_config.table_name] = generated_data
            completed_tables += 1

        # Export data
        await job_manager.update_job_progress(
            job_id,
            0.9,
            "Exporting generated data"
        )

        output_file = await _export_generated_data(
            exporter,
            all_generated_data,
            request.export_format,
            request.export_destination,
            job_id
        )

        # Complete job
        result = {
            "output_file": output_file,
            "format": request.export_format,
            "total_rows": sum(len(data) for data in all_generated_data.values()),
            "tables_generated": list(all_generated_data.keys()),
            "masking_applied": request.apply_masking
        }

        await job_manager.complete_job(
            job_id,
            "Data generation completed successfully",
            result
        )

    except Exception as e:
        logger.error(f"Error in generation job {job_id}: {str(e)}")
        await job_manager.fail_job(job_id, str(e))


async def _generate_table_data(generator, table_schema, rows, locale):
    """Generate data for a single table"""
    # This would integrate with your existing OptimizedDataGenerator
    # For now, simplified implementation
    return await asyncio.get_event_loop().run_in_executor(
        None,
        generator.generate_table_data,
        table_schema,
        rows,
        locale
    )


async def _export_generated_data(exporter, data, format_type, destination, job_id):
    """Export generated data to specified format and destination"""
    output_filename = f"synthetic_data_{job_id}.{format_type}"

    return await asyncio.get_event_loop().run_in_executor(
        None,
        exporter.export_data,
        data,
        format_type,
        destination or f"./output/{output_filename}"
    )


# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"}
    )


@app.exception_handler(GenerationError)
async def generation_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "generation_error"}
    )


# Main application runner
def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """Factory function to create configured FastAPI app"""
    if config:
        # Apply configuration to app
        pass

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server"""
    uvicorn.run(
        "synthetic_data_platform.api.rest_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()