import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, 'r') as f:
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif file_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes"""
    return Path(file_path).stat().st_size


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage"""
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    return filename.strip('_')


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss': 0, 'vms': 0, 'percent': 0}


class ProgressTracker:
    """Simple progress tracking utility"""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.last_update = self.start_time

    def update(self, increment: int = 1, message: str = None):
        """Update progress"""
        self.current += increment
        now = datetime.now()

        if (now - self.last_update).seconds >= 5 or self.current >= self.total:
            self._log_progress(message)
            self.last_update = now

    def _log_progress(self, message: str = None):
        """Log current progress"""
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time

        if self.current > 0:
            rate = self.current / elapsed.total_seconds()
            eta = timedelta(seconds=(self.total - self.current) / rate)
            eta_str = format_duration(eta.total_seconds())
        else:
            rate = 0
            eta_str = "N/A"

        progress_msg = (
            f"{self.description}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - Rate: {rate:.1f}/s - ETA: {eta_str}"
        )

        if message:
            progress_msg += f" - {message}"

        logger.info(progress_msg)

