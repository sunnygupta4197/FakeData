# app/output/audit_logger.py
"""
Audit Logger for Synthetic Data Platform
Tracks all operations for compliance and monitoring
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLogger:
    """Audit logging for compliance and monitoring"""

    def __init__(self, log_directory: str = "./audit_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.log_directory / "audit.jsonl"

    async def initialize(self):
        """Initialize audit logger"""
        logger.info("Audit logger initialized")

    async def cleanup(self):
        """Cleanup audit logger"""
        logger.info("Audit logger cleaned up")

    async def log_job_event(self, job_id: str, event_type: str,
                            user_id: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None):
        """Log a job-related event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'job_id': job_id,
            'user_id': user_id,
            'metadata': metadata or {}
        }

        try:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    async def log_data_access(self, table_name: str, operation: str,
                              user_id: Optional[str] = None,
                              row_count: int = 0):
        """Log data access events"""
        await self.log_job_event(
            job_id=f"data_access_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            event_type="DATA_ACCESS",
            user_id=user_id,
            metadata={
                'table_name': table_name,
                'operation': operation,
                'row_count': row_count
            }
        )

    async def log_security_event(self, event_type: str, details: Dict[str, Any],
                                 user_id: Optional[str] = None):
        """Log security-related events"""
        await self.log_job_event(
            job_id=f"security_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            event_type=f"SECURITY_{event_type}",
            user_id=user_id,
            metadata=details
        )

    def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get audit summary for the last N days"""
        try:
            events = []
            if self.audit_file.exists():
                with open(self.audit_file, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            events.append(event)
                        except json.JSONDecodeError:
                            continue

            # Simple summary - in production would filter by date
            event_types = {}
            users = set()

            for event in events[-1000:]:  # Last 1000 events as sample
                event_type = event.get('event_type', 'UNKNOWN')
                event_types[event_type] = event_types.get(event_type, 0) + 1
                if event.get('user_id'):
                    users.add(event['user_id'])

            return {
                'total_events': len(events),
                'event_types': event_types,
                'unique_users': len(users),
                'recent_events': events[-10:]  # Last 10 events
            }

        except Exception as e:
            logger.error(f"Error generating audit summary: {e}")
            return {}
