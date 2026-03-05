"""Cloud Execution Service for asynchronous task processing.

This module provides:
- CloudExecutor: Execute tasks in cloud environment
- TaskQueue: Queue management for cloud tasks
- StatusSync: Synchronize task status between local and cloud
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import aiohttp
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of cloud task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Priority levels for cloud tasks."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class CloudTask:
    """A task for cloud execution."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudTask":
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            parameters=data["parameters"],
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 5)),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            error=data.get("error"),
            progress=data.get("progress", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CloudConfig:
    """Configuration for cloud execution."""
    api_endpoint: str = "https://api.pyutagent.cloud"
    api_key: str = ""
    max_concurrent_tasks: int = 5
    task_timeout: int = 3600
    poll_interval: int = 5
    retry_attempts: int = 3
    webhook_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_endpoint": self.api_endpoint,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "task_timeout": self.task_timeout,
            "poll_interval": self.poll_interval,
            "retry_attempts": self.retry_attempts,
            "webhook_url": self.webhook_url,
        }


class CloudExecutor:
    """Execute tasks in cloud environment."""
    
    def __init__(self, config: CloudConfig):
        """Initialize cloud executor.
        
        Args:
            config: Cloud configuration
        """
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._tasks: Dict[str, CloudTask] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session
    
    async def submit_task(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CloudTask:
        """Submit a task for cloud execution.
        
        Args:
            task_type: Type of task
            parameters: Task parameters
            priority: Task priority
            metadata: Optional metadata
            
        Returns:
            CloudTask with task_id
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = CloudTask(
            task_id=task_id,
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            metadata=metadata or {},
        )
        
        session = await self._get_session()
        
        try:
            async with session.post(
                f"{self.config.api_endpoint}/tasks",
                json=task.to_dict()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    task.status = TaskStatus.QUEUED
                    self._tasks[task_id] = task
                    logger.info(f"[CloudExecutor] Task {task_id} submitted")
                else:
                    error = await response.text()
                    task.status = TaskStatus.FAILED
                    task.error = f"Submit failed: {error}"
                    logger.error(f"[CloudExecutor] Failed to submit task: {error}")
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"[CloudExecutor] Submit error: {e}")
        
        return task
    
    async def get_status(self, task_id: str) -> Optional[CloudTask]:
        """Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Updated CloudTask or None
        """
        session = await self._get_session()
        
        try:
            async with session.get(
                f"{self.config.api_endpoint}/tasks/{task_id}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    task = CloudTask.from_dict(data)
                    self._tasks[task_id] = task
                    return task
                else:
                    return None
        
        except Exception as e:
            logger.error(f"[CloudExecutor] Status check error: {e}")
            return None
    
    async def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None
        """
        task = await self.get_status(task_id)
        
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancelled successfully
        """
        session = await self._get_session()
        
        try:
            async with session.post(
                f"{self.config.api_endpoint}/tasks/{task_id}/cancel"
            ) as response:
                if response.status == 200:
                    if task_id in self._tasks:
                        self._tasks[task_id].status = TaskStatus.CANCELLED
                    logger.info(f"[CloudExecutor] Task {task_id} cancelled")
                    return True
                return False
        
        except Exception as e:
            logger.error(f"[CloudExecutor] Cancel error: {e}")
            return False
    
    async def wait_for_completion(
        self,
        task_id: str,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> CloudTask:
        """Wait for task completion.
        
        Args:
            task_id: Task ID
            timeout: Optional timeout in seconds
            progress_callback: Optional progress callback
            
        Returns:
            Final CloudTask
        """
        timeout = timeout or self.config.task_timeout
        start_time = datetime.now()
        
        while True:
            task = await self.get_status(task_id)
            
            if task is None:
                raise Exception(f"Task {task_id} not found")
            
            if progress_callback and task.progress > 0:
                progress_callback(task.progress)
            
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT):
                return task
            
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                task.status = TaskStatus.TIMEOUT
                task.error = "Task timed out"
                return task
            
            await asyncio.sleep(self.config.poll_interval)
    
    async def execute_with_callback(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        callback: Callable[[CloudTask], None],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> CloudTask:
        """Execute task and call callback on completion.
        
        Args:
            task_type: Type of task
            parameters: Task parameters
            callback: Completion callback
            priority: Task priority
            
        Returns:
            Submitted CloudTask
        """
        task = await self.submit_task(task_type, parameters, priority)
        
        async def _wait_and_callback():
            final_task = await self.wait_for_completion(task.task_id)
            callback(final_task)
        
        asyncio.create_task(_wait_and_callback())
        
        return task
    
    async def batch_submit(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[CloudTask]:
        """Submit multiple tasks.
        
        Args:
            tasks: List of task specifications
            
        Returns:
            List of submitted CloudTasks
        """
        submitted = []
        
        for task_spec in tasks:
            task = await self.submit_task(
                task_type=task_spec["task_type"],
                parameters=task_spec["parameters"],
                priority=task_spec.get("priority", TaskPriority.NORMAL),
                metadata=task_spec.get("metadata"),
            )
            submitted.append(task)
        
        return submitted
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class TaskQueue:
    """Local task queue for cloud execution."""
    
    def __init__(self, storage_path: str = ".pyutagent/queue"):
        """Initialize task queue.
        
        Args:
            storage_path: Path for queue storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._queue: List[CloudTask] = []
        self._load_queue()
    
    def _load_queue(self):
        """Load queue from storage."""
        queue_file = self.storage_path / "queue.json"
        
        if queue_file.exists():
            try:
                with open(queue_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._queue = [CloudTask.from_dict(t) for t in data]
            except Exception as e:
                logger.warning(f"[TaskQueue] Failed to load queue: {e}")
    
    def _save_queue(self):
        """Save queue to storage."""
        queue_file = self.storage_path / "queue.json"
        
        with open(queue_file, 'w', encoding='utf-8') as f:
            json.dump([t.to_dict() for t in self._queue], f, indent=2)
    
    def enqueue(self, task: CloudTask):
        """Add task to queue."""
        self._queue.append(task)
        self._queue.sort(key=lambda t: t.priority.value, reverse=True)
        self._save_queue()
        logger.info(f"[TaskQueue] Enqueued task {task.task_id}")
    
    def dequeue(self) -> Optional[CloudTask]:
        """Get next task from queue."""
        for task in self._queue:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.QUEUED
                self._save_queue()
                return task
        return None
    
    def peek(self) -> Optional[CloudTask]:
        """Peek at next task without removing."""
        for task in self._queue:
            if task.status == TaskStatus.PENDING:
                return task
        return None
    
    def update_task(self, task: CloudTask):
        """Update task in queue."""
        for i, t in enumerate(self._queue):
            if t.task_id == task.task_id:
                self._queue[i] = task
                self._save_queue()
                break
    
    def remove_task(self, task_id: str):
        """Remove task from queue."""
        self._queue = [t for t in self._queue if t.task_id != task_id]
        self._save_queue()
    
    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        return sum(1 for t in self._queue if t.status == TaskStatus.PENDING)
    
    def get_all(self) -> List[CloudTask]:
        """Get all tasks."""
        return self._queue.copy()


class WebhookNotifier:
    """Notify webhooks about task completion."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        """Initialize webhook notifier.
        
        Args:
            webhook_url: Webhook URL
        """
        self.webhook_url = webhook_url
    
    async def notify(self, task: CloudTask):
        """Send webhook notification."""
        if not self.webhook_url:
            return
        
        payload = {
            "event": "task_completed",
            "task_id": task.task_id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.info(f"[WebhookNotifier] Notified for task {task.task_id}")
                    else:
                        logger.warning(f"[WebhookNotifier] Webhook failed: {response.status}")
        
        except Exception as e:
            logger.error(f"[WebhookNotifier] Notification error: {e}")
