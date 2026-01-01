"""Job queue for image generation."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enum."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationParams:
    """Parameters for image generation.

    Note: steps, cfg_scale, sampler are fixed per pipeline.
    """

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    loras: list[dict] | None = None  # [{"id": "...", "weight": 1.0}]


@dataclass
class JobResult:
    """Result of a completed job."""

    image_id: str
    image_url: str
    thumbnail_url: str
    metadata: dict[str, Any]


@dataclass
class Job:
    """Image generation job."""

    id: str
    params: GenerationParams
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    preview_base64: str | None = None
    result: JobResult | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "result": {
                "image_id": self.result.image_id,
                "image_url": self.result.image_url,
                "thumbnail_url": self.result.thumbnail_url,
            }
            if self.result
            else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }


class JobQueue:
    """Queue for managing image generation jobs."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    def create_job(self, params: GenerationParams) -> Job:
        """Create a new job and add to queue.

        Args:
            params: Generation parameters

        Returns:
            Created job
        """
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, params=params, total_steps=params.steps)
        self._jobs[job_id] = job
        self._queue.put_nowait(job)
        logger.info(f"Created job {job_id}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    async def get_next_job(self) -> Job:
        """Get next job from queue (blocks until available)."""
        return await self._queue.get()

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def pending_jobs(self) -> list[Job]:
        """Get list of pending jobs."""
        return [
            job
            for job in self._jobs.values()
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        ]

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to job updates.

        Args:
            job_id: Job ID to subscribe to

        Returns:
            Queue that will receive updates
        """
        async with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            queue: asyncio.Queue = asyncio.Queue()
            self._subscribers[job_id].append(queue)
            return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from job updates."""
        async with self._lock:
            if job_id in self._subscribers:
                try:
                    self._subscribers[job_id].remove(queue)
                except ValueError:
                    pass
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    async def notify_subscribers(self, job_id: str, event: dict[str, Any]) -> None:
        """Send event to all subscribers of a job."""
        async with self._lock:
            subscribers = self._subscribers.get(job_id, [])
            for queue in subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    async def update_job_progress(
        self,
        job_id: str,
        current_step: int,
        preview_base64: str | None = None,
    ) -> None:
        """Update job progress and notify subscribers.

        Args:
            job_id: Job ID
            current_step: Current step number
            preview_base64: Optional preview image as base64
        """
        job = self._jobs.get(job_id)
        if not job:
            return

        job.current_step = current_step
        job.progress = current_step / job.total_steps if job.total_steps > 0 else 0
        job.preview_base64 = preview_base64

        await self.notify_subscribers(
            job_id,
            {
                "type": "progress",
                "current_step": current_step,
                "total_steps": job.total_steps,
                "progress": job.progress,
                "preview": preview_base64,
            },
        )

    async def mark_job_running(self, job_id: str) -> None:
        """Mark job as running."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            await self.notify_subscribers(job_id, {"type": "started"})

    async def mark_job_completed(self, job_id: str, result: JobResult) -> None:
        """Mark job as completed with result."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.result = result
            job.progress = 1.0
            job.completed_at = datetime.now()
            await self.notify_subscribers(
                job_id,
                {
                    "type": "completed",
                    "result": {
                        "image_id": result.image_id,
                        "image_url": result.image_url,
                        "thumbnail_url": result.thumbnail_url,
                    },
                },
            )

    async def mark_job_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed with error."""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = error
            job.completed_at = datetime.now()
            await self.notify_subscribers(job_id, {"type": "failed", "error": error})


# Global job queue instance
job_queue = JobQueue()
