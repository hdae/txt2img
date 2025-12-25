"""SSE (Server-Sent Events) endpoint."""

import asyncio
import json
import logging

from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request

from txt2img.core.job_queue import JobStatus, job_queue

logger = logging.getLogger(__name__)


async def job_event_generator(request: Request, job_id: str):
    """Generate SSE events for a job.

    Args:
        request: Starlette request
        job_id: Job ID to monitor

    Yields:
        SSE events
    """
    job = job_queue.get_job(job_id)
    if not job:
        yield {
            "event": "error",
            "data": json.dumps({"error": "Job not found"}),
        }
        return

    # If job is already completed, send final status
    if job.status == JobStatus.COMPLETED:
        yield {
            "event": "completed",
            "data": json.dumps(
                {
                    "result": {
                        "image_id": job.result.image_id,
                        "image_url": job.result.image_url,
                        "thumbnail_url": job.result.thumbnail_url,
                    }
                }
            ),
        }
        return

    if job.status == JobStatus.FAILED:
        yield {
            "event": "failed",
            "data": json.dumps({"error": job.error}),
        }
        return

    # Subscribe to job updates
    event_queue = await job_queue.subscribe(job_id)

    try:
        # Send initial status
        yield {
            "event": "status",
            "data": json.dumps(
                {
                    "status": job.status.value,
                    "progress": job.progress,
                    "queue_position": job_queue.queue_size,
                }
            ),
        }

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(event_queue.get(), timeout=30.0)

                event_type = event.get("type", "update")

                yield {
                    "event": event_type,
                    "data": json.dumps(event),
                }

                # End stream on completion or failure
                if event_type in ("completed", "failed"):
                    break

            except TimeoutError:
                # Send keepalive
                yield {
                    "event": "ping",
                    "data": json.dumps({"status": "alive"}),
                }

    finally:
        await job_queue.unsubscribe(job_id, event_queue)


async def sse_endpoint(request: Request, job_id: str) -> EventSourceResponse:
    """SSE endpoint for job progress.

    Args:
        request: Starlette request
        job_id: Job ID

    Returns:
        EventSourceResponse
    """
    return EventSourceResponse(
        job_event_generator(request, job_id),
        media_type="text/event-stream",
    )
