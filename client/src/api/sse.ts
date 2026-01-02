/**
 * SSE (Server-Sent Events) utilities
 */

interface SSEOptions {
    onMessage: (event: string, data: unknown) => void
    onError?: (error: Event) => void
    onOpen?: () => void
    eventTypes?: string[]
}

// Default event types for job progress
const DEFAULT_JOB_EVENT_TYPES = [
    "status",
    "started",
    "queue_update",
    "progress",
    "completed",
    "failed",
    "error",
    "ping",
]

// Default event types for gallery
const DEFAULT_GALLERY_EVENT_TYPES = [
    "connected",
    "new_image",
    "ping",
]

/**
 * Create an SSE connection and return cleanup function
 */
function createSSEConnection(
    url: string,
    options: SSEOptions
): () => void {
    const eventSource = new EventSource(url)

    eventSource.onopen = () => {
        options.onOpen?.()
    }

    eventSource.onerror = (error) => {
        options.onError?.(error)
    }

    // Handle named events
    const eventTypes = options.eventTypes ?? DEFAULT_JOB_EVENT_TYPES

    for (const eventType of eventTypes) {
        eventSource.addEventListener(eventType, (event) => {
            try {
                const data = JSON.parse(event.data)
                options.onMessage(eventType, data)
            } catch {
                options.onMessage(eventType, event.data)
            }
        })
    }

    // Handle generic message events
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data)
            options.onMessage("message", data)
        } catch {
            options.onMessage("message", event.data)
        }
    }

    // Return cleanup function
    return () => {
        eventSource.close()
    }
}

/**
 * Connect to job SSE endpoint
 */
export function connectToJobSSE(
    jobId: string,
    options: Omit<SSEOptions, "eventTypes">
): () => void {
    return createSSEConnection(`/api/jobs/${jobId}/sse`, {
        ...options,
        eventTypes: DEFAULT_JOB_EVENT_TYPES,
    })
}

/**
 * Connect to gallery SSE endpoint
 */
export function connectToGallerySSE(
    options: Omit<SSEOptions, "eventTypes">
): () => void {
    return createSSEConnection("/api/gallery/sse", {
        ...options,
        eventTypes: DEFAULT_GALLERY_EVENT_TYPES,
    })
}
