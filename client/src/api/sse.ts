/**
 * SSE (Server-Sent Events) utilities
 */

export interface SSEOptions {
    onMessage: (event: string, data: unknown) => void
    onError?: (error: Event) => void
    onOpen?: () => void
}

/**
 * Create an SSE connection and return cleanup function
 */
export function createSSEConnection(
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
    const eventTypes = [
        "status",
        "progress",
        "completed",
        "failed",
        "ping",
        "connected",
        "new_image",
    ]

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
    options: SSEOptions
): () => void {
    return createSSEConnection(`/api/sse/${jobId}`, options)
}

/**
 * Connect to gallery SSE endpoint
 */
export function connectToGallerySSE(options: SSEOptions): () => void {
    return createSSEConnection("/api/sse/gallery", options)
}
