/**
 * API client for txt2img server
 */

import type {
    GenerateRequest,
    GenerateResponse,
    ImageListResponse,
    ServerInfo,
} from "./types"

const API_BASE = "/api"

class APIError extends Error {
    public status: number
    public statusText: string
    public detail?: string

    constructor(status: number, statusText: string, detail?: string) {
        super(detail || statusText)
        this.name = "APIError"
        this.status = status
        this.statusText = statusText
        this.detail = detail
    }
}

async function handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
        let detail: string | undefined
        try {
            const json = await response.json()
            detail = json.detail || json.error
        } catch {
            // ignore
        }
        throw new APIError(response.status, response.statusText, detail)
    }
    return response.json()
}

/**
 * Get server information including model name, LoRAs, and parameter schema
 */
export async function getServerInfo(): Promise<ServerInfo> {
    const response = await fetch(`${API_BASE}/info`)
    return handleResponse<ServerInfo>(response)
}

/**
 * Create a new image generation job
 */
export async function createGenerateJob(
    request: GenerateRequest
): Promise<GenerateResponse> {
    const response = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
    })
    return handleResponse<GenerateResponse>(response)
}

/**
 * Get list of generated images
 */
export async function getImages(
    limit = 50,
    offset = 0
): Promise<ImageListResponse> {
    const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
    })
    const response = await fetch(`${API_BASE}/images?${params}`)
    return handleResponse<ImageListResponse>(response)
}

/**
 * Get full image URL
 */
export function getImageUrl(imageId: string, ext: "png" | "webp" = "webp"): string {
    return `${API_BASE}/images/${imageId}.${ext}`
}

/**
 * Get thumbnail URL
 */
export function getThumbnailUrl(imageId: string): string {
    return `${API_BASE}/thumbs/${imageId}.webp`
}
