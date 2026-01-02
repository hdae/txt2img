/**
 * API type definitions
 * Based on server schemas (txt2img/api/schemas.py)
 */

// =============================================================================
// Request Types
// =============================================================================

export interface LoraRequest {
    id: string
    weight?: number
    trigger_weight?: number
}

export interface GenerateRequest {
    prompt: string
    negative_prompt?: string
    width?: number
    height?: number
    seed?: number | null
    loras?: LoraRequest[] | null
}

// =============================================================================
// Response Types
// =============================================================================

export interface GenerateResponse {
    job_id: string
    sse_url: string
}

export interface ImageInfo {
    id: string
    thumbnail_url: string
    full_url: string
    metadata: Record<string, unknown>
}

export interface ImageListResponse {
    images: ImageInfo[]
    total: number
    offset: number
    limit: number
}

export interface LoraInfo {
    id: string
    name: string
    trigger_words: string[]
    weight: number
    trigger_weight: number
}

export interface ServerInfo {
    model_name: string
    training_resolution: string
    available_loras: LoraInfo[]
    parameter_schema: ParameterSchema
}

export interface ErrorResponse {
    error: string
    detail?: string
}

// =============================================================================
// Parameter Schema (from /api/info)
// =============================================================================

export type PromptStyle = "tags" | "natural"
export type ModelType = "sdxl" | "chroma" | "flux_dev" | "flux_schnell" | "zimage"

export interface PropertySchema {
    type: string | string[]
    default?: unknown
    description?: string
    minimum?: number
    maximum?: number
    items?: {
        type: string
        properties?: Record<string, PropertySchema>
        required?: string[]
    }
}

export interface ParameterSchema {
    model_type: ModelType
    prompt_style: PromptStyle
    properties: Record<string, PropertySchema>
    required: string[]
    fixed: {
        steps: number
        cfg_scale: number
        sampler?: string
    }
}

// =============================================================================
// SSE Event Types
// =============================================================================

export interface SSEStatusEvent {
    status: "queued" | "processing" | "completed" | "failed"
    progress?: number
    queue_position?: number
}

export interface SSEProgressEvent {
    type: "progress"
    step: number
    total_steps: number
    preview?: string // base64
}

export interface SSECompletedEvent {
    type: "completed"
    result: {
        image_id: string
        image_url: string
        thumbnail_url: string
    }
}

export interface SSEFailedEvent {
    type: "failed"
    error: string
}

export interface SSEGalleryEvent {
    type: "new_image"
    image: ImageInfo
}

export type SSEEvent =
    | SSEStatusEvent
    | SSEProgressEvent
    | SSECompletedEvent
    | SSEFailedEvent
    | SSEGalleryEvent
