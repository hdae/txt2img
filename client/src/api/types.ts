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
    cfg_scale?: number
    sampler?: string
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
    output_format: "png" | "webp"
    available_loras: LoraInfo[]
    parameter_schema: ParameterSchema
}

// =============================================================================
// Parameter Schema (from /api/info)
// =============================================================================

type PromptStyle = "tags" | "natural"
type ModelType = "sdxl" | "chroma" | "flux_dev" | "flux_schnell" | "zimage" | "anima"

interface PropertySchema {
    type: string | string[]
    default?: unknown
    description?: string
    minimum?: number
    maximum?: number
    step?: number
    enum?: string[]
    items?: {
        type: string
        properties?: Record<string, PropertySchema>
        required?: string[]
    }
}

export interface ParameterSchema {
    model_type: ModelType
    prompt_style: PromptStyle
    defaults?: {
        cfg_scale?: number
        sampler?: string
        steps?: number
    }
    properties: Record<string, PropertySchema>
    required: string[]
    fixed: {
        steps: number
        cfg_scale?: number
    }
}
