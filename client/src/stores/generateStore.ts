/**
 * Generate store - manages generation form state
 */

import { create } from "zustand"
import { immer } from "zustand/middleware/immer"

import type { LoraRequest, ParameterSchema } from "@/api/types"

// =============================================================================
// Types
// =============================================================================

interface GenerateFormState {
    prompt: string
    negativePrompt: string
    aspectRatio: string  // "1:1", "3:4", etc. - persisted
    width: number  // Not persisted, calculated from aspectRatio + trainingResolution
    height: number  // Not persisted
    cfgScale: number  // Not persisted, use schema default
    sampler: string  // Not persisted, use schema default
    seed: number | null
    seedLocked: boolean
    loras: LoraRequest[]
}

type JobStatus = "idle" | "queued" | "processing" | "completed" | "failed"

interface JobProgress {
    jobId: string | null
    status: JobStatus
    queuePosition: number | null
    step: number
    totalSteps: number
    preview: string | null
    result: {
        imageId: string
        imageUrl: string
        thumbnailUrl: string
    } | null
    error: string | null
}

interface GenerateStore {
    // Form state
    form: GenerateFormState

    // Job state
    job: JobProgress

    // Actions - Form
    setPrompt: (prompt: string) => void
    setNegativePrompt: (negativePrompt: string) => void
    setWidth: (width: number) => void
    setHeight: (height: number) => void
    setSize: (width: number, height: number) => void
    setAspectRatio: (aspectRatio: string) => void
    setCfgScale: (cfgScale: number) => void
    setSampler: (sampler: string) => void
    setSeed: (seed: number | null) => void
    toggleSeedLocked: () => void
    addLora: (lora: LoraRequest) => void
    removeLora: (loraId: string) => void
    updateLoraWeight: (loraId: string, weight: number) => void
    updateLoraTriggerWeight: (loraId: string, triggerWeight: number) => void
    clearLoras: () => void
    resetForm: (schema?: ParameterSchema) => void

    // Actions - Job
    startJob: (jobId: string) => void
    updateJobStatus: (status: JobStatus) => void
    updateQueuePosition: (position: number) => void
    updateProgress: (step: number, totalSteps: number, preview?: string) => void
    completeJob: (result: JobProgress["result"]) => void
    failJob: (error: string) => void
    resetJob: () => void
}

// =============================================================================
// Default Values
// =============================================================================

const defaultForm: GenerateFormState = {
    prompt: "",
    negativePrompt: "",
    aspectRatio: "1:1",
    width: 1024,
    height: 1024,
    cfgScale: 7.0,
    sampler: "euler_a",
    seed: null,
    seedLocked: false,
    loras: [],
}

const defaultJob: JobProgress = {
    jobId: null,
    status: "idle",
    queuePosition: null,
    step: 0,
    totalSteps: 0,
    preview: null,
    result: null,
    error: null,
}

// =============================================================================
// Store
// =============================================================================

import { persist } from "zustand/middleware"

export const useGenerateStore = create<GenerateStore>()(
    persist(
        immer((set) => ({
            form: defaultForm,
            job: defaultJob,

            // Form actions
            setPrompt: (prompt) =>
                set((state) => {
                    state.form.prompt = prompt
                }),

            setNegativePrompt: (negativePrompt) =>
                set((state) => {
                    state.form.negativePrompt = negativePrompt
                }),

            setWidth: (width) =>
                set((state) => {
                    state.form.width = width
                }),

            setHeight: (height) =>
                set((state) => {
                    state.form.height = height
                }),

            setSize: (width, height) =>
                set((state) => {
                    state.form.width = width
                    state.form.height = height
                }),

            setSeed: (seed) =>
                set((state) => {
                    state.form.seed = seed
                }),

            setAspectRatio: (aspectRatio) =>
                set((state) => {
                    state.form.aspectRatio = aspectRatio
                }),

            setCfgScale: (cfgScale) =>
                set((state) => {
                    state.form.cfgScale = cfgScale
                }),

            setSampler: (sampler) =>
                set((state) => {
                    state.form.sampler = sampler
                }),

            toggleSeedLocked: () =>
                set((state) => {
                    state.form.seedLocked = !state.form.seedLocked
                }),

            addLora: (lora) =>
                set((state) => {
                    const exists = state.form.loras.find((l) => l.id === lora.id)
                    if (!exists) {
                        state.form.loras.push(lora)
                    }
                }),

            removeLora: (loraId) =>
                set((state) => {
                    state.form.loras = state.form.loras.filter((l) => l.id !== loraId)
                }),

            updateLoraWeight: (loraId, weight) =>
                set((state) => {
                    const lora = state.form.loras.find((l) => l.id === loraId)
                    if (lora) {
                        lora.weight = weight
                    }
                }),

            updateLoraTriggerWeight: (loraId, triggerWeight) =>
                set((state) => {
                    const lora = state.form.loras.find((l) => l.id === loraId)
                    if (lora) {
                        lora.trigger_weight = triggerWeight
                    }
                }),

            clearLoras: () =>
                set((state) => {
                    state.form.loras = []
                }),

            resetForm: (schema) =>
                set((state) => {
                    const cfgDefault = (schema?.properties?.cfg_scale?.default as number) ?? 7.0
                    const samplerDefault = (schema?.properties?.sampler?.default as string) ?? "euler_a"
                    state.form = {
                        ...defaultForm,
                        aspectRatio: state.form.aspectRatio, // Keep current aspect ratio
                        width: (schema?.properties.width?.default as number) ?? 1024,
                        height: (schema?.properties.height?.default as number) ?? 1024,
                        cfgScale: cfgDefault,
                        sampler: samplerDefault,
                    }
                }),

            // Job actions
            startJob: (jobId) =>
                set((state) => {
                    state.job = {
                        ...defaultJob,
                        jobId,
                        status: "queued",
                    }
                }),

            updateJobStatus: (status) =>
                set((state) => {
                    state.job.status = status
                }),

            updateQueuePosition: (position) =>
                set((state) => {
                    state.job.queuePosition = position
                }),

            updateProgress: (step, totalSteps, preview) =>
                set((state) => {
                    state.job.status = "processing"
                    state.job.step = step
                    state.job.totalSteps = totalSteps
                    if (preview) {
                        state.job.preview = preview
                    }
                }),

            completeJob: (result) =>
                set((state) => {
                    state.job.status = "completed"
                    state.job.result = result
                }),

            failJob: (error) =>
                set((state) => {
                    state.job.status = "failed"
                    state.job.error = error
                }),

            resetJob: () =>
                set((state) => {
                    state.job = defaultJob
                }),
        })),
        {
            name: "txt2img-generate-form",
            partialize: (state) => ({
                form: {
                    // Persist user preferences
                    prompt: state.form.prompt,
                    negativePrompt: state.form.negativePrompt,
                    aspectRatio: state.form.aspectRatio,
                    cfgScale: state.form.cfgScale,
                    sampler: state.form.sampler,
                    seedLocked: state.form.seedLocked,
                    seed: state.form.seedLocked ? state.form.seed : null,
                    loras: state.form.loras,
                }
            }),
        }
    )
)
