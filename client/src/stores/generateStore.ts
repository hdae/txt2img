/**
 * Generate store - manages generation form state
 */

import { create } from "zustand"
import { immer } from "zustand/middleware/immer"

import type { LoraRequest, ParameterSchema } from "@/api/types"

// =============================================================================
// Types
// =============================================================================

export interface GenerateFormState {
    prompt: string
    negativePrompt: string
    width: number
    height: number
    seed: number | null
    seedLocked: boolean
    loras: LoraRequest[]
}

export type JobStatus = "idle" | "queued" | "processing" | "completed" | "failed"

export interface JobProgress {
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
    width: 1024,
    height: 1024,
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

export const useGenerateStore = create<GenerateStore>()(
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
                state.form = {
                    ...defaultForm,
                    width: (schema?.properties.width?.default as number) ?? 1024,
                    height: (schema?.properties.height?.default as number) ?? 1024,
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
    }))
)
