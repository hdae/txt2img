/**
 * useGenerate hook - wrapper for GenerateContext
 *
 * Provides generate/cancel functions and job state from the store.
 */

import { useGenerateContext } from "@/contexts/GenerateContext"
import { useGenerateStore } from "@/stores/generateStore"

export function useGenerate() {
    const { generate, cancel } = useGenerateContext()
    const job = useGenerateStore((state) => state.job)

    return {
        generate,
        cancel,
        isGenerating: job.status === "queued" || job.status === "processing",
        job,
    }
}
