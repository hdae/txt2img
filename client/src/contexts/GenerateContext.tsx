/**
 * GenerateContext - Global SSE connection management for image generation
 *
 * This context manages SSE connections at the app level, preventing
 * connection loss when switching tabs.
 */

import { createContext, useCallback, useContext, useRef, type ReactNode } from "react"
import toast from "react-hot-toast"

import { createGenerateJob } from "@/api/client"
import { connectToJobSSE } from "@/api/sse"
import type { GenerateRequest } from "@/api/types"
import { useGenerateStore } from "@/stores/generateStore"

// =============================================================================
// Types
// =============================================================================

interface GenerateContextValue {
    generate: () => Promise<void>
    cancel: () => void
}

// =============================================================================
// Context
// =============================================================================

const GenerateContext = createContext<GenerateContextValue | null>(null)

/**
 * Generate a random seed
 */
function generateRandomSeed(): number {
    return Math.floor(Math.random() * 2147483647)
}

// =============================================================================
// Provider
// =============================================================================

interface GenerateProviderProps {
    children: ReactNode
}

export function GenerateProvider({ children }: GenerateProviderProps) {
    const cleanupRef = useRef<(() => void) | null>(null)

    // Get store actions (stable references)
    const {
        startJob,
        updateJobStatus,
        updateQueuePosition,
        updateProgress,
        completeJob,
        failJob,
        resetJob,
        setSeed,
    } = useGenerateStore()

    const generate = useCallback(async () => {
        // Cleanup previous SSE connection
        cleanupRef.current?.()

        // Get current form state
        const currentForm = useGenerateStore.getState().form

        // Generate seed if not locked
        let seedToUse = currentForm.seed
        if (!currentForm.seedLocked) {
            seedToUse = generateRandomSeed()
            setSeed(seedToUse)
        }

        // Build request
        const request: GenerateRequest = {
            prompt: currentForm.prompt,
            negative_prompt: currentForm.negativePrompt || undefined,
            width: currentForm.width,
            height: currentForm.height,
            seed: seedToUse,
            loras: currentForm.loras.length > 0 ? currentForm.loras : undefined,
        }

        try {
            // Create job
            const response = await createGenerateJob(request)
            startJob(response.job_id)

            // Connect to SSE
            cleanupRef.current = connectToJobSSE(response.job_id, {
                onMessage: (event: string, data: unknown) => {
                    const payload = data as Record<string, unknown>

                    switch (event) {
                        case "status":
                            if (payload.status) {
                                const status = payload.status === "running" ? "processing" : payload.status
                                updateJobStatus(status as "queued" | "processing")
                            }
                            if (typeof payload.queue_position === "number") {
                                updateQueuePosition(payload.queue_position)
                            }
                            break

                        case "started":
                            updateJobStatus("processing")
                            break

                        case "queue_update":
                            if (typeof payload.queue_position === "number") {
                                updateQueuePosition(payload.queue_position)
                            }
                            break

                        case "progress":
                            updateProgress(
                                payload.current_step as number,
                                payload.total_steps as number,
                                payload.preview as string | undefined
                            )
                            break

                        case "completed": {
                            const result = payload.result as {
                                image_id: string
                                image_url: string
                                thumbnail_url: string
                            }
                            completeJob({
                                imageId: result.image_id,
                                imageUrl: result.image_url,
                                thumbnailUrl: result.thumbnail_url,
                            })
                            toast.success("画像生成完了!")
                            cleanupRef.current?.()
                            cleanupRef.current = null
                            break
                        }

                        case "failed":
                            failJob(payload.error as string)
                            toast.error(`生成失敗: ${payload.error}`)
                            cleanupRef.current?.()
                            cleanupRef.current = null
                            break

                        case "ping":
                            // Keepalive, ignore
                            break

                        case "error":
                            failJob(payload.error as string || "Unknown server error")
                            toast.error(`サーバーエラー: ${payload.error || "不明なエラー"}`)
                            cleanupRef.current?.()
                            cleanupRef.current = null
                            break
                    }
                },
                onError: () => {
                    failJob("SSE connection error")
                    toast.error("接続エラーが発生しました")
                },
            })
        } catch (error) {
            const message = error instanceof Error ? error.message : "Unknown error"
            failJob(message)
            toast.error(`エラー: ${message}`)
        }
    }, [
        setSeed,
        startJob,
        updateJobStatus,
        updateQueuePosition,
        updateProgress,
        completeJob,
        failJob,
    ])

    const cancel = useCallback(() => {
        cleanupRef.current?.()
        cleanupRef.current = null
        resetJob()
    }, [resetJob])

    return (
        <GenerateContext.Provider value={{ generate, cancel }}>
            {children}
        </GenerateContext.Provider>
    )
}

// =============================================================================
// Hook
// =============================================================================

// eslint-disable-next-line react-refresh/only-export-components
export function useGenerateContext() {
    const context = useContext(GenerateContext)
    if (!context) {
        throw new Error("useGenerateContext must be used within a GenerateProvider")
    }
    return context
}
