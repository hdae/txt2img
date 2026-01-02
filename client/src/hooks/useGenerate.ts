/**
 * useGenerate hook - handles image generation flow
 */

import { useCallback, useEffect, useRef } from "react"
import toast from "react-hot-toast"

import { createGenerateJob } from "@/api/client"
import { connectToJobSSE } from "@/api/sse"
import type { GenerateRequest } from "@/api/types"
import { useGenerateStore } from "@/stores/generateStore"

/**
 * Generate a random seed
 */
function generateRandomSeed(): number {
    return Math.floor(Math.random() * 2147483647)
}

export function useGenerate() {
    const {
        form,
        job,
        startJob,
        updateJobStatus,
        updateQueuePosition,
        updateProgress,
        completeJob,
        failJob,
        resetJob,
        setSeed,
    } = useGenerateStore()

    const cleanupRef = useRef<(() => void) | null>(null)

    // Cleanup SSE on unmount
    useEffect(() => {
        return () => {
            cleanupRef.current?.()
        }
    }, [])

    const generate = useCallback(async () => {
        // Cleanup previous SSE connection
        cleanupRef.current?.()

        // Generate seed if not locked
        let seedToUse = form.seed
        if (!form.seedLocked) {
            seedToUse = generateRandomSeed()
            setSeed(seedToUse) // Update store with generated seed
        }

        // Build request
        const request: GenerateRequest = {
            prompt: form.prompt,
            negative_prompt: form.negativePrompt || undefined,
            width: form.width,
            height: form.height,
            seed: seedToUse,
            loras: form.loras.length > 0 ? form.loras : undefined,
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
                            // Initial status event
                            if (payload.status) {
                                // Map 'running' to 'processing' for consistency
                                const status = payload.status === "running" ? "processing" : payload.status
                                updateJobStatus(status as "queued" | "processing")
                            }
                            if (typeof payload.queue_position === "number") {
                                updateQueuePosition(payload.queue_position)
                            }
                            break

                        case "started":
                            // Job started processing
                            updateJobStatus("processing")
                            break

                        case "queue_update":
                            // Queue position changed
                            if (typeof payload.queue_position === "number") {
                                updateQueuePosition(payload.queue_position)
                            }
                            break

                        case "progress":
                            // Server sends 'current_step', not 'step'
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
                            break
                        }

                        case "failed":
                            failJob(payload.error as string)
                            toast.error(`生成失敗: ${payload.error}`)
                            cleanupRef.current?.()
                            break

                        case "ping":
                            // Keepalive, ignore
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
        form,
        startJob,
        updateJobStatus,
        updateQueuePosition,
        updateProgress,
        completeJob,
        failJob,
        setSeed,
    ])

    const cancel = useCallback(() => {
        cleanupRef.current?.()
        resetJob()
    }, [resetJob])

    return {
        generate,
        cancel,
        isGenerating: job.status === "queued" || job.status === "processing",
        job,
    }
}
