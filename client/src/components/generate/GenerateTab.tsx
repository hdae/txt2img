/**
 * GenerateTab - Main generation interface with 2-column layout for PC
 */

import { Box, Button, Flex, Grid, Spinner, Text } from "@radix-ui/themes"
import { Sparkles } from "lucide-react"

import { LoraSelector } from "@/components/generate/LoraSelector"
import { NegativePrompt } from "@/components/generate/NegativePrompt"
import { ProgressView } from "@/components/generate/ProgressView"
import { PromptEditor } from "@/components/generate/PromptEditor"
import { ResultView } from "@/components/generate/ResultView"
import { SeedInput } from "@/components/generate/SeedInput"
import { SizeSelector } from "@/components/generate/SizeSelector"
import { useGenerate } from "@/hooks/useGenerate"
import { useServerInfo } from "@/hooks/useServerInfo"

export const GenerateTab = () => {
    const { data: serverInfo, isLoading: isLoadingInfo } = useServerInfo()
    const { generate, job } = useGenerate()

    const isSDXL = serverInfo?.parameter_schema?.model_type === "sdxl"
    const hasNegativePrompt = serverInfo?.parameter_schema?.properties?.negative_prompt !== undefined

    // Disable button until completed or failed (or idle)
    const isButtonDisabled = job.status === "queued" || job.status === "processing"

    if (isLoadingInfo) {
        return (
            <Flex align="center" justify="center" py="9">
                <Spinner size="3" />
                <Text ml="3">サーバー情報を取得中...</Text>
            </Flex>
        )
    }

    return (
        <Grid
            columns={{ initial: "1", md: "3fr 2fr" }}
            gap="4"
        >
            {/* Left Column: Input (larger) */}
            <Flex direction="column" gap="4">
                {/* Prompt */}
                <PromptEditor />

                {/* Negative Prompt (SDXL only) */}
                {hasNegativePrompt && <NegativePrompt />}

                {/* Size & Seed */}
                <Flex gap="4" wrap="wrap">
                    <Box style={{ flex: 1, minWidth: 180 }}>
                        <SizeSelector trainingResolution={parseInt(serverInfo?.training_resolution || "1024")} />
                    </Box>
                    <Box style={{ flex: 1, minWidth: 180 }}>
                        <SeedInput />
                    </Box>
                </Flex>

                {/* Generate Button (between Size/Seed and LoRA) */}
                <Button
                    size="3"
                    onClick={generate}
                    disabled={isButtonDisabled}
                    style={{ cursor: isButtonDisabled ? "not-allowed" : "pointer" }}
                >
                    {isButtonDisabled ? (
                        <>
                            <Spinner size="2" />
                            生成中...
                        </>
                    ) : (
                        <>
                            <Sparkles size={18} />
                            生成
                        </>
                    )}
                </Button>

                {/* LoRA Selector (SDXL only) */}
                {isSDXL && serverInfo?.available_loras && serverInfo.available_loras.length > 0 && (
                    <LoraSelector loras={serverInfo.available_loras} />
                )}
            </Flex>

            {/* Right Column: Output */}
            <Flex direction="column" gap="4">
                {/* Progress */}
                {(job.status === "queued" || job.status === "processing") && <ProgressView />}

                {/* Result */}
                {job.status === "completed" && job.result && (
                    <ResultView result={job.result} outputFormat={serverInfo?.output_format || "png"} />
                )}

                {/* Error */}
                {job.status === "failed" && job.error && (
                    <Box
                        p="3"
                        style={{
                            backgroundColor: "var(--red-3)",
                            borderRadius: "var(--radius-2)",
                            border: "1px solid var(--red-6)",
                        }}
                    >
                        <Text color="red" size="2">
                            エラー: {job.error}
                        </Text>
                    </Box>
                )}

                {/* Placeholder for idle state */}
                {job.status === "idle" && (
                    <Box
                        p="6"
                        style={{
                            backgroundColor: "var(--gray-2)",
                            borderRadius: "var(--radius-3)",
                            border: "1px dashed var(--gray-6)",
                            textAlign: "center",
                        }}
                    >
                        <Text color="gray" size="2">
                            生成された画像はここに表示されます
                        </Text>
                    </Box>
                )}
            </Flex>
        </Grid>
    )
}
