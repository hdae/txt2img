/**
 * GenerateTab - Main generation interface
 */

import { Box, Button, Flex, Spinner, Text } from "@radix-ui/themes"
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
    const { generate, isGenerating, job } = useGenerate()

    const isSDXL = serverInfo?.parameter_schema?.model_type === "sdxl"
    const hasNegativePrompt = serverInfo?.parameter_schema?.properties?.negative_prompt !== undefined

    if (isLoadingInfo) {
        return (
            <Flex align="center" justify="center" py="9">
                <Spinner size="3" />
                <Text ml="3">サーバー情報を取得中...</Text>
            </Flex>
        )
    }

    return (
        <Flex direction="column" gap="4">
            {/* Model Info */}
            <Flex justify="between" align="center">
                <Text size="2" color="gray">
                    モデル: {serverInfo?.model_name || "不明"}
                </Text>
                <Text size="2" color="gray">
                    解像度: {serverInfo?.training_resolution || "不明"}
                </Text>
            </Flex>

            {/* Prompt */}
            <Box>
                <Text as="label" size="2" weight="medium" mb="1">
                    プロンプト
                </Text>
                <PromptEditor />
            </Box>

            {/* Negative Prompt (SDXL only) */}
            {hasNegativePrompt && <NegativePrompt />}

            {/* Size & Seed */}
            <Flex gap="4" wrap="wrap">
                <Box style={{ flex: 1, minWidth: 200 }}>
                    <SizeSelector />
                </Box>
                <Box style={{ flex: 1, minWidth: 200 }}>
                    <SeedInput />
                </Box>
            </Flex>

            {/* LoRA Selector (SDXL only) */}
            {isSDXL && serverInfo?.available_loras && serverInfo.available_loras.length > 0 && (
                <LoraSelector loras={serverInfo.available_loras} />
            )}

            {/* Generate Button */}
            <Button
                size="3"
                onClick={generate}
                disabled={isGenerating}
                style={{ cursor: isGenerating ? "not-allowed" : "pointer" }}
            >
                {isGenerating ? (
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

            {/* Progress */}
            {(job.status === "queued" || job.status === "processing") && <ProgressView />}

            {/* Result */}
            {job.status === "completed" && job.result && <ResultView />}

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
        </Flex>
    )
}
