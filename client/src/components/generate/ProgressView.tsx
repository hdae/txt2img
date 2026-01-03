/**
 * ProgressView - Display generation progress
 */

import { Box, Flex, Progress, Text } from "@radix-ui/themes"

import { useGenerateStore } from "@/stores/generateStore"

export const ProgressView = () => {
    const job = useGenerateStore((state) => state.job)

    const progressPercent =
        job.totalSteps > 0 ? (job.step / job.totalSteps) * 100 : 0

    return (
        <Box
            p="4"
            style={{
                backgroundColor: "var(--gray-2)",
                borderRadius: "var(--radius-3)",
                border: "1px solid var(--gray-6)",
            }}
        >
            <Flex direction="column" gap="3">
                {/* Status */}
                <Flex justify="between" align="center">
                    <Text size="2" weight="medium">
                        {job.status === "queued" ? "キュー待機中" : "生成中"}
                    </Text>
                    {job.status === "queued" && job.queuePosition !== null && (
                        <Text size="2" color="gray">
                            あと {job.queuePosition + 1} 件
                        </Text>
                    )}
                </Flex>

                {/* Progress bar */}
                {job.status === "processing" && (
                    <>
                        <Progress value={progressPercent} size="2" />
                        <Flex justify="between">
                            <Text size="1" color="gray">
                                ステップ {job.step} / {job.totalSteps}
                            </Text>
                            <Text size="1" color="gray">
                                {progressPercent.toFixed(0)}%
                            </Text>
                        </Flex>
                    </>
                )}

                {/* Preview */}
                {job.preview && (
                    <Box
                        style={{
                            borderRadius: "var(--radius-2)",
                            overflow: "hidden",
                        }}
                    >
                        <img
                            src={`data:image/jpeg;base64,${job.preview}`}
                            alt="Preview"
                            style={{
                                width: "100%",
                                maxWidth: 300,
                                height: "auto",
                                opacity: 0.8,
                            }}
                        />
                    </Box>
                )}
            </Flex>
        </Box>
    )
}
