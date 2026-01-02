/**
 * ImageModal - Full-screen modal for image preview with metadata
 */

import { Badge, Box, Flex, Grid, IconButton, ScrollArea, Text } from "@radix-ui/themes"
import { Download, ExternalLink, X } from "lucide-react"
import toast from "react-hot-toast"

interface ImageModalProps {
    imageUrl: string
    metadata: Record<string, unknown>
    onClose: () => void
}

// Parse prompt tags with weights (e.g., "(tag:1.2)" or "tag")
function parsePromptTags(prompt: string): { tag: string; weight: number }[] {
    const results: { tag: string; weight: number }[] = []

    // Split by comma and process each part
    const parts = prompt.split(",").map((p) => p.trim()).filter(Boolean)

    for (const part of parts) {
        // Match (text:weight) pattern
        const weightMatch = part.match(/^\((.+):(\d+\.?\d*)\)$/)
        if (weightMatch) {
            results.push({
                tag: weightMatch[1].trim(),
                weight: parseFloat(weightMatch[2]),
            })
        } else {
            // Check for multiple parentheses (Compel style: ((tag)) = 1.21^n)
            const parenMatch = part.match(/^(\(+)([^()]+)(\)+)$/)
            if (parenMatch && parenMatch[1].length === parenMatch[3].length) {
                const depth = parenMatch[1].length
                results.push({
                    tag: parenMatch[2].trim(),
                    weight: Math.pow(1.1, depth),
                })
            } else {
                // Plain tag
                results.push({
                    tag: part.replace(/[()]/g, "").trim(),
                    weight: 1.0,
                })
            }
        }
    }

    return results
}

export const ImageModal = ({ imageUrl, metadata, onClose }: ImageModalProps) => {
    const handleDownload = async () => {
        const response = await fetch(imageUrl)
        const blob = await response.blob()
        const blobUrl = URL.createObjectURL(blob)
        const filename = imageUrl.split("/").pop() || "image.png"
        const a = document.createElement("a")
        a.href = blobUrl
        a.download = filename
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(blobUrl)
        toast.success("ダウンロードしました")
    }

    const handleOpenInNewTab = () => {
        window.open(imageUrl, "_blank")
    }

    // Extract metadata
    const prompt = (metadata?.prompt as string) || ""
    const negativePrompt = (metadata?.negative_prompt as string) || ""
    const seed = metadata?.seed as number | undefined
    const steps = metadata?.steps as number | undefined
    const cfgScale = metadata?.cfg_scale as number | undefined
    const width = metadata?.width as number | undefined
    const height = metadata?.height as number | undefined
    const modelName = metadata?.model_name as string | undefined

    const promptTags = prompt ? parsePromptTags(prompt) : []
    const negativeTags = negativePrompt ? parsePromptTags(negativePrompt) : []

    return (
        <Box
            onClick={onClose}
            style={{
                position: "fixed",
                inset: 0,
                backgroundColor: "rgba(0, 0, 0, 0.95)",
                zIndex: 1000,
                cursor: "pointer",
            }}
        >
            {/* Desktop: side-by-side layout, Mobile: stacked */}
            <Grid
                columns={{ initial: "1", md: "1fr 320px" }}
                rows={{ initial: "1fr auto", md: "1fr" }}
                style={{ height: "100%" }}
            >
                {/* Image area */}
                <Flex
                    align="center"
                    justify="center"
                    p="4"
                    onClick={(e) => e.stopPropagation()}
                    style={{ cursor: "default", overflow: "hidden" }}
                >
                    <img
                        src={imageUrl}
                        alt="Preview"
                        style={{
                            maxWidth: "100%",
                            maxHeight: "100%",
                            objectFit: "contain",
                        }}
                    />
                </Flex>

                {/* Metadata panel */}
                <Box
                    onClick={(e) => e.stopPropagation()}
                    style={{
                        backgroundColor: "var(--gray-1)",
                        cursor: "default",
                        maxHeight: "100vh",
                        overflow: "hidden",
                    }}
                >
                    {/* Header with actions */}
                    <Flex
                        justify="between"
                        align="center"
                        p="3"
                        style={{
                            borderBottom: "1px solid var(--gray-4)",
                        }}
                    >
                        <Text size="2" weight="medium">画像情報</Text>
                        <Flex gap="1">
                            <IconButton
                                variant="ghost"
                                size="1"
                                onClick={handleDownload}
                                style={{ cursor: "pointer" }}
                            >
                                <Download size={16} />
                            </IconButton>
                            <IconButton
                                variant="ghost"
                                size="1"
                                onClick={handleOpenInNewTab}
                                style={{ cursor: "pointer" }}
                            >
                                <ExternalLink size={16} />
                            </IconButton>
                            <IconButton
                                variant="ghost"
                                size="1"
                                onClick={onClose}
                                style={{ cursor: "pointer" }}
                            >
                                <X size={16} />
                            </IconButton>
                        </Flex>
                    </Flex>

                    {/* Scrollable content */}
                    <ScrollArea style={{ height: "calc(100vh - 48px)" }}>
                        <Flex direction="column" gap="3" p="3">
                            {/* Prompt tags */}
                            {promptTags.length > 0 && (
                                <Box>
                                    <Text size="1" color="gray" mb="1" style={{ display: "block" }}>
                                        プロンプト
                                    </Text>
                                    <Flex gap="1" wrap="wrap">
                                        {promptTags.map((item, i) => (
                                            <Badge
                                                key={i}
                                                size="1"
                                                variant={item.weight !== 1.0 ? "solid" : "soft"}
                                                color={item.weight > 1.0 ? "violet" : item.weight < 1.0 ? "gray" : undefined}
                                            >
                                                {item.tag}
                                                {item.weight !== 1.0 && (
                                                    <Text size="1" color="gray" ml="1">
                                                        {item.weight.toFixed(2)}
                                                    </Text>
                                                )}
                                            </Badge>
                                        ))}
                                    </Flex>
                                </Box>
                            )}

                            {/* Negative prompt tags */}
                            {negativeTags.length > 0 && (
                                <Box>
                                    <Text size="1" color="gray" mb="1" style={{ display: "block" }}>
                                        ネガティブ
                                    </Text>
                                    <Flex gap="1" wrap="wrap">
                                        {negativeTags.map((item, i) => (
                                            <Badge
                                                key={i}
                                                size="1"
                                                variant="outline"
                                                color="red"
                                            >
                                                {item.tag}
                                                {item.weight !== 1.0 && (
                                                    <Text size="1" color="gray" ml="1">
                                                        {item.weight.toFixed(2)}
                                                    </Text>
                                                )}
                                            </Badge>
                                        ))}
                                    </Flex>
                                </Box>
                            )}

                            {/* Other metadata */}
                            <Flex direction="column" gap="2">
                                {modelName && (
                                    <Flex justify="between">
                                        <Text size="1" color="gray">モデル</Text>
                                        <Text size="1">{modelName}</Text>
                                    </Flex>
                                )}
                                {seed !== undefined && (
                                    <Flex justify="between">
                                        <Text size="1" color="gray">Seed</Text>
                                        <Text size="1">{seed}</Text>
                                    </Flex>
                                )}
                                {steps !== undefined && (
                                    <Flex justify="between">
                                        <Text size="1" color="gray">Steps</Text>
                                        <Text size="1">{steps}</Text>
                                    </Flex>
                                )}
                                {cfgScale !== undefined && (
                                    <Flex justify="between">
                                        <Text size="1" color="gray">CFG Scale</Text>
                                        <Text size="1">{cfgScale}</Text>
                                    </Flex>
                                )}
                                {width && height && (
                                    <Flex justify="between">
                                        <Text size="1" color="gray">サイズ</Text>
                                        <Text size="1">{width} × {height}</Text>
                                    </Flex>
                                )}
                            </Flex>
                        </Flex>
                    </ScrollArea>
                </Box>
            </Grid>
        </Box>
    )
}
