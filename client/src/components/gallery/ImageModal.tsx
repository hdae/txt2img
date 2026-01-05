/**
 * ImageModal - Full-screen modal for image preview with metadata
 */

import { useCallback, useEffect, useRef } from "react"

import { Badge, Box, Flex, Grid, IconButton, ScrollArea, Text } from "@radix-ui/themes"
import { ChevronLeft, ChevronRight, Download, ExternalLink, X } from "lucide-react"
import toast from "react-hot-toast"

interface ImageModalProps {
    imageUrl: string
    metadata: Record<string, unknown>
    onClose: () => void
    onPrev?: () => void
    onNext?: () => void
}

// Parsed tag with weight and nesting info
interface ParsedTag {
    tag: string
    weight: number
    isBreak?: boolean
    depth?: number // Nesting depth for visual indication
}

// Parse prompt tags with proper BREAK handling and nested group support
function parsePromptTags(prompt: string): ParsedTag[] {
    const results: ParsedTag[] = []

    // First, split by BREAK (case insensitive, word boundary)
    const segments = prompt.split(/\bBREAK\b/i)

    segments.forEach((segment, segIndex) => {
        // Add BREAK separator between segments (not before first)
        if (segIndex > 0) {
            results.push({ tag: "BREAK", weight: 1.0, isBreak: true })
        }

        // Split segment by comma or newline
        const parts = segment.split(/[,\n]/).map((p) => p.trim()).filter(Boolean)

        for (const part of parts) {
            // Parse the part with stack-based approach for nested weights
            const parsed = parseWeightedPart(part)
            if (parsed) {
                results.push(parsed)
            }
        }
    })

    return results
}

// Parse a single part that may have weights
function parseWeightedPart(part: string): ParsedTag | null {
    if (!part.trim()) return null

    // Match explicit weight: (text:1.5) or (text:0.8)
    const explicitMatch = part.match(/^\((.+):([\d.]+)\)$/)
    if (explicitMatch) {
        const innerText = explicitMatch[1].trim()
        const weight = parseFloat(explicitMatch[2])
        // Check if inner has more nesting
        const depth = countNestingDepth(innerText)
        return {
            tag: innerText.replace(/[()[\]]/g, "").trim(),
            weight: weight,
            depth: depth > 0 ? depth : undefined,
        }
    }

    // Match nested parentheses: ((word)) = 1.1^n
    const parenMatch = part.match(/^(\(+)(.+?)(\)+)$/)
    if (parenMatch && parenMatch[1].length === parenMatch[3].length) {
        const depth = parenMatch[1].length
        const innerText = parenMatch[2].trim()
        return {
            tag: innerText,
            weight: Math.pow(1.1, depth),
            depth: depth,
        }
    }

    // Match nested brackets: [[word]] = 0.9^n
    const bracketMatch = part.match(/^(\[+)(.+?)(\]+)$/)
    if (bracketMatch && bracketMatch[1].length === bracketMatch[3].length) {
        const depth = bracketMatch[1].length
        const innerText = bracketMatch[2].trim()
        return {
            tag: innerText,
            weight: Math.pow(1 / 1.1, depth),
            depth: depth,
        }
    }

    // Plain tag (strip any remaining brackets)
    const cleanTag = part.replace(/[()[\]]/g, "").trim()
    if (!cleanTag) return null

    return { tag: cleanTag, weight: 1.0 }
}

// Count nesting depth for visual indication
function countNestingDepth(text: string): number {
    let depth = 0
    let maxDepth = 0
    for (const char of text) {
        if (char === "(" || char === "[") {
            depth++
            maxDepth = Math.max(maxDepth, depth)
        } else if (char === ")" || char === "]") {
            depth--
        }
    }
    return maxDepth
}

// Get badge color based on weight (gradual 0.1 steps)
function getWeightColor(weight: number): "violet" | "purple" | "plum" | "gray" | "bronze" | "amber" | undefined {
    if (weight >= 1.5) return "violet"      // Very high
    if (weight >= 1.3) return "purple"      // High
    if (weight >= 1.1) return "plum"        // Slightly high
    if (weight <= 0.5) return "bronze"      // Very low
    if (weight <= 0.7) return "gray"        // Low
    if (weight <= 0.9) return "amber"       // Slightly low
    return undefined                         // Normal (1.0)
}

// Get border style based on nesting depth
function getDepthStyle(depth?: number): React.CSSProperties {
    if (!depth || depth <= 1) return {}
    // Add subtle border to indicate nesting
    const intensity = Math.min(depth, 5) // Cap at 5
    return {
        boxShadow: `inset 0 0 0 ${intensity}px rgba(255, 255, 255, 0.1)`,
    }
}

// Metadata content component (must be outside render)
const MetadataContent = ({
    promptTags,
    negativeTags,
    modelName,
    seed,
    steps,
    cfgScale,
    width,
    height,
}: {
    promptTags: ParsedTag[]
    negativeTags: ParsedTag[]
    modelName?: string
    seed?: number
    steps?: number
    cfgScale?: number
    width?: number
    height?: number
}) => (
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
                            variant={item.isBreak ? "solid" : item.weight !== 1.0 ? "solid" : "soft"}
                            color={item.isBreak ? "orange" : getWeightColor(item.weight)}
                            style={{
                                ...getDepthStyle(item.depth),
                                maxWidth: "200px",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                            }}
                            title={item.weight !== 1.0 ? `${item.tag} (${item.weight.toFixed(2)})` : item.tag}
                        >
                            {item.tag}
                            {item.weight !== 1.0 && !item.isBreak && (
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
                            style={{
                                maxWidth: "200px",
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                whiteSpace: "nowrap",
                            }}
                            title={item.weight !== 1.0 ? `${item.tag} (${item.weight.toFixed(2)})` : item.tag}
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
)

export const ImageModal = ({ imageUrl, metadata, onClose, onPrev, onNext }: ImageModalProps) => {
    const touchStartX = useRef<number | null>(null)

    // Keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                onClose()
            } else if (e.key === "ArrowLeft" && onPrev) {
                onPrev()
            } else if (e.key === "ArrowRight" && onNext) {
                onNext()
            }
        }

        window.addEventListener("keydown", handleKeyDown)
        return () => window.removeEventListener("keydown", handleKeyDown)
    }, [onClose, onPrev, onNext])

    // Swipe handling
    const handleTouchStart = useCallback((e: React.TouchEvent) => {
        touchStartX.current = e.touches[0].clientX
    }, [])

    const handleTouchEnd = useCallback((e: React.TouchEvent) => {
        if (touchStartX.current === null) return

        const touchEndX = e.changedTouches[0].clientX
        const diff = touchStartX.current - touchEndX
        const threshold = 50

        if (diff > threshold && onNext) {
            onNext()
        } else if (diff < -threshold && onPrev) {
            onPrev()
        }

        touchStartX.current = null
    }, [onPrev, onNext])

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

    const metadataProps = {
        promptTags,
        negativeTags,
        modelName,
        seed,
        steps,
        cfgScale,
        width,
        height,
    }

    return (
        <Box
            onTouchStart={handleTouchStart}
            onTouchEnd={handleTouchEnd}
            style={{
                position: "fixed",
                inset: 0,
                backgroundColor: "rgba(0, 0, 0, 0.95)",
                zIndex: 1000,
            }}
        >
            {/* PC Layout: side-by-side */}
            <Box display={{ initial: "none", md: "block" }} style={{ height: "100vh", overflow: "hidden" }}>
                <Grid columns="1fr 320px" style={{ height: "100%", maxHeight: "100vh" }}>
                    {/* Image area with navigation */}
                    <Flex
                        align="center"
                        justify="center"
                        onClick={onClose}
                        style={{
                            cursor: "pointer",
                            overflow: "hidden",
                            position: "relative",
                            maxHeight: "100vh",
                        }}
                    >
                        {/* Prev button */}
                        {onPrev && (
                            <IconButton
                                variant="soft"
                                size="3"
                                onClick={(e) => { e.stopPropagation(); onPrev() }}
                                style={{
                                    position: "absolute",
                                    left: "16px",
                                    cursor: "pointer",
                                }}
                            >
                                <ChevronLeft size={24} />
                            </IconButton>
                        )}

                        <img
                            src={imageUrl}
                            alt="Preview"
                            onClick={(e) => e.stopPropagation()}
                            style={{
                                maxWidth: "100%",
                                maxHeight: "100%",
                                objectFit: "contain",
                                cursor: "default",
                            }}
                        />

                        {/* Next button */}
                        {onNext && (
                            <IconButton
                                variant="soft"
                                size="3"
                                onClick={(e) => { e.stopPropagation(); onNext() }}
                                style={{
                                    position: "absolute",
                                    right: "16px",
                                    cursor: "pointer",
                                }}
                            >
                                <ChevronRight size={24} />
                            </IconButton>
                        )}
                    </Flex>

                    {/* Metadata panel - full height with scroll */}
                    <Flex
                        direction="column"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            backgroundColor: "var(--gray-1)",
                            cursor: "default",
                            height: "100vh",
                            maxHeight: "100vh",
                            overflow: "hidden",
                        }}
                    >
                        {/* Header */}
                        <Flex
                            justify="between"
                            align="center"
                            p="3"
                            style={{ borderBottom: "1px solid var(--gray-4)", flexShrink: 0 }}
                        >
                            <Text size="2" weight="medium">画像情報</Text>
                            <Flex gap="1">
                                <IconButton variant="soft" size="1" onClick={handleDownload}>
                                    <Download size={16} />
                                </IconButton>
                                <IconButton variant="soft" size="1" onClick={handleOpenInNewTab}>
                                    <ExternalLink size={16} />
                                </IconButton>
                                <IconButton variant="soft" size="1" onClick={onClose}>
                                    <X size={16} />
                                </IconButton>
                            </Flex>
                        </Flex>
                        {/* Scrollable content */}
                        <ScrollArea style={{ flex: 1, minHeight: 0 }}>
                            <MetadataContent {...metadataProps} />
                        </ScrollArea>
                    </Flex>
                </Grid>
            </Box>

            {/* Mobile Layout: stacked */}
            <Flex
                direction="column"
                display={{ initial: "flex", md: "none" }}
                style={{ height: "100%" }}
            >
                {/* Image area - takes remaining space */}
                <Flex
                    flexGrow="1"
                    align="center"
                    justify="center"
                    p="4"
                    onClick={onClose}
                    style={{ cursor: "pointer", overflow: "hidden", minHeight: 0 }}
                >
                    <img
                        src={imageUrl}
                        alt="Preview"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            maxWidth: "100%",
                            maxHeight: "100%",
                            objectFit: "contain",
                            cursor: "default",
                        }}
                    />
                </Flex>

                {/* Metadata panel - fixed height with scroll */}
                <Box
                    onClick={(e) => e.stopPropagation()}
                    style={{
                        backgroundColor: "var(--gray-1)",
                        cursor: "default",
                        height: "200px",
                        flexShrink: 0,
                    }}
                >
                    {/* Header */}
                    <Flex
                        justify="between"
                        align="center"
                        p="2"
                        style={{ borderBottom: "1px solid var(--gray-4)" }}
                    >
                        <Text size="2" weight="medium">画像情報</Text>
                        <Flex gap="1">
                            <IconButton variant="soft" size="1" onClick={handleDownload}>
                                <Download size={16} />
                            </IconButton>
                            <IconButton variant="soft" size="1" onClick={handleOpenInNewTab}>
                                <ExternalLink size={16} />
                            </IconButton>
                            <IconButton variant="soft" size="1" onClick={onClose}>
                                <X size={16} />
                            </IconButton>
                        </Flex>
                    </Flex>
                    {/* Scrollable content */}
                    <ScrollArea>
                        <MetadataContent {...metadataProps} />
                    </ScrollArea>
                </Box>
            </Flex>
        </Box>
    )
}
