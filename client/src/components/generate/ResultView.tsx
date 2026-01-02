/**
 * ResultView - Display generated image result
 */

import { Box, Button, Card, Flex, Text } from "@radix-ui/themes"
import { Check, Copy, Download } from "lucide-react"
import { useState } from "react"
import toast from "react-hot-toast"

import { getImageUrl } from "@/api/client"
import { useGenerateStore } from "@/stores/generateStore"

export const ResultView = () => {
    const job = useGenerateStore((state) => state.job)
    const form = useGenerateStore((state) => state.form)
    const [copied, setCopied] = useState(false)

    if (!job.result) return null

    const handleDownload = async (format: "png" | "webp") => {
        const url = getImageUrl(job.result!.imageId, format)
        const response = await fetch(url)
        const blob = await response.blob()
        const blobUrl = URL.createObjectURL(blob)

        const a = document.createElement("a")
        a.href = blobUrl
        a.download = `${job.result!.imageId}.${format}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(blobUrl)

        toast.success(`${format.toUpperCase()}でダウンロードしました`)
    }

    const handleCopyPrompt = async () => {
        await navigator.clipboard.writeText(form.prompt)
        setCopied(true)
        toast.success("プロンプトをコピーしました")
        setTimeout(() => setCopied(false), 2000)
    }

    return (
        <Card size="2">
            <Flex direction="column" gap="3">
                {/* Image */}
                <Box
                    style={{
                        borderRadius: "var(--radius-2)",
                        overflow: "hidden",
                    }}
                >
                    <img
                        src={job.result.imageUrl}
                        alt="Generated"
                        style={{
                            width: "100%",
                            height: "auto",
                            display: "block",
                        }}
                    />
                </Box>

                {/* Actions */}
                <Flex gap="2" wrap="wrap">
                    <Button variant="soft" size="2" onClick={() => handleDownload("png")}>
                        <Download size={16} />
                        PNG
                    </Button>
                    <Button variant="soft" size="2" onClick={() => handleDownload("webp")}>
                        <Download size={16} />
                        WebP
                    </Button>
                    <Button variant="ghost" size="2" onClick={handleCopyPrompt}>
                        {copied ? <Check size={16} /> : <Copy size={16} />}
                        プロンプト
                    </Button>
                </Flex>

                {/* Metadata */}
                <Box>
                    <Text size="1" color="gray">
                        サイズ: {form.width}×{form.height}
                        {form.seed !== null && ` | シード: ${form.seed}`}
                    </Text>
                </Box>
            </Flex>
        </Card>
    )
}
