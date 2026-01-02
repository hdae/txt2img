/**
 * ResultView - Display generated image (simplified)
 */

import { Box, Button, Flex } from "@radix-ui/themes"
import { Download, ExternalLink } from "lucide-react"
import toast from "react-hot-toast"

import { getImageUrl } from "@/api/client"

interface ResultViewProps {
    result: {
        imageId: string
        imageUrl: string
        thumbnailUrl: string
    }
    outputFormat: "png" | "webp"
}

export const ResultView = ({ result, outputFormat }: ResultViewProps) => {
    const fullImageUrl = getImageUrl(result.imageId, outputFormat)

    const handleDownload = async () => {
        const response = await fetch(fullImageUrl)
        const blob = await response.blob()
        const blobUrl = URL.createObjectURL(blob)

        const a = document.createElement("a")
        a.href = blobUrl
        a.download = `${result.imageId}.${outputFormat}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(blobUrl)

        toast.success("ダウンロードしました")
    }

    const handleOpenInNewTab = () => {
        window.open(fullImageUrl, "_blank")
    }

    return (
        <Box>
            {/* Actions (above image) */}
            <Flex gap="2" mb="3">
                <Button variant="soft" size="2" onClick={handleDownload} style={{ flex: 1 }}>
                    <Download size={16} />
                    ダウンロード
                </Button>
                <Button variant="soft" size="2" onClick={handleOpenInNewTab} style={{ flex: 1 }}>
                    <ExternalLink size={16} />
                    新しいタブで開く
                </Button>
            </Flex>

            {/* Image - clickable to open in new tab */}
            <Box
                style={{
                    borderRadius: "var(--radius-3)",
                    overflow: "hidden",
                    cursor: "pointer",
                }}
                onClick={handleOpenInNewTab}
            >
                <img
                    src={result.imageUrl}
                    alt="Generated"
                    style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                    }}
                />
            </Box>
        </Box>
    )
}
