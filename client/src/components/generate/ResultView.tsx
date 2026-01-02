/**
 * ResultView - Display generated image with modal preview
 */

import { useState } from "react"

import { Box, Button, Flex, IconButton } from "@radix-ui/themes"
import { Download, ExternalLink, X } from "lucide-react"
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
    const [isModalOpen, setIsModalOpen] = useState(false)

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

    const openModal = () => setIsModalOpen(true)
    const closeModal = () => setIsModalOpen(false)

    return (
        <>
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

                {/* Image - clickable to open modal */}
                <Box
                    style={{
                        borderRadius: "var(--radius-3)",
                        overflow: "hidden",
                        cursor: "pointer",
                    }}
                    onClick={openModal}
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

            {/* Modal Overlay */}
            {isModalOpen && (
                <Box
                    onClick={closeModal}
                    style={{
                        position: "fixed",
                        inset: 0,
                        backgroundColor: "rgba(0, 0, 0, 0.9)",
                        zIndex: 1000,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        cursor: "pointer",
                    }}
                >
                    {/* Close button */}
                    <IconButton
                        onClick={(e) => {
                            e.stopPropagation()
                            closeModal()
                        }}
                        variant="ghost"
                        color="gray"
                        highContrast
                        size="3"
                        style={{
                            position: "absolute",
                            top: 16,
                            right: 16,
                            cursor: "pointer",
                        }}
                    >
                        <X size={24} />
                    </IconButton>

                    {/* Full size image */}
                    <img
                        src={fullImageUrl}
                        alt="Generated (Full)"
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            maxWidth: "100vw",
                            maxHeight: "100vh",
                            objectFit: "contain",
                            cursor: "default",
                        }}
                    />
                </Box>
            )}
        </>
    )
}
