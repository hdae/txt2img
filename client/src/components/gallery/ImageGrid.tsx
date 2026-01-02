/**
 * ImageGrid - Display images in a responsive grid with modal preview
 */

import { useCallback, useMemo, useState } from "react"

import { Box, Grid } from "@radix-ui/themes"

import { ImageModal } from "./ImageModal"

interface ImageGridProps {
    images: {
        id: string
        thumbnail_url: string
        full_url: string
        metadata: Record<string, unknown>
    }[]
    getThumbnailUrl: (id: string) => string
    getFullUrl: (id: string) => string
    onReachEnd?: () => void
}

export const ImageGrid = ({ images, getThumbnailUrl, getFullUrl, onReachEnd }: ImageGridProps) => {
    const [selectedImageId, setSelectedImageId] = useState<string | null>(null)

    // Build ID to index map for navigation
    const idToIndex = useMemo(() => {
        const map = new Map<string, number>()
        images.forEach((img, idx) => map.set(img.id, idx))
        return map
    }, [images])

    const selectedIndex = selectedImageId !== null ? idToIndex.get(selectedImageId) ?? null : null
    const selectedImage = selectedIndex !== null ? images[selectedIndex] : null

    const handlePrev = useCallback(() => {
        if (selectedIndex !== null && selectedIndex > 0) {
            setSelectedImageId(images[selectedIndex - 1].id)
        }
    }, [selectedIndex, images])

    const handleNext = useCallback(() => {
        if (selectedIndex !== null && selectedIndex < images.length - 1) {
            setSelectedImageId(images[selectedIndex + 1].id)
        } else if (selectedIndex !== null && selectedIndex === images.length - 1 && onReachEnd) {
            // At the end, fetch more
            onReachEnd()
        }
    }, [selectedIndex, images, onReachEnd])

    const hasPrev = selectedIndex !== null && selectedIndex > 0
    // Always show next if onReachEnd is available (more can be loaded)
    const hasNext = selectedIndex !== null && (selectedIndex < images.length - 1 || !!onReachEnd)

    return (
        <>
            <Grid
                columns={{ initial: "3", sm: "4", md: "5", lg: "6" }}
                gap="2"
            >
                {images.map((image) => (
                    <Box
                        key={image.id}
                        onClick={() => setSelectedImageId(image.id)}
                        style={{
                            aspectRatio: "1 / 1",
                            borderRadius: "var(--radius-2)",
                            overflow: "hidden",
                            cursor: "pointer",
                            backgroundColor: "var(--gray-3)",
                        }}
                    >
                        <img
                            src={getThumbnailUrl(image.id)}
                            alt=""
                            loading="lazy"
                            style={{
                                width: "100%",
                                height: "100%",
                                objectFit: "cover",
                                transition: "transform 0.2s ease",
                            }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.transform = "scale(1.05)"
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.transform = "scale(1)"
                            }}
                        />
                    </Box>
                ))}
            </Grid>

            {/* Modal */}
            {selectedImage && (
                <ImageModal
                    imageUrl={getFullUrl(selectedImage.id)}
                    metadata={selectedImage.metadata}
                    onClose={() => setSelectedImageId(null)}
                    onPrev={hasPrev ? handlePrev : undefined}
                    onNext={hasNext ? handleNext : undefined}
                />
            )}
        </>
    )
}
