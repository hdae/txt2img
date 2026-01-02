/**
 * ImageGrid - Display images in a responsive grid with modal preview
 */

import { useCallback, useState } from "react"

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
}

export const ImageGrid = ({ images, getThumbnailUrl, getFullUrl }: ImageGridProps) => {
    const [selectedIndex, setSelectedIndex] = useState<number | null>(null)

    const selectedImage = selectedIndex !== null ? images[selectedIndex] : null

    const handlePrev = useCallback(() => {
        if (selectedIndex !== null && selectedIndex > 0) {
            setSelectedIndex(selectedIndex - 1)
        }
    }, [selectedIndex])

    const handleNext = useCallback(() => {
        if (selectedIndex !== null && selectedIndex < images.length - 1) {
            setSelectedIndex(selectedIndex + 1)
        }
    }, [selectedIndex, images.length])

    const hasPrev = selectedIndex !== null && selectedIndex > 0
    const hasNext = selectedIndex !== null && selectedIndex < images.length - 1

    return (
        <>
            <Grid
                columns={{ initial: "3", sm: "4", md: "5", lg: "6" }}
                gap="2"
            >
                {images.map((image, index) => (
                    <Box
                        key={image.id}
                        onClick={() => setSelectedIndex(index)}
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
                    onClose={() => setSelectedIndex(null)}
                    onPrev={hasPrev ? handlePrev : undefined}
                    onNext={hasNext ? handleNext : undefined}
                />
            )}
        </>
    )
}
