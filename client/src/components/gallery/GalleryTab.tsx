/**
 * GalleryTab - Display generated images in a grid
 * Uses GalleryContext for state management
 */

import { useEffect, useRef } from "react"

import { Box, Flex, Spinner, Text } from "@radix-ui/themes"

import { useGalleryContext } from "@/contexts/GalleryContext"

import { ImageGrid } from "./ImageGrid"

export const GalleryTab = () => {
    const {
        images,
        isLoading,
        isError,
        isFetchingNextPage,
        hasNextPage,
        fetchNextPage,
        getThumbnailUrl,
        getFullUrl,
    } = useGalleryContext()

    const loadMoreRef = useRef<HTMLDivElement>(null)

    // Auto-load more when scrolling to bottom
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
                    fetchNextPage()
                }
            },
            { threshold: 0.1 }
        )

        const el = loadMoreRef.current
        if (el) {
            observer.observe(el)
        }

        return () => {
            if (el) {
                observer.unobserve(el)
            }
        }
    }, [hasNextPage, isFetchingNextPage, fetchNextPage])

    if (isLoading) {
        return (
            <Flex align="center" justify="center" py="9">
                <Spinner size="3" />
            </Flex>
        )
    }

    if (isError) {
        return (
            <Box py="6">
                <Text color="red">画像の読み込みに失敗しました</Text>
            </Box>
        )
    }

    if (images.length === 0) {
        return (
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
                    生成された画像がまだありません
                </Text>
            </Box>
        )
    }

    return (
        <Flex direction="column" gap="4">
            <ImageGrid
                images={images}
                getThumbnailUrl={getThumbnailUrl}
                getFullUrl={getFullUrl}
                onReachEnd={hasNextPage && !isFetchingNextPage ? fetchNextPage : undefined}
            />

            {/* Intersection observer target for auto-loading */}
            <div ref={loadMoreRef} style={{ height: "1px" }} />

            {isFetchingNextPage && (
                <Flex justify="center" py="4">
                    <Spinner size="2" />
                </Flex>
            )}
        </Flex>
    )
}
