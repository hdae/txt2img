/**
 * GalleryTab - Display generated images in a grid
 */

import { useEffect, useRef } from "react"

import { Box, Flex, Spinner, Text } from "@radix-ui/themes"
import { useInfiniteQuery } from "@tanstack/react-query"

import { getImages, getImageUrl, getThumbnailUrl } from "@/api/client"
import { connectToGallerySSE } from "@/api/sse"
import type { ImageInfo } from "@/api/types"
import { useServerInfo } from "@/hooks/useServerInfo"

import { ImageGrid } from "./ImageGrid"

export const GalleryTab = () => {
    const { data: serverInfo } = useServerInfo()
    const outputFormat = serverInfo?.output_format || "png"
    const loadMoreRef = useRef<HTMLDivElement>(null)

    const {
        data,
        fetchNextPage,
        hasNextPage,
        isFetchingNextPage,
        isLoading,
        isError,
        refetch,
    } = useInfiniteQuery({
        queryKey: ["gallery"],
        queryFn: async ({ pageParam = 0 }) => {
            return getImages(50, pageParam)
        },
        getNextPageParam: (lastPage) => {
            const nextOffset = lastPage.offset + lastPage.limit
            return nextOffset < lastPage.total ? nextOffset : undefined
        },
        initialPageParam: 0,
    })

    // Subscribe to gallery SSE for real-time updates
    useEffect(() => {
        const cleanup = connectToGallerySSE({
            onMessage: (event: string) => {
                if (event === "new_image") {
                    // Refetch first page to show new image
                    refetch()
                }
            },
        })
        return cleanup
    }, [refetch])

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

    // Flatten all pages into single array
    const images: ImageInfo[] = data?.pages.flatMap((page) => page.images) ?? []

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
                getThumbnailUrl={(id: string) => getThumbnailUrl(id)}
                getFullUrl={(id: string) => getImageUrl(id, outputFormat)}
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
