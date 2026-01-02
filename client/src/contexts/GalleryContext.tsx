/**
 * GalleryContext - Global gallery state management
 *
 * Manages gallery SSE connection and query cache at app level,
 * preventing re-fetches when switching tabs.
 */

import { createContext, useContext, useEffect, useRef, type ReactNode } from "react"

import { useInfiniteQuery } from "@tanstack/react-query"

import { getImages, getImageUrl, getThumbnailUrl } from "@/api/client"
import { connectToGallerySSE } from "@/api/sse"
import type { ImageInfo } from "@/api/types"
import { useServerInfo } from "@/hooks/useServerInfo"

// =============================================================================
// Types
// =============================================================================

interface GalleryContextValue {
    images: ImageInfo[]
    isLoading: boolean
    isError: boolean
    isFetchingNextPage: boolean
    hasNextPage: boolean
    fetchNextPage: () => void
    getThumbnailUrl: (id: string) => string
    getFullUrl: (id: string) => string
}

// =============================================================================
// Context
// =============================================================================

const GalleryContext = createContext<GalleryContextValue | null>(null)

// =============================================================================
// Provider
// =============================================================================

interface GalleryProviderProps {
    children: ReactNode
}

export function GalleryProvider({ children }: GalleryProviderProps) {
    const sseCleanupRef = useRef<(() => void) | null>(null)

    const { data: serverInfo } = useServerInfo()
    const outputFormat = serverInfo?.output_format || "png"

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

    // Subscribe to gallery SSE for real-time updates (persistent)
    useEffect(() => {
        sseCleanupRef.current = connectToGallerySSE({
            onMessage: (event: string) => {
                if (event === "new_image") {
                    refetch()
                }
            },
        })

        return () => {
            sseCleanupRef.current?.()
        }
    }, [refetch])

    // Flatten all pages into single array
    const images: ImageInfo[] = data?.pages.flatMap((page) => page.images) ?? []

    const value: GalleryContextValue = {
        images,
        isLoading,
        isError,
        isFetchingNextPage,
        hasNextPage: hasNextPage ?? false,
        fetchNextPage,
        getThumbnailUrl: (id: string) => getThumbnailUrl(id),
        getFullUrl: (id: string) => getImageUrl(id, outputFormat),
    }

    return (
        <GalleryContext.Provider value={value}>
            {children}
        </GalleryContext.Provider>
    )
}

// =============================================================================
// Hook
// =============================================================================

// eslint-disable-next-line react-refresh/only-export-components
export function useGalleryContext() {
    const context = useContext(GalleryContext)
    if (!context) {
        throw new Error("useGalleryContext must be used within a GalleryProvider")
    }
    return context
}
