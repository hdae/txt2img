/**
 * useServerInfo hook - fetches server information using TanStack Query
 */

import { useQuery } from "@tanstack/react-query"

import { getServerInfo } from "@/api/client"
import type { ServerInfo } from "@/api/types"

export const serverInfoQueryKey = ["serverInfo"] as const

export function useServerInfo() {
    return useQuery<ServerInfo>({
        queryKey: serverInfoQueryKey,
        queryFn: getServerInfo,
        staleTime: 1000 * 60 * 5, // 5 minutes
        refetchOnWindowFocus: false,
    })
}
