/**
 * Tag Database - Load and search danbooru tags for autocomplete
 *
 * CSV Format: name,type,postCount,"aliases"
 * Type: 0=general, 1=artist, 3=copyright, 4=character, 5=meta
 */

import { parse } from "@std/csv"

// =============================================================================
// Types
// =============================================================================

interface Tag {
    name: string
    type: number
    postCount: number
    aliases: string[]
}

// Type mapping for display
const TAG_TYPE_NAMES: Record<number, string> = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}

// =============================================================================
// Database
// =============================================================================

let tags: Tag[] = []
let isLoaded = false
let loadPromise: Promise<void> | null = null

/**
 * Load tags from CSV file
 */
async function loadTags(): Promise<void> {
    if (isLoaded) return

    const response = await fetch("/danbooru.csv")
    const csvText = await response.text()

    // Parse CSV using @std/csv
    const rows = parse(csvText, {
        skipFirstRow: false,
    })

    tags = rows.map((row) => ({
        name: row[0],
        type: parseInt(row[1], 10) || 0,
        postCount: parseInt(row[2], 10) || 0,
        aliases: row[3] ? row[3].split(",").map((a) => a.trim()) : [],
    }))

    // Sort by post count (descending) for default ordering
    tags.sort((a, b) => b.postCount - a.postCount)

    isLoaded = true
    console.log(`[TagDatabase] Loaded ${tags.length} tags`)
}

/**
 * Ensure tags are loaded (singleton pattern)
 */
export async function ensureTagsLoaded(): Promise<void> {
    if (isLoaded) return
    if (loadPromise) return loadPromise
    loadPromise = loadTags()
    return loadPromise
}

// =============================================================================
// Search
// =============================================================================

interface SearchResult {
    tag: Tag
    matchedBy: "name" | "alias"
    matchedAlias?: string
}

/**
 * Search tags by partial match (name and aliases)
 * Query should be normalized (lowercase, spaces → underscores)
 * @param query - Search query (already normalized)
 * @param limit - Maximum results
 * @returns Matching tags sorted by relevance (post count)
 */
export function searchTags(query: string, limit = 20): SearchResult[] {
    if (!isLoaded || !query) return []

    const q = query.toLowerCase()
    // Also create a space version for matching display names
    const qWithSpaces = q.replace(/_/g, " ")
    const results: SearchResult[] = []

    for (const tag of tags) {
        // Match by name (underscored)
        if (tag.name.toLowerCase().includes(q)) {
            results.push({ tag, matchedBy: "name" })
            if (results.length >= limit) break
            continue
        }

        // Match by alias (may contain spaces or Japanese)
        const matchedAlias = tag.aliases.find((alias) => {
            const aliasLower = alias.toLowerCase()
            // Match both underscore and space versions
            return aliasLower.includes(q) || aliasLower.includes(qWithSpaces)
        })
        if (matchedAlias) {
            results.push({ tag, matchedBy: "alias", matchedAlias })
            if (results.length >= limit) break
        }
    }

    return results
}

/**
 * Get tag type name for display
 */
export function getTagTypeName(type: number): string {
    return TAG_TYPE_NAMES[type] || "unknown"
}

/**
 * Get all loaded tags (for debugging)
 */
export function getTagCount(): number {
    return tags.length
}
