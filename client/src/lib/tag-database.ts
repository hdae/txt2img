/**
 * Tag database for autocomplete
 * Loads and searches danbooru.csv
 */

import { parse } from "@std/csv"

// =============================================================================
// Types
// =============================================================================

/**
 * Tag category types (from Danbooru)
 * @see https://github.com/DominikDoom/a1111-sd-webui-tagcomplete
 */
export type TagCategory = 0 | 1 | 3 | 4 | 5

export const TAG_CATEGORIES = {
    0: { name: "general", color: "var(--blue-9)" },
    1: { name: "artist", color: "var(--red-9)" },
    3: { name: "copyright", color: "var(--purple-9)" },
    4: { name: "character", color: "var(--green-9)" },
    5: { name: "meta", color: "var(--orange-9)" },
} as const

export interface Tag {
    name: string
    category: TagCategory
    postCount: number
    aliases: string[]
}

export interface TagSearchResult {
    tag: Tag
    matchedOn: "name" | "alias"
    matchedAlias?: string
}

// =============================================================================
// Tag Database
// =============================================================================

let tags: Tag[] = []
let isLoaded = false
let loadPromise: Promise<void> | null = null

/**
 * Load tags from CSV file
 */
export async function loadTagDatabase(): Promise<void> {
    if (isLoaded) return
    if (loadPromise) return loadPromise

    loadPromise = (async () => {
        try {
            const response = await fetch("/danbooru.csv")
            if (!response.ok) {
                throw new Error(`Failed to load tags: ${response.status}`)
            }

            const text = await response.text()
            const rows = parse(text)

            tags = rows.map((row) => ({
                name: row[0] ?? "",
                category: (parseInt(row[1] ?? "0", 10) as TagCategory) || 0,
                postCount: parseInt(row[2] ?? "0", 10) || 0,
                aliases: row[3] ? row[3].split(",").map((a) => a.trim()).filter(Boolean) : [],
            }))

            isLoaded = true
            console.log(`Loaded ${tags.length} tags`)
        } catch (error) {
            console.error("Failed to load tag database:", error)
            throw error
        }
    })()

    return loadPromise
}

/**
 * Check if database is loaded
 */
export function isTagDatabaseLoaded(): boolean {
    return isLoaded
}

/**
 * Search tags by prefix
 * @param query Search query (prefix match)
 * @param limit Maximum number of results
 */
export function searchTags(query: string, limit = 20): TagSearchResult[] {
    if (!isLoaded || !query) return []

    const lowerQuery = query.toLowerCase()
    const results: TagSearchResult[] = []

    for (const tag of tags) {
        if (results.length >= limit) break

        // Match on name
        if (tag.name.toLowerCase().startsWith(lowerQuery)) {
            results.push({ tag, matchedOn: "name" })
            continue
        }

        // Match on aliases
        for (const alias of tag.aliases) {
            if (alias.toLowerCase().startsWith(lowerQuery)) {
                results.push({ tag, matchedOn: "alias", matchedAlias: alias })
                break
            }
        }
    }

    // Sort by post count (popularity)
    results.sort((a, b) => b.tag.postCount - a.tag.postCount)

    return results.slice(0, limit)
}

/**
 * Get tag count
 */
export function getTagCount(): number {
    return tags.length
}
