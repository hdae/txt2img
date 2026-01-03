/**
 * Tag autocomplete extension for CodeMirror
 */

import {
    autocompletion,
    type CompletionContext,
    type CompletionResult,
} from "@codemirror/autocomplete";

import {
    loadTagDatabase,
    searchTags,
    TAG_CATEGORIES,
} from "./tag-database";

// =============================================================================
// Autocomplete Source
// =============================================================================

/**
 * Get the current word being typed (after last comma or at start)
 */
function getCurrentWord(context: CompletionContext): { from: number; text: string } | null {
    const line = context.state.doc.lineAt(context.pos)
    const lineText = line.text.slice(0, context.pos - line.from)

    // Find the last comma or start of line
    const lastComma = lineText.lastIndexOf(",")
    const wordStart = lastComma === -1 ? 0 : lastComma + 1

    // Get the text after the comma (trimmed)
    const text = lineText.slice(wordStart).trimStart()
    const from = line.from + wordStart + (lineText.slice(wordStart).length - lineText.slice(wordStart).trimStart().length)

    // Only trigger if there's at least 1 character
    if (text.length < 1) return null

    return { from, text }
}

/**
 * Tag completion source for CodeMirror
 */
async function tagCompletionSource(context: CompletionContext): Promise<CompletionResult | null> {
    // Ensure database is loaded
    await loadTagDatabase()

    const word = getCurrentWord(context)
    if (!word) return null

    const results = searchTags(word.text, 30)
    if (results.length === 0) return null

    return {
        from: word.from,
        options: results.map((result) => {
            // Convert underscores to spaces for display
            const displayName = result.tag.name.replace(/_/g, " ")
            return {
                label: result.tag.name,
                displayLabel: displayName,
                detail: result.matchedOn === "alias" ? `← ${result.matchedAlias}` : undefined,
                info: `${result.tag.postCount.toLocaleString()} posts`,
                type: TAG_CATEGORIES[result.tag.category]?.name ?? "general",
                boost: result.tag.postCount / 10000000,
                // Apply with spaces instead of underscores
                apply: displayName,
            }
        }),
        validFor: /^[\w\s_()-]*$/,
    }
}

// =============================================================================
// Extension
// =============================================================================

/**
 * Create tag autocomplete extension for prompt editors
 */
export function tagAutocomplete() {
    return autocompletion({
        override: [tagCompletionSource],
        activateOnTyping: true,
        maxRenderedOptions: 30,
        icons: false,
    })
}
