/**
 * Tag Autocomplete - CodeMirror autocomplete extension for tags
 *
 * Features:
 * - Partial match search (name and aliases)
 * - Auto-insert comma on completion (with underscore → space)
 * - Alias resolution (replace with canonical tag name)
 * - Category-based styling
 * - Japanese input support
 * - Match highlighting (underline)
 */

import {
    autocompletion,
    type Completion,
    type CompletionContext,
    type CompletionResult,
} from "@codemirror/autocomplete";
import { EditorView } from "@codemirror/view";

import { ensureTagsLoaded, getTagTypeName, searchTags } from "./tag-database";

// =============================================================================
// Completion Source
// =============================================================================

/**
 * Get the current tag being typed (text after last comma or start)
 */
function getCurrentTagInput(context: CompletionContext): { from: number; text: string } | null {
    const line = context.state.doc.lineAt(context.pos)
    const lineText = line.text.slice(0, context.pos - line.from)

    // Find last comma or start of line
    const lastComma = lineText.lastIndexOf(",")
    const afterComma = lineText.slice(lastComma + 1)

    // Trim leading spaces
    const trimmedText = afterComma.trimStart()
    const leadingSpaces = afterComma.length - trimmedText.length

    // Don't complete if nothing typed (unless explicit)
    if (trimmedText.length === 0 && !context.explicit) {
        return null
    }

    return {
        from: line.from + lastComma + 1 + leadingSpaces,
        text: trimmedText,
    }
}

/**
 * Normalize search query: trim, lowercase, space → underscore
 */
function normalizeQuery(text: string): string {
    return text.trim().toLowerCase().replace(/\s+/g, "_")
}

/**
 * Find all match positions for highlighting
 * Returns flat array: [start1, end1, start2, end2, ...]
 */
function findMatchPositions(label: string, query: string): number[] {
    const positions: number[] = []
    const labelLower = label.toLowerCase()
    const queryLower = query.toLowerCase().replace(/_/g, " ")

    if (!queryLower) return positions

    let pos = 0
    while (pos < labelLower.length) {
        const idx = labelLower.indexOf(queryLower, pos)
        if (idx === -1) break
        positions.push(idx, idx + queryLower.length)
        pos = idx + queryLower.length
    }

    return positions
}

/**
 * Tag completion source for CodeMirror
 */
async function tagCompletionSource(
    context: CompletionContext
): Promise<CompletionResult | null> {
    // Ensure tags are loaded
    await ensureTagsLoaded()

    const input = getCurrentTagInput(context)
    if (!input) return null

    // Normalize query (space → underscore, lowercase)
    const query = normalizeQuery(input.text)
    if (!query) return null

    const results = searchTags(query, 50)
    if (results.length === 0) return null

    // Store query for getMatch
    const inputText = input.text

    const completions: Completion[] = results.map((result) => {
        const { tag, matchedBy, matchedAlias } = result
        const typeName = getTagTypeName(tag.type)

        // Use space-separated version for display
        const tagWithSpaces = tag.name.replace(/_/g, " ")

        // For alias match: show alias as label, tag name as detail
        // For name match: show tag name as label
        const isAliasMatch = matchedBy === "alias" && matchedAlias
        const label = isAliasMatch ? matchedAlias : tagWithSpaces
        const detail = isAliasMatch ? `→ ${tagWithSpaces}` : undefined

        // Format post count with commas (e.g., 1,234,567)
        const postCountFormatted = tag.postCount.toLocaleString()

        // Check if tag is already used in the document
        const docText = context.state.doc.toString().toLowerCase()
        const tagLower = tagWithSpaces.toLowerCase()
        // Count occurrences (split by comma and check)
        const existingTags = docText.split(",").map(t => t.trim())
        const isAlreadyUsed = existingTags.filter(t => t === tagLower).length > 0

        return {
            // Label is used for display and highlighting
            label,
            detail,
            type: typeName,
            // Info popup shown when completion is selected
            info: () => {
                const div = document.createElement("div")
                div.style.padding = "4px 8px"
                div.style.fontSize = "12px"
                div.style.whiteSpace = "nowrap"

                if (isAlreadyUsed) {
                    div.innerHTML = `
                        <div style="color: #f80; font-weight: bold;">⚠ 既に使用しています</div>
                        <div style="margin-top: 2px;">${typeName} / ${postCountFormatted}</div>
                    `
                } else {
                    div.textContent = `${typeName} / ${postCountFormatted}`
                }
                return div
            },
            // Always insert the canonical tag name (with spaces) + comma + space
            apply: (view: EditorView, _completion: Completion, from: number, to: number) => {
                const insert = `${tagWithSpaces}, `
                view.dispatch({
                    changes: { from, to, insert },
                    selection: { anchor: from + insert.length },
                })
            },
            // Boost by post count (normalize to -99 to 99 range)
            boost: Math.min(99, Math.floor(Math.log10(tag.postCount + 1) * 10)),
        }
    })

    return {
        from: input.from,
        options: completions,
        // Don't use default filtering - we do our own search
        filter: false,
        // Provide match positions for highlighting
        getMatch: (completion: Completion) => {
            return findMatchPositions(completion.label, inputText)
        },
    }
}

// =============================================================================
// Extension
// =============================================================================

/**
 * Create tag autocomplete extension for CodeMirror
 */
export function tagAutocomplete() {
    return autocompletion({
        override: [tagCompletionSource],
        activateOnTyping: true,
        maxRenderedOptions: 30,
        activateOnTypingDelay: 50,
    })
}

// =============================================================================
// Styles
// =============================================================================

/**
 * Tag category colors (matching Danbooru colors)
 */
export const tagCategoryStyles = EditorView.baseTheme({
    // General (type 0) - blue
    ".cm-completionIcon-general": {
        "&:after": { content: "'●'", color: "#0075f8" },
    },
    // Artist (type 1) - red
    ".cm-completionIcon-artist": {
        "&:after": { content: "'●'", color: "#c00" },
    },
    // Copyright (type 3) - purple
    ".cm-completionIcon-copyright": {
        "&:after": { content: "'●'", color: "#a0a" },
    },
    // Character (type 4) - green
    ".cm-completionIcon-character": {
        "&:after": { content: "'●'", color: "#0a0" },
    },
    // Meta (type 5) - orange
    ".cm-completionIcon-meta": {
        "&:after": { content: "'●'", color: "#f80" },
    },
    // Match highlighting - underline matched text
    ".cm-completionMatchedText": {
        textDecoration: "underline",
    },
})
