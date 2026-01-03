/**
 * Custom CodeMirror keymap for prompt editing
 *
 * Based on CodeMirror 6 default keymaps with customizations:
 * - Redo: Ctrl-Y AND Ctrl-Shift-Z
 * - Tab: Accept completion (instead of indent)
 * - Ctrl-Enter: Trigger external generate action
 * - Ctrl-Up/Down: Adjust tag weight
 */

import { acceptCompletion } from "@codemirror/autocomplete"
import {
    history,
    redo,
    redoSelection,
    undo,
    undoSelection
} from "@codemirror/commands"
import { Prec } from "@codemirror/state"
import { EditorView, keymap, type KeyBinding } from "@codemirror/view"

// ============================================================================
// Custom Commands
// ============================================================================

/**
 * Callback for Ctrl+Enter (generate trigger)
 */
let onGenerateCallback: (() => void) | null = null

export function setOnGenerateCallback(callback: () => void) {
    onGenerateCallback = callback
}

/**
 * Command: Trigger generate action
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function triggerGenerate(_view: EditorView): boolean {
    console.log("[KeyMap] triggerGenerate called, callback:", onGenerateCallback)
    if (onGenerateCallback) {
        onGenerateCallback()
        return true
    }
    return false
}

// ============================================================================
// Tag Weight Adjustment
// ============================================================================

/**
 * Find complete tag boundaries around a position in line text
 */
function findTagBoundaries(lineText: string, pos: number): { start: number; end: number } {
    // Clamp position
    pos = Math.max(0, Math.min(pos, lineText.length))

    // If at comma or end, look at preceding tag
    if (pos > 0 && (lineText[pos] === "," || pos === lineText.length)) {
        pos = pos - 1
    }

    // Find start (after previous comma or start of line)
    let start = lineText.lastIndexOf(",", pos)
    start = start === -1 ? 0 : start + 1

    // Find end (at next comma or end of line)
    let end = lineText.indexOf(",", pos)
    end = end === -1 ? lineText.length : end

    return { start, end }
}

/**
 * Expand selection to complete tag boundaries for multi-tag selection
 */
function expandToTagBoundaries(lineText: string, from: number, to: number): { start: number; end: number } {
    // Find the start of the first tag
    let start = lineText.lastIndexOf(",", from)
    start = start === -1 ? 0 : start + 1

    // Find the end of the last tag
    let end = lineText.indexOf(",", to - 1)
    end = end === -1 ? lineText.length : end

    return { start, end }
}

/**
 * Escape unescaped parentheses in tag text
 * Only escapes ( and ) that are not already escaped with \
 */
function escapeParentheses(text: string): string {
    // Replace unescaped ( and ) only
    // Negative lookbehind: (?<!\\) matches positions not preceded by \
    return text
        .replace(/(?<!\\)\(/g, "\\(")
        .replace(/(?<!\\)\)/g, "\\)")
}

/**
 * Apply weight adjustment to text
 */
function applyWeight(text: string, delta: number): string {
    text = text.trim()
    if (!text) return text

    // Parse existing weight: (content:weight)
    const weightMatch = text.match(/^\((.+):(-?\d+\.?\d*)\)$/)

    if (weightMatch) {
        const innerContent = weightMatch[1]
        const weight = parseFloat(weightMatch[2])
        const newWeight = weight + delta

        // Skip 1.0 (equivalent to no weight)
        if (Math.abs(newWeight - 1.0) < 0.05) {
            return innerContent
        }

        return `(${innerContent}:${newWeight.toFixed(1)})`
    } else {
        // No weight yet
        const targetWeight = 1.0 + delta

        // If target is 1.0, no change needed
        if (Math.abs(targetWeight - 1.0) < 0.05) {
            return text
        }

        // Escape unescaped parentheses
        const escapedText = escapeParentheses(text)
        return `(${escapedText}:${targetWeight.toFixed(1)})`
    }
}

/**
 * Command: Adjust tag weight(s) based on selection
 *
 * 3 patterns:
 * 1. No selection → Expand to current tag
 * 2. Selection within single tag (no comma) → Apply to selection as-is
 * 3. Selection spans multiple tags → Expand to complete tag boundaries and group
 */
function changeTagWeight(view: EditorView, delta: number): boolean {
    const { state, dispatch } = view
    const { from, to } = state.selection.main

    // Get line info
    const line = state.doc.lineAt(from)
    const lineText = line.text
    const fromInLine = from - line.from
    const toInLine = to - line.from

    let targetFrom: number
    let targetTo: number
    let targetText: string

    if (from === to) {
        // Pattern 1: No selection → Expand to current tag
        const bounds = findTagBoundaries(lineText, fromInLine)
        targetFrom = line.from + bounds.start
        targetTo = line.from + bounds.end
        targetText = lineText.slice(bounds.start, bounds.end).trim()
    } else {
        // Check if selection contains comma
        const selectedText = lineText.slice(fromInLine, toInLine)
        const hasComma = selectedText.includes(",")

        if (hasComma) {
            // Pattern 3: Multiple tags → Expand to complete boundaries
            const bounds = expandToTagBoundaries(lineText, fromInLine, toInLine)
            targetFrom = line.from + bounds.start
            targetTo = line.from + bounds.end
            targetText = lineText.slice(bounds.start, bounds.end).trim()
        } else {
            // Pattern 2: Single tag selection → Use as-is
            targetFrom = from
            targetTo = to
            targetText = selectedText.trim()
        }
    }

    if (!targetText) return false

    // Apply weight adjustment
    const newText = applyWeight(targetText, delta)

    // Preserve leading whitespace from original
    const originalSlice = state.sliceDoc(targetFrom, targetTo)
    const leadingSpace = originalSlice.match(/^\s*/)?.[0] || ""

    dispatch({
        changes: { from: targetFrom, to: targetTo, insert: leadingSpace + newText },
        selection: { anchor: targetFrom + leadingSpace.length, head: targetFrom + leadingSpace.length + newText.length },
    })

    return true
}

/**
 * Command: Increase tag weight
 */
function increaseTagWeight(view: EditorView): boolean {
    return changeTagWeight(view, 0.1)
}

/**
 * Command: Decrease tag weight
 */
function decreaseTagWeight(view: EditorView): boolean {
    return changeTagWeight(view, -0.1)
}

// ============================================================================
// Keymaps
// ============================================================================

/**
 * Custom history keymap with both Ctrl-Y and Ctrl-Shift-Z for redo
 */
const customHistoryKeymap: readonly KeyBinding[] = [
    { key: "Mod-z", run: undo, preventDefault: true },
    { key: "Mod-y", run: redo, preventDefault: true },
    { key: "Mod-Shift-z", run: redo, preventDefault: true },
    { key: "Mod-u", run: undoSelection, preventDefault: true },
    { key: "Mod-Shift-u", run: redoSelection, preventDefault: true },
]

/**
 * Custom prompt editing keymap
 */
const promptKeymap: readonly KeyBinding[] = [
    // Tab accepts completion (instead of indent)
    { key: "Tab", run: acceptCompletion },
    // Ctrl+Enter triggers generate
    { key: "Mod-Enter", run: triggerGenerate, preventDefault: true },
    // Ctrl+Up/Down adjusts tag weight
    { key: "Mod-ArrowUp", run: increaseTagWeight, preventDefault: true },
    { key: "Mod-ArrowDown", run: decreaseTagWeight, preventDefault: true },
]

/**
 * Get prompt editor extensions with custom keymap
 */
export function getPromptEditorExtensions() {
    return [
        history(),
        // Use highest priority to ensure our keybindings take precedence
        Prec.highest(keymap.of(promptKeymap)),
        keymap.of(customHistoryKeymap),
    ]
}
