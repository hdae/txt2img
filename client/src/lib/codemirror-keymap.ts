/**
 * Custom CodeMirror keymap for prompt editing
 *
 * Based on CodeMirror 6 default keymaps with customizations for Windows/WSL:
 * - Redo: Ctrl-Y AND Ctrl-Shift-Z
 * - Standard text editing shortcuts
 */

import {
    history,
    redo,
    redoSelection,
    undo,
    undoSelection
} from "@codemirror/commands"
import { keymap, type KeyBinding } from "@codemirror/view"

// ============================================================================
// Default keymap reference (from @codemirror/commands historyKeymap):
// ============================================================================
// Mod-z          → undo
// Mod-y          → redo (Windows/Linux)
// Mod-Shift-z    → redo (macOS)
// Mod-u          → undoSelection
// Alt-u (Mac)    → undoSelection
// Mod-Shift-u    → redoSelection
// ============================================================================

/**
 * Custom history keymap with both Ctrl-Y and Ctrl-Shift-Z for redo
 */
const customHistoryKeymap: readonly KeyBinding[] = [
    { key: "Mod-z", run: undo, preventDefault: true },
    { key: "Mod-y", run: redo, preventDefault: true },
    { key: "Mod-Shift-z", run: redo, preventDefault: true },  // Added for consistency
    { key: "Mod-u", run: undoSelection, preventDefault: true },
    { key: "Mod-Shift-u", run: redoSelection, preventDefault: true },
]

/**
 * Get prompt editor extensions with custom keymap
 */
export function getPromptEditorExtensions() {
    return [
        history(),
        keymap.of(customHistoryKeymap),
    ]
}
