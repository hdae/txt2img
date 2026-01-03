/**
 * CodeMirror theme for Radix UI integration
 */

import { EditorView } from "@codemirror/view"
import { createTheme } from "@uiw/codemirror-themes"

// Radix UI compatible dark theme
export const radixDarkTheme = createTheme({
    theme: "dark",
    settings: {
        background: "var(--gray-2)",
        foreground: "var(--gray-12)",
        caret: "#ffffff",  // Base caret color, will be overridden by thickCaretStyle
        selection: "var(--accent-5)",
        selectionMatch: "var(--accent-4)",
        lineHighlight: "transparent",
        gutterBackground: "var(--gray-2)",
        gutterForeground: "var(--gray-8)",
    },
    styles: [],
})

// Custom caret style: white and visible
export const thickCaretStyle = EditorView.theme({
    "&.cm-focused .cm-cursor": {
        borderLeftWidth: "1px !important",
        borderLeftColor: "#ffffff !important",
    },
    ".cm-cursor": {
        borderLeftWidth: "1px !important",
        borderLeftColor: "#ffffff !important",
    },
})
