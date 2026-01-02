/**
 * CodeMirror theme for Radix UI integration
 */

import { createTheme } from "@uiw/codemirror-themes"

// Radix UI compatible dark theme
export const radixDarkTheme = createTheme({
    theme: "dark",
    settings: {
        background: "var(--gray-2)",
        foreground: "var(--gray-12)",
        caret: "var(--accent-9)",
        selection: "var(--accent-5)",
        selectionMatch: "var(--accent-4)",
        lineHighlight: "transparent",
        gutterBackground: "var(--gray-2)",
        gutterForeground: "var(--gray-8)",
    },
    styles: [],
})
