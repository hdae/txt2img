/**
 * PromptEditor - CodeMirror-based prompt editor with Radix UI theme
 */

import { EditorView } from "@codemirror/view"
import { Box, Text } from "@radix-ui/themes"
import CodeMirror from "@uiw/react-codemirror"

import { getPromptEditorExtensions } from "@/lib/codemirror-keymap"
import { radixDarkTheme } from "@/lib/codemirror-theme"
import { tagAutocomplete } from "@/lib/tag-autocomplete"
import { useGenerateStore } from "@/stores/generateStore"

export const PromptEditor = () => {
    const prompt = useGenerateStore((state) => state.form.prompt)
    const setPrompt = useGenerateStore((state) => state.setPrompt)

    return (
        <Box>
            <Text as="label" size="2" weight="medium" mb="1" style={{ display: "block" }}>
                プロンプト
            </Text>
            <CodeMirror
                value={prompt}
                onChange={setPrompt}
                height="120px"
                placeholder="プロンプトを入力..."
                theme={radixDarkTheme}
                extensions={[
                    EditorView.lineWrapping,
                    ...getPromptEditorExtensions(),
                    tagAutocomplete(),
                ]}
                basicSetup={{
                    lineNumbers: false,
                    foldGutter: false,
                    highlightActiveLine: false,
                    history: false,
                    autocompletion: false,  // Use custom autocomplete
                }}
                style={{
                    fontSize: 14,
                    borderRadius: "var(--radius-2)",
                    border: "1px solid var(--gray-6)",
                    overflow: "hidden",
                }}
            />
        </Box>
    )
}
