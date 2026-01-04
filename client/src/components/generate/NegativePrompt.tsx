/**
 * NegativePrompt - CodeMirror-based negative prompt editor with Radix UI theme and tag autocomplete
 */

import { EditorView } from "@codemirror/view"
import { Box, Flex, Text } from "@radix-ui/themes"
import CodeMirror from "@uiw/react-codemirror"
import { useMemo } from "react"

import { FieldResetButton } from "@/components/generate/FieldResetButton"
import { useServerInfo } from "@/hooks/useServerInfo"
import { getPromptEditorExtensions } from "@/lib/codemirror-keymap"
import { radixDarkTheme, thickCaretStyle } from "@/lib/codemirror-theme"
import { tagAutocomplete, tagCategoryStyles } from "@/lib/tag-autocomplete"
import { useGenerateStore } from "@/stores/generateStore"

export const NegativePrompt = () => {
    const negativePrompt = useGenerateStore((state) => state.form.negativePrompt)
    const setNegativePrompt = useGenerateStore((state) => state.setNegativePrompt)
    const { data: serverInfo } = useServerInfo()

    // Check if prompt style is tags (enable autocomplete)
    const isTagMode = serverInfo?.parameter_schema?.prompt_style === "tags"

    // Build extensions based on prompt style
    const extensions = useMemo(() => {
        const base = [
            EditorView.lineWrapping,
            thickCaretStyle,
            ...getPromptEditorExtensions(),
        ]

        if (isTagMode) {
            base.push(tagAutocomplete(), tagCategoryStyles)
        }

        return base
    }, [isTagMode])

    return (
        <Box>
            <Flex align="center" gap="2" mb="1">
                <Text as="label" size="2" weight="medium">
                    ネガティブプロンプト
                </Text>
                <FieldResetButton fieldName="ネガティブプロンプト" onReset={() => setNegativePrompt("")} />
            </Flex>
            <CodeMirror
                value={negativePrompt}
                onChange={setNegativePrompt}
                minHeight="60px"
                placeholder="避けたい要素..."
                theme={radixDarkTheme}
                extensions={extensions}
                basicSetup={{
                    lineNumbers: false,
                    foldGutter: false,
                    highlightActiveLine: false,
                    history: false,
                    autocompletion: false,
                    indentOnInput: false,
                }}
                style={{
                    fontSize: 15,
                    borderRadius: "var(--radius-2)",
                    border: "1px solid var(--gray-6)",
                    overflow: "hidden",
                }}
            />
        </Box>
    )
}
