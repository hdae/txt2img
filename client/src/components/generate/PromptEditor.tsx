/**
 * PromptEditor - CodeMirror-based prompt editor with Radix UI theme and tag autocomplete
 */

import { EditorView } from "@codemirror/view"
import { Box, Flex, Text } from "@radix-ui/themes"
import CodeMirror from "@uiw/react-codemirror"
import { useEffect, useMemo } from "react"

import { FieldResetButton } from "@/components/generate/FieldResetButton"
import { useGenerate } from "@/hooks/useGenerate"
import { useServerInfo } from "@/hooks/useServerInfo"
import { getPromptEditorExtensions, setOnGenerateCallback } from "@/lib/codemirror-keymap"
import { radixDarkTheme, thickCaretStyle } from "@/lib/codemirror-theme"
import { tagAutocomplete, tagCategoryStyles } from "@/lib/tag-autocomplete"
import { useGenerateStore } from "@/stores/generateStore"

export const PromptEditor = () => {
    const prompt = useGenerateStore((state) => state.form.prompt)
    const setPrompt = useGenerateStore((state) => state.setPrompt)
    const { data: serverInfo } = useServerInfo()
    const { generate } = useGenerate()

    // Check if prompt style is tags (enable autocomplete)
    const isTagMode = serverInfo?.parameter_schema?.prompt_style === "tags"

    // Set up Ctrl+Enter callback for generate
    useEffect(() => {
        setOnGenerateCallback(() => {
            generate()
        })
        return () => setOnGenerateCallback(() => { })
    }, [generate])

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
                    プロンプト
                </Text>
                <FieldResetButton fieldName="プロンプト" onReset={() => setPrompt("")} />
            </Flex>
            <CodeMirror
                value={prompt}
                onChange={setPrompt}
                minHeight="80px"
                placeholder="プロンプトを入力..."
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
