/**
 * NegativePrompt - CodeMirror-based negative prompt editor
 */

import { Box, Text } from "@radix-ui/themes"
import CodeMirror from "@uiw/react-codemirror"

import { useGenerateStore } from "@/stores/generateStore"

export const NegativePrompt = () => {
    const negativePrompt = useGenerateStore((state) => state.form.negativePrompt)
    const setNegativePrompt = useGenerateStore((state) => state.setNegativePrompt)

    return (
        <Box>
            <Text as="label" size="2" weight="medium" mb="1" style={{ display: "block" }}>
                ネガティブプロンプト
            </Text>
            <CodeMirror
                value={negativePrompt}
                onChange={setNegativePrompt}
                height="80px"
                placeholder="避けたい要素..."
                theme="dark"
                basicSetup={{
                    lineNumbers: false,
                    foldGutter: false,
                    highlightActiveLine: false,
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
