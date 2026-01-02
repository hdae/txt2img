/**
 * PromptEditor - CodeMirror-based prompt editor with LPW syntax highlighting
 */

import CodeMirror from "@uiw/react-codemirror"

import { useGenerateStore } from "@/stores/generateStore"

export const PromptEditor = () => {
    const prompt = useGenerateStore((state) => state.form.prompt)
    const setPrompt = useGenerateStore((state) => state.setPrompt)

    return (
        <CodeMirror
            value={prompt}
            onChange={setPrompt}
            height="120px"
            placeholder="プロンプトを入力..."
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
    )
}
