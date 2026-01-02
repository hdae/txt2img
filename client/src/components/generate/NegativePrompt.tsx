/**
 * NegativePrompt - Collapsible negative prompt input
 */

import { useState } from "react"

import * as Collapsible from "@radix-ui/react-collapsible"
import { Box, Button, Text, TextArea } from "@radix-ui/themes"
import { ChevronDown, ChevronRight } from "lucide-react"

import { useGenerateStore } from "@/stores/generateStore"

export const NegativePrompt = () => {
    const [open, setOpen] = useState(false)
    const negativePrompt = useGenerateStore((state) => state.form.negativePrompt)
    const setNegativePrompt = useGenerateStore((state) => state.setNegativePrompt)

    return (
        <Box>
            <Collapsible.Root open={open} onOpenChange={setOpen}>
                <Collapsible.Trigger asChild>
                    <Button variant="ghost" size="1" style={{ cursor: "pointer" }}>
                        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        <Text size="2" weight="medium">
                            ネガティブプロンプト
                        </Text>
                    </Button>
                </Collapsible.Trigger>
                <Collapsible.Content>
                    <Box pt="2">
                        <TextArea
                            value={negativePrompt}
                            onChange={(e) => setNegativePrompt(e.target.value)}
                            placeholder="避けたい要素..."
                            rows={3}
                            style={{ width: "100%" }}
                        />
                    </Box>
                </Collapsible.Content>
            </Collapsible.Root>
        </Box>
    )
}
