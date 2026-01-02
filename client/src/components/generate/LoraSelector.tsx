/**
 * LoraSelector - LoRA selection and weight adjustment
 */

import { Badge, Box, Card, Flex, IconButton, Select, Slider, Text } from "@radix-ui/themes"
import { X } from "lucide-react"

import type { LoraInfo } from "@/api/types"
import { useGenerateStore } from "@/stores/generateStore"

interface LoraSelectorProps {
    loras: LoraInfo[]
}

export const LoraSelector = ({ loras }: LoraSelectorProps) => {
    const selectedLoras = useGenerateStore((state) => state.form.loras)
    const addLora = useGenerateStore((state) => state.addLora)
    const removeLora = useGenerateStore((state) => state.removeLora)
    const updateLoraWeight = useGenerateStore((state) => state.updateLoraWeight)
    const updateLoraTriggerWeight = useGenerateStore((state) => state.updateLoraTriggerWeight)

    const availableLoras = loras.filter(
        (lora) => !selectedLoras.find((selected) => selected.id === lora.id)
    )

    const handleAddLora = (loraId: string) => {
        const lora = loras.find((l) => l.id === loraId)
        if (lora) {
            addLora({
                id: lora.id,
                weight: lora.weight,
                trigger_weight: lora.trigger_weight,
            })
        }
    }

    return (
        <Box>
            <Flex justify="between" align="center" mb="2">
                <Text as="label" size="2" weight="medium">
                    LoRA
                </Text>
                {availableLoras.length > 0 && (
                    <Select.Root onValueChange={handleAddLora} size="1">
                        <Select.Trigger placeholder="LoRAを追加..." variant="soft" />
                        <Select.Content>
                            {availableLoras.map((lora) => (
                                <Select.Item key={lora.id} value={lora.id}>
                                    {lora.name}
                                </Select.Item>
                            ))}
                        </Select.Content>
                    </Select.Root>
                )}
            </Flex>

            <Flex direction="column" gap="2">
                {selectedLoras.map((selected) => {
                    const loraInfo = loras.find((l) => l.id === selected.id)
                    if (!loraInfo) return null

                    return (
                        <Card key={selected.id} size="1">
                            <Flex direction="column" gap="2">
                                <Flex justify="between" align="center">
                                    <Flex gap="2" align="center">
                                        <Text size="2" weight="medium">
                                            {loraInfo.name}
                                        </Text>
                                        {loraInfo.trigger_words.length > 0 && (
                                            <Badge size="1" variant="soft">
                                                {loraInfo.trigger_words[0]}
                                            </Badge>
                                        )}
                                    </Flex>
                                    <IconButton
                                        size="1"
                                        variant="ghost"
                                        color="gray"
                                        onClick={() => removeLora(selected.id)}
                                    >
                                        <X size={14} />
                                    </IconButton>
                                </Flex>

                                <Flex gap="4">
                                    <Box style={{ flex: 1 }}>
                                        <Flex justify="between" mb="1">
                                            <Text size="1" color="gray">Weight</Text>
                                            <Text size="1" color="gray">{(selected.weight ?? 1.0).toFixed(1)}</Text>
                                        </Flex>
                                        <Slider
                                            value={[selected.weight ?? 1.0]}
                                            onValueChange={([v]) => updateLoraWeight(selected.id, v)}
                                            min={0}
                                            max={2}
                                            step={0.1}
                                            size="1"
                                        />
                                    </Box>
                                    <Box style={{ flex: 1 }}>
                                        <Flex justify="between" mb="1">
                                            <Text size="1" color="gray">Trigger</Text>
                                            <Text size="1" color="gray">{(selected.trigger_weight ?? 0.5).toFixed(1)}</Text>
                                        </Flex>
                                        <Slider
                                            value={[selected.trigger_weight ?? 0.5]}
                                            onValueChange={([v]) => updateLoraTriggerWeight(selected.id, v)}
                                            min={0}
                                            max={2}
                                            step={0.1}
                                            size="1"
                                        />
                                    </Box>
                                </Flex>
                            </Flex>
                        </Card>
                    )
                })}

                {selectedLoras.length === 0 && (
                    <Text size="2" color="gray">
                        LoRAが選択されていません
                    </Text>
                )}
            </Flex>
        </Box>
    )
}
