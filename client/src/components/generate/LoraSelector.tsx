/**
 * LoraSelector - LoRA selection with dialog and RadioCards
 */

import { useState } from "react"

import {
    Badge,
    Box,
    Button,
    Card,
    Dialog,
    Flex,
    Grid,
    IconButton,
    RadioCards,
    Slider,
    Text,
} from "@radix-ui/themes"
import { Plus, X } from "lucide-react"

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

    const [dialogOpen, setDialogOpen] = useState(false)
    const [selectedLoraId, setSelectedLoraId] = useState<string>("")

    const availableLoras = loras.filter(
        (lora) => !selectedLoras.find((selected) => selected.id === lora.id)
    )

    const handleAddLora = () => {
        if (!selectedLoraId) return
        const lora = loras.find((l) => l.id === selectedLoraId)
        if (lora) {
            addLora({
                id: lora.id,
                weight: lora.weight,
                trigger_weight: lora.trigger_weight,
            })
            setSelectedLoraId("")
            setDialogOpen(false)
        }
    }

    return (
        <Box>
            <Flex justify="between" align="center" mb="2">
                <Text as="label" size="2" weight="medium">
                    LoRA
                </Text>
                {availableLoras.length > 0 && (
                    <Dialog.Root open={dialogOpen} onOpenChange={setDialogOpen}>
                        <Dialog.Trigger>
                            <Button variant="soft" size="1">
                                <Plus size={14} />
                                追加
                            </Button>
                        </Dialog.Trigger>
                        <Dialog.Content maxWidth="500px">
                            <Dialog.Title>LoRAを追加</Dialog.Title>
                            <Dialog.Description size="2" mb="4">
                                適用するLoRAを選択してください
                            </Dialog.Description>

                            <RadioCards.Root
                                value={selectedLoraId}
                                onValueChange={setSelectedLoraId}
                                columns="1"
                            >
                                {availableLoras.map((lora) => (
                                    <RadioCards.Item key={lora.id} value={lora.id}>
                                        <Flex direction="column" width="100%" gap="1">
                                            <Flex justify="between" align="center">
                                                <Text weight="bold">{lora.name}</Text>
                                                <Text size="1" color="gray">{lora.id}</Text>
                                            </Flex>
                                            {lora.trigger_words.length > 0 && (
                                                <Flex gap="1" wrap="wrap">
                                                    {lora.trigger_words.slice(0, 3).map((word) => (
                                                        <Badge key={word} size="1" variant="soft">
                                                            {word}
                                                        </Badge>
                                                    ))}
                                                    {lora.trigger_words.length > 3 && (
                                                        <Badge size="1" variant="soft" color="gray">
                                                            +{lora.trigger_words.length - 3}
                                                        </Badge>
                                                    )}
                                                </Flex>
                                            )}
                                        </Flex>
                                    </RadioCards.Item>
                                ))}
                            </RadioCards.Root>

                            <Flex gap="3" mt="4" justify="end">
                                <Dialog.Close>
                                    <Button variant="soft" color="gray">
                                        キャンセル
                                    </Button>
                                </Dialog.Close>
                                <Button onClick={handleAddLora} disabled={!selectedLoraId}>
                                    追加
                                </Button>
                            </Flex>
                        </Dialog.Content>
                    </Dialog.Root>
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

                                <Grid columns={{ initial: "1", sm: "2" }} gap="3">
                                    <Box>
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
                                    <Box>
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
                                </Grid>
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
