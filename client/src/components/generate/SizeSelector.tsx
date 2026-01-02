/**
 * SizeSelector - Width and height selection with presets
 */

import { Box, Flex, Select, Slider, Text } from "@radix-ui/themes"

import { useGenerateStore } from "@/stores/generateStore"

const SIZE_PRESETS = [
    { label: "1:1 (1024×1024)", width: 1024, height: 1024 },
    { label: "4:3 (1152×896)", width: 1152, height: 896 },
    { label: "3:4 (896×1152)", width: 896, height: 1152 },
    { label: "16:9 (1344×768)", width: 1344, height: 768 },
    { label: "9:16 (768×1344)", width: 768, height: 1344 },
    { label: "3:2 (1216×832)", width: 1216, height: 832 },
    { label: "2:3 (832×1216)", width: 832, height: 1216 },
] as const

export const SizeSelector = () => {
    const width = useGenerateStore((state) => state.form.width)
    const height = useGenerateStore((state) => state.form.height)
    const setSize = useGenerateStore((state) => state.setSize)
    const setWidth = useGenerateStore((state) => state.setWidth)
    const setHeight = useGenerateStore((state) => state.setHeight)

    const currentPreset = SIZE_PRESETS.find(
        (p) => p.width === width && p.height === height
    )

    const handlePresetChange = (value: string) => {
        if (value === "custom") return
        const preset = SIZE_PRESETS.find((p) => `${p.width}x${p.height}` === value)
        if (preset) {
            setSize(preset.width, preset.height)
        }
    }

    return (
        <Flex direction="column" gap="2">
            <Text as="label" size="2" weight="medium">
                サイズ
            </Text>

            <Select.Root
                value={currentPreset ? `${currentPreset.width}x${currentPreset.height}` : "custom"}
                onValueChange={handlePresetChange}
            >
                <Select.Trigger style={{ width: "100%" }} />
                <Select.Content>
                    {SIZE_PRESETS.map((preset) => (
                        <Select.Item
                            key={`${preset.width}x${preset.height}`}
                            value={`${preset.width}x${preset.height}`}
                        >
                            {preset.label}
                        </Select.Item>
                    ))}
                    {!currentPreset && <Select.Item value="custom">カスタム ({width}×{height})</Select.Item>}
                </Select.Content>
            </Select.Root>

            <Box>
                <Flex justify="between" mb="1">
                    <Text size="1" color="gray">幅: {width}</Text>
                </Flex>
                <Slider
                    value={[width]}
                    onValueChange={([v]) => setWidth(v)}
                    min={256}
                    max={2048}
                    step={64}
                />
            </Box>

            <Box>
                <Flex justify="between" mb="1">
                    <Text size="1" color="gray">高さ: {height}</Text>
                </Flex>
                <Slider
                    value={[height]}
                    onValueChange={([v]) => setHeight(v)}
                    min={256}
                    max={2048}
                    step={64}
                />
            </Box>
        </Flex>
    )
}
