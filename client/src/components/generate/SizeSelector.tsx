/**
 * SizeSelector - Aspect ratio presets with calculated dimensions
 */

import { Flex, Select, Text } from "@radix-ui/themes";

import { useGenerateStore } from "@/stores/generateStore";

const ASPECT_PRESETS = [
    { label: "1:1 (正方形)", ratio: [1, 1] },
    { label: "4:3 (標準)", ratio: [4, 3] },
    { label: "3:4 (縦・標準)", ratio: [3, 4] },
    { label: "16:9 (ワイド)", ratio: [16, 9] },
    { label: "9:16 (縦・ワイド)", ratio: [9, 16] },
    { label: "3:2 (写真)", ratio: [3, 2] },
    { label: "2:3 (縦・写真)", ratio: [2, 3] },
    { label: "21:9 (シネマ)", ratio: [21, 9] },
] as const

/**
 * Calculate dimensions from base size and aspect ratio
 */
function calculateDimensions(
    baseSize: number,
    ratioW: number,
    ratioH: number
): { width: number; height: number } {
    const stride = 64
    const targetPixels = baseSize * baseSize
    const targetRatio = ratioW / ratioH

    const idealWidth = Math.sqrt(targetPixels * targetRatio)
    const idealHeight = Math.sqrt(targetPixels / targetRatio)

    const width = Math.round(idealWidth / stride) * stride
    const height = Math.round(idealHeight / stride) * stride

    return { width, height }
}

interface SizeSelectorProps {
    trainingResolution: number
}

export const SizeSelector = ({ trainingResolution }: SizeSelectorProps) => {
    const width = useGenerateStore((state) => state.form.width)
    const height = useGenerateStore((state) => state.form.height)
    const setSize = useGenerateStore((state) => state.setSize)

    // Find current preset by comparing dimensions
    const currentPreset = ASPECT_PRESETS.find((preset) => {
        const dims = calculateDimensions(trainingResolution, preset.ratio[0], preset.ratio[1])
        return dims.width === width && dims.height === height
    })

    const handlePresetChange = (value: string) => {
        const preset = ASPECT_PRESETS.find((p) => p.label === value)
        if (preset) {
            const { width: w, height: h } = calculateDimensions(
                trainingResolution,
                preset.ratio[0],
                preset.ratio[1]
            )
            setSize(w, h)
        }
    }

    return (
        <Flex direction="column" gap="2">
            <Text as="label" size="2" weight="medium">
                サイズ
            </Text>
            <Select.Root
                value={currentPreset?.label || ""}
                onValueChange={handlePresetChange}
                size="2"
            >
                <Select.Trigger placeholder="アスペクト比を選択..." style={{ width: "100%" }} />
                <Select.Content>
                    {ASPECT_PRESETS.map((preset) => {
                        const dims = calculateDimensions(trainingResolution, preset.ratio[0], preset.ratio[1])
                        return (
                            <Select.Item key={preset.label} value={preset.label}>
                                {preset.label} ({dims.width}×{dims.height})
                            </Select.Item>
                        )
                    })}
                </Select.Content>
            </Select.Root>
        </Flex>
    )
}
