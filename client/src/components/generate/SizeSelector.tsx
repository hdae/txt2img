/**
 * SizeSelector - Aspect ratio presets with calculated dimensions
 */

import { Flex, Select, Text } from "@radix-ui/themes";

import { useGenerateStore } from "@/stores/generateStore";

const ASPECT_PRESETS = [
    { label: "1:1 (正方形)", ratio: [1, 1], key: "1:1" },
    { label: "4:3 (標準)", ratio: [4, 3], key: "4:3" },
    { label: "3:4 (縦・標準)", ratio: [3, 4], key: "3:4" },
    { label: "16:9 (ワイド)", ratio: [16, 9], key: "16:9" },
    { label: "9:16 (縦・ワイド)", ratio: [9, 16], key: "9:16" },
    { label: "3:2 (写真)", ratio: [3, 2], key: "3:2" },
    { label: "2:3 (縦・写真)", ratio: [2, 3], key: "2:3" },
    { label: "21:9 (シネマ)", ratio: [21, 9], key: "21:9" },
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
    const aspectRatio = useGenerateStore((state) => state.form.aspectRatio) ?? "1:1"
    const setAspectRatio = useGenerateStore((state) => state.setAspectRatio)
    const setSize = useGenerateStore((state) => state.setSize)

    // Find current preset by aspectRatio key
    const currentPreset = ASPECT_PRESETS.find((p) => p.key === aspectRatio) ?? ASPECT_PRESETS[0]
    const dims = calculateDimensions(trainingResolution, currentPreset.ratio[0], currentPreset.ratio[1])

    const handlePresetChange = (value: string) => {
        const preset = ASPECT_PRESETS.find((p) => p.key === value)
        if (preset) {
            setAspectRatio(preset.key)
            const { width: w, height: h } = calculateDimensions(
                trainingResolution,
                preset.ratio[0],
                preset.ratio[1]
            )
            setSize(w, h)
        }
    }

    return (
        <Flex direction="column" justify="between" style={{ height: 60 }}>
            <Flex justify="between" align="center">
                <Text as="label" size="2" weight="medium">
                    サイズ
                </Text>
                <Text size="1" color="gray">{dims.width}×{dims.height}</Text>
            </Flex>
            <Select.Root
                value={aspectRatio}
                onValueChange={handlePresetChange}
                size="2"
            >
                <Select.Trigger style={{ width: "100%" }} />
                <Select.Content>
                    {ASPECT_PRESETS.map((preset) => {
                        const d = calculateDimensions(trainingResolution, preset.ratio[0], preset.ratio[1])
                        return (
                            <Select.Item key={preset.key} value={preset.key}>
                                {preset.label} ({d.width}×{d.height})
                            </Select.Item>
                        )
                    })}
                </Select.Content>
            </Select.Root>
        </Flex>
    )
}
