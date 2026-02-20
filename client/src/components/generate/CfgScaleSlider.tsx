/**
 * CFG Scale slider component
 */

import { Flex, Slider, Text } from "@radix-ui/themes"

import { FieldResetButton } from "@/components/generate/FieldResetButton"
import { useServerInfo } from "@/hooks/useServerInfo"
import { useGenerateStore } from "@/stores/generateStore"

export const CfgScaleSlider = () => {
    const cfgScale = useGenerateStore((state) => state.form.cfgScale) ?? 7.0
    const setCfgScale = useGenerateStore((state) => state.setCfgScale)
    const { data: serverInfo } = useServerInfo()

    const cfgScaleSchema = serverInfo?.parameter_schema?.properties?.cfg_scale
    const defaultValue =
        (cfgScaleSchema?.default as number | undefined) ??
        serverInfo?.parameter_schema?.defaults?.cfg_scale ??
        serverInfo?.parameter_schema?.fixed?.cfg_scale ??
        7.0
    const min = (cfgScaleSchema?.minimum as number | undefined) ?? 0
    const max = (cfgScaleSchema?.maximum as number | undefined) ?? 15
    const step = (cfgScaleSchema?.step as number | undefined) ?? 0.5

    return (
        <Flex direction="column" justify="between" style={{ height: 60 }}>
            <Flex justify="between" align="center">
                <Flex align="center" gap="2">
                    <Text as="label" size="2" weight="medium">
                        CFG Scale
                    </Text>
                    <FieldResetButton fieldName="CFG Scale" onReset={() => setCfgScale(defaultValue)} />
                </Flex>
                <Text size="1" color="gray">{cfgScale.toFixed(1)}</Text>
            </Flex>
            <Flex align="center" style={{ height: 32 }}>
                <Slider
                    value={[cfgScale]}
                    onValueChange={([value]) => setCfgScale(value)}
                    min={min}
                    max={max}
                    step={step}
                    size="1"
                    style={{ flex: 1 }}
                />
            </Flex>
        </Flex>
    )
}
