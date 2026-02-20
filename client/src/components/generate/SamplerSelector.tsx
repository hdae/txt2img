/**
 * Sampler selector component
 */

import { Flex, Select, Text } from "@radix-ui/themes"

import { FieldResetButton } from "@/components/generate/FieldResetButton"
import { useServerInfo } from "@/hooks/useServerInfo"
import { useGenerateStore } from "@/stores/generateStore"

export const SamplerSelector = () => {
    const sampler = useGenerateStore((state) => state.form.sampler) ?? "euler_a"
    const setSampler = useGenerateStore((state) => state.setSampler)
    const { data: serverInfo } = useServerInfo()

    // Get available samplers from schema
    const availableSamplers = serverInfo?.parameter_schema?.properties?.sampler?.enum ?? [
        "euler",
        "euler_a",
        "dpm++_2m",
    ]
    const defaultSampler =
        (serverInfo?.parameter_schema?.properties?.sampler?.default as string | undefined) ??
        serverInfo?.parameter_schema?.defaults?.sampler ??
        "euler_a"

    // Display names for samplers
    const samplerNames: Record<string, string> = {
        euler: "Euler",
        euler_a: "Euler a",
        "dpm++_2m": "DPM++ 2M",
    }

    return (
        <Flex direction="column" justify="between" style={{ height: 60 }}>
            <Flex align="center" gap="2">
                <Text as="label" size="2" weight="medium">
                    サンプラー
                </Text>
                <FieldResetButton fieldName="サンプラー" onReset={() => setSampler(defaultSampler)} />
            </Flex>
            <Select.Root
                value={sampler}
                onValueChange={setSampler}
            >
                <Select.Trigger style={{ width: "100%" }} />
                <Select.Content>
                    {(availableSamplers as string[]).map((s) => (
                        <Select.Item key={s} value={s}>
                            {samplerNames[s] ?? s}
                        </Select.Item>
                    ))}
                </Select.Content>
            </Select.Root>
        </Flex>
    )
}
