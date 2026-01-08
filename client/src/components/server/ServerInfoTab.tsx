/**
 * ServerInfoTab - Display server information
 */

import { Badge, Box, Card, Code, Flex, Separator, Text } from "@radix-ui/themes"

import { useServerInfo } from "@/hooks/useServerInfo"

export const ServerInfoTab = () => {
    const { data: serverInfo, isLoading, error } = useServerInfo()

    if (isLoading) {
        return <Text color="gray">読み込み中...</Text>
    }

    if (error) {
        return <Text color="red">エラー: {error.message}</Text>
    }

    if (!serverInfo) {
        return <Text color="gray">サーバー情報がありません</Text>
    }

    const schema = serverInfo.parameter_schema

    return (
        <Flex direction="column" gap="4" style={{ maxWidth: 600 }}>
            <Card>
                <Flex direction="column" gap="3">
                    <Text size="3" weight="bold">モデル情報</Text>
                    <Separator size="4" />

                    <Flex justify="between">
                        <Text color="gray">モデル名</Text>
                        <Text>{serverInfo.model_name}</Text>
                    </Flex>

                    <Flex justify="between">
                        <Text color="gray">モデルタイプ</Text>
                        <Text>{schema.model_type}</Text>
                    </Flex>

                    <Flex justify="between">
                        <Text color="gray">プロンプトスタイル</Text>
                        <Text>{schema.prompt_style === "tags" ? "タグ形式" : "自然言語"}</Text>
                    </Flex>

                    <Flex justify="between">
                        <Text color="gray">学習解像度</Text>
                        <Text>{serverInfo.training_resolution}px</Text>
                    </Flex>

                    <Flex justify="between">
                        <Text color="gray">出力形式</Text>
                        <Text>{serverInfo.output_format.toUpperCase()}</Text>
                    </Flex>
                </Flex>
            </Card>

            <Card>
                <Flex direction="column" gap="3">
                    <Text size="3" weight="bold">固定パラメータ</Text>
                    <Separator size="4" />

                    <Flex justify="between">
                        <Text color="gray">ステップ数</Text>
                        <Code>{schema.fixed.steps}</Code>
                    </Flex>
                </Flex>
            </Card>

            {serverInfo.available_loras.length > 0 && (
                <Card>
                    <Flex direction="column" gap="3">
                        <Text size="3" weight="bold">利用可能なLoRA ({serverInfo.available_loras.length})</Text>
                        <Separator size="4" />

                        {serverInfo.available_loras.map((lora) => (
                            <Box key={lora.id}>
                                <Flex justify="between" align="center">
                                    <Text weight="medium">{lora.name}</Text>
                                    <Code size="1">{lora.id}</Code>
                                </Flex>
                                {lora.trigger_words.length > 0 && (
                                    <Flex gap="1" mt="1" wrap="wrap">
                                        {lora.trigger_words.map((word) => (
                                            <Badge key={word} size="1" variant="soft">
                                                {word}
                                            </Badge>
                                        ))}
                                    </Flex>
                                )}
                            </Box>
                        ))}
                    </Flex>
                </Card>
            )}
        </Flex>
    )
}
