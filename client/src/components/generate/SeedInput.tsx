/**
 * SeedInput - Random seed input
 */

import { Box, Flex, IconButton, Text, TextField } from "@radix-ui/themes"
import { Shuffle } from "lucide-react"

import { useGenerateStore } from "@/stores/generateStore"

export const SeedInput = () => {
    const seed = useGenerateStore((state) => state.form.seed)
    const setSeed = useGenerateStore((state) => state.setSeed)

    const handleChange = (value: string) => {
        if (value === "") {
            setSeed(null)
        } else {
            const num = parseInt(value, 10)
            if (!isNaN(num) && num >= 0) {
                setSeed(num)
            }
        }
    }

    const handleRandomize = () => {
        setSeed(Math.floor(Math.random() * 2147483647))
    }

    return (
        <Box>
            <Text as="label" size="2" weight="medium" mb="2" style={{ display: "block" }}>
                シード
            </Text>
            <Flex gap="2" align="center">
                <TextField.Root
                    value={seed === null ? "" : seed.toString()}
                    onChange={(e) => handleChange(e.target.value)}
                    placeholder="ランダム"
                    type="number"
                    style={{ flex: 1 }}
                />
                <IconButton
                    variant="soft"
                    onClick={handleRandomize}
                    title="ランダムシードを生成"
                >
                    <Shuffle size={16} />
                </IconButton>
            </Flex>
            <Text size="1" color="gray" mt="1">
                空欄でランダム
            </Text>
        </Box>
    )
}
