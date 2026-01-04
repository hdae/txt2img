/**
 * SeedInput - Seed with lock toggle
 */

import { Flex, Switch, Text, TextField } from "@radix-ui/themes"

import { useGenerateStore } from "@/stores/generateStore"

export const SeedInput = () => {
    const seed = useGenerateStore((state) => state.form.seed)
    const seedLocked = useGenerateStore((state) => state.form.seedLocked)
    const setSeed = useGenerateStore((state) => state.setSeed)
    const toggleSeedLocked = useGenerateStore((state) => state.toggleSeedLocked)

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

    return (
        <Flex direction="column" justify="between" style={{ height: 60 }}>
            <Flex justify="between" align="center">
                <Text as="label" size="2" weight="medium">
                    シード
                </Text>
                <Flex align="center" gap="2">
                    <Text size="1" color="gray">
                        固定
                    </Text>
                    <Switch
                        size="1"
                        checked={seedLocked}
                        onCheckedChange={toggleSeedLocked}
                    />
                </Flex>
            </Flex>
            <TextField.Root
                value={seed === null ? "" : seed.toString()}
                onChange={(e) => handleChange(e.target.value)}
                placeholder={seedLocked ? "シードを入力..." : "ランダム"}
                type="number"
                disabled={!seedLocked}
            />
        </Flex>
    )
}
