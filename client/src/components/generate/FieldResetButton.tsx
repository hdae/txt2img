/**
 * FieldResetButton - Reusable reset button with confirmation dialog for individual fields
 */

import { AlertDialog, Button, Flex, IconButton } from "@radix-ui/themes"
import { RotateCcw } from "lucide-react"

interface FieldResetButtonProps {
    fieldName: string
    onReset: () => void
}

export const FieldResetButton = ({ fieldName, onReset }: FieldResetButtonProps) => {
    return (
        <AlertDialog.Root>
            <AlertDialog.Trigger>
                <IconButton variant="ghost" color="red" size="1">
                    <RotateCcw size={14} />
                </IconButton>
            </AlertDialog.Trigger>
            <AlertDialog.Content maxWidth="400px">
                <AlertDialog.Title>{fieldName}をリセット</AlertDialog.Title>
                <AlertDialog.Description size="2">
                    {fieldName}をデフォルト値に戻します。よろしいですか？
                </AlertDialog.Description>
                <Flex gap="3" mt="4" justify="end">
                    <AlertDialog.Cancel>
                        <Button variant="soft" color="gray">
                            キャンセル
                        </Button>
                    </AlertDialog.Cancel>
                    <AlertDialog.Action>
                        <Button variant="solid" color="red" onClick={onReset}>
                            リセット
                        </Button>
                    </AlertDialog.Action>
                </Flex>
            </AlertDialog.Content>
        </AlertDialog.Root>
    )
}
