/**
 * Main App component with Radix UI Themes and TanStack Query setup
 */

import "@radix-ui/themes/styles.css"

import { Box, Container, Tabs, Theme } from "@radix-ui/themes"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Toaster } from "react-hot-toast"

import { GenerateTab } from "@/components/generate/GenerateTab"

const queryClient = new QueryClient()

export const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <Theme appearance="dark" accentColor="violet" grayColor="slate" radius="medium">
        <Box
          style={{
            minHeight: "100vh",
            backgroundColor: "var(--color-background)",
          }}
        >
          <Container size="3" py="4">
            <Tabs.Root defaultValue="generate">
              <Tabs.List size="2">
                <Tabs.Trigger value="generate">生成</Tabs.Trigger>
                <Tabs.Trigger value="gallery">ギャラリー</Tabs.Trigger>
              </Tabs.List>

              <Box pt="4">
                <Tabs.Content value="generate">
                  <GenerateTab />
                </Tabs.Content>

                <Tabs.Content value="gallery">
                  <Box py="4">ギャラリー（実装予定）</Box>
                </Tabs.Content>
              </Box>
            </Tabs.Root>
          </Container>
        </Box>
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: "var(--gray-2)",
              color: "var(--gray-12)",
              border: "1px solid var(--gray-6)",
            },
          }}
        />
      </Theme>
    </QueryClientProvider>
  )
}
