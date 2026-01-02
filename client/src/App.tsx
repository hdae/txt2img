/**
 * Main App component with Radix UI Themes and TanStack Query setup
 */

import "@radix-ui/themes/styles.css"

import { Box, Flex, ScrollArea, Tabs, Theme } from "@radix-ui/themes"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Toaster } from "react-hot-toast"

import { GalleryTab } from "@/components/gallery/GalleryTab"
import { GenerateTab } from "@/components/generate/GenerateTab"
import { ServerInfoTab } from "@/components/server/ServerInfoTab"
import { GalleryProvider } from "@/contexts/GalleryContext"
import { GenerateProvider } from "@/contexts/GenerateContext"

const queryClient = new QueryClient()

export const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <GenerateProvider>
        <GalleryProvider>
          <Theme appearance="dark" accentColor="violet" grayColor="slate" radius="medium">
            <Flex direction="column" style={{ height: "100vh" }}>
              <Tabs.Root defaultValue="generate" style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
                <Tabs.List size="2">
                  <Tabs.Trigger value="generate">生成</Tabs.Trigger>
                  <Tabs.Trigger value="gallery">ギャラリー</Tabs.Trigger>
                  <Tabs.Trigger value="server">サーバー情報</Tabs.Trigger>
                </Tabs.List>

                <Box style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
                  <ScrollArea style={{ height: "100%" }}>
                    <Box p="4">
                      <Tabs.Content value="generate">
                        <GenerateTab />
                      </Tabs.Content>

                      <Tabs.Content value="gallery">
                        <GalleryTab />
                      </Tabs.Content>

                      <Tabs.Content value="server">
                        <ServerInfoTab />
                      </Tabs.Content>
                    </Box>
                  </ScrollArea>
                </Box>
              </Tabs.Root>
            </Flex>
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
        </GalleryProvider>
      </GenerateProvider>
    </QueryClientProvider>
  )
}
