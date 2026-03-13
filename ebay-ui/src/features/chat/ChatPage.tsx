import { useEffect, useRef } from "react"
import { Box, Paper } from "@mui/material"

import { useChatSession } from "../../hooks/useChatSession.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx"
import ChatLayout from "./ChatLayout.tsx"
import ChatInput from "./ChatInput.tsx"
import MessageBubble from "./MessageBubble.tsx"
import SearchBlockView from "./SearchBlockView.tsx"
import WelcomePanel from "./WelcomePanel.tsx"

export default function ChatPage() {
  const {
    chat,
    steps,
    running,
    loadingQuery,
    hasSearches,
    handleSend,
    resetChat,
    finalPayload,
    plannedTasks
  } = useChatSession()

  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end"
    })
  }, [chat, steps, running, loadingQuery])

  const showWelcome = !hasSearches && chat.length <= 1

  useEffect(() => {
    const handleSendChat = (e: CustomEvent<string>) => {
      handleSend(e.detail)
    }
    window.addEventListener("send-chat", handleSendChat as EventListener)
    return () => window.removeEventListener("send-chat", handleSendChat as EventListener)
  }, [handleSend])

  return (
    <ChatLayout
      onNewChat={resetChat}
      composer={<ChatInput onSend={handleSend} disabled={running} />}
    >
      <Box
        sx={{
          width: "100%",
          maxWidth: 880,
          mx: "auto",
          px: { xs: 2, md: 3 },
          pt: { xs: 3, md: 4 },
          pb: 2
        }}
      >
        {showWelcome ? <WelcomePanel /> : null}

        {chat.map((entry, index) =>
          entry.type === "message" ? (
            <Box key={`msg-${index}`} mb={2.5}>
              <MessageBubble role={entry.msg.role}>
                {entry.msg.content}
              </MessageBubble>
            </Box>
          ) : (
            <Box key={`search-${index}`}>
              <SearchBlockView search={entry.search} />
            </Box>
          )
        )}

        {running ? (
          <Box mt={1.5} mb={3}>
            <MessageBubble role="assistant">
              <Paper
                elevation={0}
                sx={{
                  p: 2.25,
                  borderRadius: 4,
                  border: "1px solid #e5e7eb",
                  bgcolor: "#ffffff"
                }}
              >
                <AIThinkingPipeline
                  agentTrace={steps}
                  loading
                  query={loadingQuery ?? undefined}
                  plannedTasks={plannedTasks}
                />
              </Paper>
            </MessageBubble>
          </Box>
        ) : null}

        {!running &&
          finalPayload &&
          !finalPayload.results?.length &&
          !finalPayload.sellerSummary?.seller_name &&
          !finalPayload.analysis &&
          !chat.some((entry) => entry.type === "search") ? (
          <Box mt={1.5} mb={3}>
            <MessageBubble role="assistant">
              {finalPayload.finalAnswer || "Ho completato l’analisi della richiesta."}
            </MessageBubble>
          </Box>
        ) : null}

        <div ref={bottomRef} />
      </Box>
    </ChatLayout>
  )
}