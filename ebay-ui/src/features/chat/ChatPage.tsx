import { useEffect, useRef } from "react"
import { Box, Paper } from "@mui/material"

import { useChatSession } from "../../hooks/useChatSession.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx"
import ChatLayout from "./ChatLayout.tsx"
import ChatInput from "./ChatInput.tsx"
import MessageBubble from "./MessageBubble.tsx"
import SearchBlockView from "./SearchBlockView.tsx"
import WelcomePanel from "./WelcomePanel.tsx"

import ErrorBoundary from "./ErrorBoundary.tsx"

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
  }, [chat, steps, running, loadingQuery, finalPayload?.finalAnswer])

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
      <ErrorBoundary>
        <Box
          sx={{
            width: "100%",
            maxWidth: 1200,
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

          {/* STREAMING RESPONSE OR THINKING PIPELINE */}
          {running && (
            <Box mt={1.5} mb={3}>
              {/* Show thinking pipeline ONLY if we don't have enough answer chunks to cover the UI */}
              {(!finalPayload?.finalAnswer || finalPayload.finalAnswer.length < 50) && (
                <MessageBubble role="assistant" isTyping={true}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2.25,
                      borderRadius: 4,
                      border: "1px solid #f0f0f0",
                      bgcolor: "#f8fafc"
                    }}
                  >
                    <AIThinkingPipeline
                      agentTrace={steps}
                      loading={!finalPayload?.finalAnswer}
                      query={loadingQuery ?? undefined}
                      plannedTasks={plannedTasks}
                    />
                  </Paper>
                </MessageBubble>
              )}

              {/* Show streaming answer if it exists */}
              {finalPayload?.finalAnswer ? (
                <MessageBubble role="assistant">
                  {finalPayload.finalAnswer}
                </MessageBubble>
              ) : null}
            </Box>
          )}

          <div ref={bottomRef} />
        </Box>
      </ErrorBoundary>
    </ChatLayout>
  )
}