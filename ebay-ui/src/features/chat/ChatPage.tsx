import { useEffect, useRef, useState } from "react"
import { Box, Collapse, Paper } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"

import { useChatSession } from "../../hooks/useChatSession.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx"
import ChatLayout from "./ChatLayout.tsx"
import ChatInput from "./ChatInput.tsx"
import MessageBubble from "./MessageBubble.tsx"
import SearchBlockView from "./SearchBlockView.tsx"
import WelcomePanel from "./WelcomePanel.tsx"
import ErrorBoundary from "./ErrorBoundary.tsx"
import type { AgentStep, PlannedTask } from "../agent/types"

interface ThinkingPillProps {
  steps: AgentStep[]
  loading: boolean
  query?: string
  plannedTasks?: PlannedTask[]
}

function ThinkingPill({ steps, loading, query, plannedTasks }: ThinkingPillProps) {
  const [open, setOpen] = useState(false)

  const label = loading
    ? "L'agente sta ragionando…"
    : `Ragionamento completato${steps.length > 0 ? ` · ${steps.length} passi` : ""}`

  return (
    <Box sx={{ mb: 1.5 }}>
      <Box
        onClick={() => setOpen((v) => !v)}
        sx={{
          display: "inline-flex",
          alignItems: "center",
          gap: 0.75,
          px: 1.5,
          py: 0.6,
          mb: open ? 1 : 0,
          border: "1px solid #e5e7eb",
          borderRadius: "20px",
          cursor: "pointer",
          bgcolor: "#fafafa",
          transition: "background 0.15s",
          "&:hover": { bgcolor: "#f3f4f6" },
          userSelect: "none"
        }}
      >
        <Box
          sx={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            bgcolor: loading ? "#f59e0b" : "#10a37f",
            animation: loading ? "dotPulse 1.2s infinite ease-in-out" : "none",
            "@keyframes dotPulse": {
              "0%, 100%": { opacity: 1 },
              "50%": { opacity: 0.35 }
            }
          }}
        />
        <Box component="span" sx={{ fontSize: 12, color: "#6b7280", lineHeight: 1 }}>
          {label}
        </Box>
        <KeyboardArrowDownIcon
          sx={{
            fontSize: 14,
            color: "#9ca3af",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.2s"
          }}
        />
      </Box>

      <Collapse in={open} timeout={200}>
        <Paper
          elevation={0}
          sx={{ p: 2, borderRadius: 3, border: "1px solid #e5e7eb", bgcolor: "#f8fafc" }}
        >
          <AIThinkingPipeline
            agentTrace={steps}
            loading={loading}
            query={query}
            plannedTasks={plannedTasks}
          />
        </Paper>
      </Collapse>
    </Box>
  )
}

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
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [chat, steps, running, loadingQuery, finalPayload?.finalAnswer])

  useEffect(() => {
    const handleSendChat = (e: CustomEvent<string>) => {
      handleSend(e.detail)
    }
    window.addEventListener("send-chat", handleSendChat as EventListener)
    return () => window.removeEventListener("send-chat", handleSendChat as EventListener)
  }, [handleSend])

  const showWelcome = !hasSearches && chat.length <= 1

  // Cast to the correct types — useChatSession may return unknown[] depending on its definition
  const typedSteps = (steps ?? []) as AgentStep[]
  const typedPlannedTasks = (plannedTasks ?? []) as PlannedTask[]

  return (
    <ChatLayout
      onNewChat={resetChat}
      composer={<ChatInput onSend={handleSend} disabled={running} />}
    >
      <ErrorBoundary>
        <Box
          sx={{
            width: "100%",
            maxWidth: 720,
            mx: "auto",
            px: { xs: 2, md: 0 },
            pt: { xs: 3, md: 4 },
            pb: 4
          }}
        >
          {showWelcome && <WelcomePanel />}

          {chat.map((entry, index) =>
            entry.type === "message" ? (
              <MessageBubble key={`msg-${index}`} role={entry.msg.role}>
                {entry.msg.content}
              </MessageBubble>
            ) : (
              <Box key={`search-${index}`} sx={{ ml: { md: "44px" } }}>
                <SearchBlockView search={entry.search} />
              </Box>
            )
          )}

          {running && (
            <Box sx={{ ml: { md: "44px" } }}>
              <ThinkingPill
                steps={typedSteps}
                loading={!finalPayload?.finalAnswer}
                query={loadingQuery ?? undefined}
                plannedTasks={typedPlannedTasks}
              />
              {finalPayload?.finalAnswer && (
                <MessageBubble role="assistant">
                  {finalPayload.finalAnswer}
                </MessageBubble>
              )}
            </Box>
          )}

          <div ref={bottomRef} />
        </Box>
      </ErrorBoundary>
    </ChatLayout>
  )
}