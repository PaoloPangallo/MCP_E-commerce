import { useEffect, useRef } from "react"
import { Box } from "@mui/material"

import { useChatSession } from "../../hooks/useChatSession.ts"
import { ThinkingPill } from "../agent/components/ThinkingPill.tsx"
import ChatLayout from "./ChatLayout.tsx"
import ChatInput from "./ChatInput.tsx"
import MessageBubble from "./MessageBubble.tsx"
import SearchBlockView from "./SearchBlockView.tsx"
import WelcomePanel from "./WelcomePanel.tsx"
import ErrorBoundary from "./ErrorBoundary.tsx"
import VisionAnalysisCard from "./VisionAnalysisCard.tsx"
import { useScrollContainer } from "./ScrollContainerContext.ts"
import type { AgentStep, PlannedTask } from "../agent/types"


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
  const scrollContainerRef = useScrollContainer()
  const isAtBottomRef = useRef(true)
  const isAutoScrollingRef = useRef(false)

  // Aggiorna se siamo vicini al fondo durante lo scroll dell'utente
  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) return

    const handleScroll = () => {
      // Ignora l'evento di scroll se l'abbiamo scatenato noi programmaticamente
      if (isAutoScrollingRef.current) return
      
      const threshold = 150 // pixel di tolleranza
      const isAtBottom =
        container.scrollHeight - container.scrollTop <= container.clientHeight + threshold

      isAtBottomRef.current = isAtBottom
    }

    container.addEventListener("scroll", handleScroll, { passive: true })
    return () => container.removeEventListener("scroll", handleScroll)
  }, [])

  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) return

    // Se l'utente ha scrollato in su, non forziamo l'aggancio in basso per non disturbarlo
    if (running && !isAtBottomRef.current) return

    isAutoScrollingRef.current = true
    
    // Lo scroll "smooth" in questa fase causa sfasamenti visivi (jittering) e trigger involontari
    // degli eventi di scroll che rovinano la logica sticky. Usare "auto" per pinning perfetto stile ChatGPT.
    container.scrollTo({ top: container.scrollHeight, behavior: "auto" })
    
    requestAnimationFrame(() => {
      setTimeout(() => {
        isAutoScrollingRef.current = false
      }, 50)
    })
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
            maxWidth: 1000,
            mx: "auto",
            px: { xs: 2, md: 0 },
            pt: { xs: 3, md: 4 },
            pb: 4,
            minHeight: showWelcome ? '70vh' : 'auto',
            display: showWelcome ? 'flex' : 'block',
            flexDirection: showWelcome ? 'column' : 'initial',
            justifyContent: showWelcome ? 'center' : 'initial'
          }}
        >
          {showWelcome && <WelcomePanel />}

          {chat.map((entry, index) =>
            entry.type === "message" ? (
              <MessageBubble 
                key={`msg-${index}`} 
                role={entry.msg.role}
                image={entry.msg.image}
              >
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
              {finalPayload?.visionAnalysis && (
                <VisionAnalysisCard
                  description={finalPayload.visionAnalysis.description}
                  tags={finalPayload.visionAnalysis.tags}
                />
              )}
              <ThinkingPill
                steps={typedSteps}
                loading={!finalPayload?.finalAnswer}
                query={loadingQuery ?? undefined}
                plannedTasks={typedPlannedTasks}
              />

              {/* BLOCCO RISULTATI LIVE: Mostriamo i dati tecnici (deals, search, etc.) 
                  appena arrivano, anche se l'agente sta ancora scrivendo la risposta. */}
              {finalPayload && (
                <Box sx={{ mt: 2 }}>
                  <SearchBlockView 
                    search={{
                      query: loadingQuery || "",
                      results: finalPayload.results || [],
                      analysis: finalPayload.analysis,
                      metrics: finalPayload.metrics,
                      rag_context: finalPayload.ragContext,
                      agent_trace: typedSteps,
                      seller_summary: finalPayload.sellerSummary,
                      comparison: finalPayload.comparison,
                      item_details: finalPayload.itemDetails,
                      shipping_costs: finalPayload.shippingCosts,
                      metadata: finalPayload.metadata,
                      deals: finalPayload.deals,
                      market_trends: finalPayload.marketTrends,
                      vision_analysis: finalPayload.visionAnalysis,
                      errors: finalPayload.errors,
                      mode: "hybrid" // Default per il live view
                    }} 
                  />
                </Box>
              )}

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