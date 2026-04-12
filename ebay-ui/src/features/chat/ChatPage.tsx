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
  const openedLinksRef = useRef<Set<string>>(new Set())

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
    
    const timeoutId = setTimeout(() => {
      isAutoScrollingRef.current = false
    }, 50)

    return () => clearTimeout(timeoutId)
  }, [chat, steps, running, loadingQuery, finalPayload?.finalAnswer])

  useEffect(() => {
    const handleSendChat = (e: CustomEvent<string>) => {
      handleSend(e.detail)
      // Scroll to bottom immediately so the user's attention moves to
      // the thinking pill / incoming results area, regardless of where
      // in the UI the action button was (carousel, detail panel, etc.)
      const container = scrollContainerRef.current
      if (container) {
        container.scrollTo({ top: container.scrollHeight, behavior: "smooth" })
      }
    }
    window.addEventListener("send-chat", handleSendChat as EventListener)
    return () => window.removeEventListener("send-chat", handleSendChat as EventListener)
  }, [handleSend])

  // Cast to the correct types
  const typedSteps = (steps ?? []) as AgentStep[]
  const typedPlannedTasks = (plannedTasks ?? []) as PlannedTask[]

  // EFFETTO: Apertura automatica del link contact_seller (solo vecchio tool non-playwright)
  // Note: contact_seller_playwright invia il messaggio autonomamente e mostra
  // il risultato in-chat via ContactSellerCard — non occorre aprire il browser.
  useEffect(() => {
    if (!running) return
    const timers: ReturnType<typeof setTimeout>[] = []

    // Solo per il vecchio contact_seller (non playwright) che restituisce contact_url
    typedSteps.forEach((step: AgentStep) => {
      const toolName = step.action?.toLowerCase() || ""
      const isLegacyContact = toolName === "contact_seller"
      const obsData = step.observation_data as any
      const resultData = obsData?.data || obsData
      const contactUrl = resultData?.contact_url

      if (isLegacyContact && contactUrl && !openedLinksRef.current.has(step.step.toString())) {
        openedLinksRef.current.add(step.step.toString())
        const t = setTimeout(() => {
          window.open(contactUrl, "_blank")
        }, 300)
        timers.push(t)
      }
    })

    return () => timers.forEach(clearTimeout)
  }, [typedSteps, running])

  // Reset del Set quando la query cambia o finisce
  useEffect(() => {
    if (!running) {
      openedLinksRef.current = new Set()
    }
  }, [running])

  const showWelcome = !hasSearches && chat.length <= 1

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
            <>
              <Box sx={{ ml: { md: "44px" } }}>
                {finalPayload?.visionAnalysis && (
                  <VisionAnalysisCard
                    description={finalPayload.visionAnalysis.description}
                    tags={finalPayload.visionAnalysis.tags}
                    brand={finalPayload.visionAnalysis.brand}
                    condition_clues={finalPayload.visionAnalysis.condition_clues}
                    confidence={finalPayload.visionAnalysis.confidence}
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
                      hideTrace={true}
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
                        contact_seller_playwright: (finalPayload as any).contactSeller,
                        mode: "hybrid"
                      } as any} 
                    />
                  </Box>
                )}
              </Box>

              {finalPayload?.finalAnswer && (
                <MessageBubble role="assistant">
                  {finalPayload.finalAnswer}
                </MessageBubble>
              )}
            </>
          )}

          <div ref={bottomRef} />
        </Box>
      </ErrorBoundary>
    </ChatLayout>
  )
}