import { useEffect, useMemo } from "react"

import type {
  ChatEntry,
  Message,
  SearchBlock
} from "../types/searchTypes.ts"

import { useAgentStream } from "../features/agent/hooks/useAgentStream.ts"
import { useChatStore } from "../features/chat/store/chatStore.ts"

function detectMode(
  resultsCount: number,
  hasSeller: boolean,
  hasComparison: boolean
): "search" | "seller" | "hybrid" | "comparison" {
  if (hasComparison) return "comparison"
  if (resultsCount > 0 && hasSeller) return "hybrid"
  if (hasSeller) return "seller"
  return "search"
}

export function useChatSession() {
  const sessions = useChatStore((state) => state.sessions)
  const activeSessionId = useChatStore((state) => state.activeSessionId)
  const loadingQuery = useChatStore((state) => state.loadingQuery)
  const cache = useChatStore((state) => state.cache)

  const activeSession = useMemo(() => {
    const sid = activeSessionId || sessions[0]?.id
    return sessions.find(s => s.id === sid) || sessions[0]
  }, [sessions, activeSessionId])

  const chat = activeSession?.chat || []

  // Extract all needed actions from store to avoid getState() calls
  const resetConversation = useChatStore((state) => state.resetConversation)
  const setLoadingQuery = useChatStore((state) => state.setLoadingQuery)
  const appendMessage = useChatStore((state) => state.appendMessage)
  const appendAssistantMessage = useChatStore((state) => state.appendAssistantMessage)
  const appendSearchBlock = useChatStore((state) => state.appendSearchBlock)
  const switchSession = useChatStore((state) => state.switchSession)
  const setCachedSearch = useChatStore((state) => state.setCachedSearch)

  const { steps, running, finalPayload, plannedTasks, run, reset } = useAgentStream({
    sessionId: activeSessionId
  })

  // Watch for payload completion
  useEffect(() => {
    if (!finalPayload || running || !loadingQuery) return

    const cacheKey = loadingQuery.toLowerCase()
    
    // Safety check against processing the same payload multiple times
    if (chat.some(entry => entry.type === "search" && entry.search?.query === loadingQuery && entry.search?.analysis === finalPayload.analysis)) {
       return 
    }

    const sellerSummary = finalPayload.sellerSummary || null
    const results = finalPayload.results || []
    const hasComparison = !!finalPayload.comparison

    const mode = detectMode(results.length, !!sellerSummary?.seller_name, hasComparison)

    const newSearch: SearchBlock = {
      query: loadingQuery,
      results,
      analysis: finalPayload.analysis,
      metrics: finalPayload.metrics,
      rag_context: finalPayload.ragContext,
      timings: undefined,
      agent_trace: finalPayload.trace?.length ? finalPayload.trace : [],
      seller_summary: sellerSummary,
      comparison: finalPayload.comparison || null,
      item_details: finalPayload.itemDetails || null,
      shipping_costs: finalPayload.shippingCosts || null,
      metadata: finalPayload.metadata || null,
      final_answer: finalPayload.finalAnswer || "Ho completato l’analisi della richiesta.",
      vision_analysis: finalPayload.visionAnalysis || null,
      mode,
      errors: finalPayload.errors
    }

    const hasStructuredBlock =
      results.length > 0 ||
      !!sellerSummary?.seller_name ||
      !!newSearch.analysis ||
      !!newSearch.agent_trace?.length ||
      !!newSearch.errors?.length ||
      !!newSearch.comparison ||
      !!newSearch.item_details ||
      !!newSearch.shipping_costs ||
      !!newSearch.vision_analysis ||
      !!finalPayload.plannedTasks?.length ||
      (!!finalPayload.toolStates && Object.keys(finalPayload.toolStates).length > 0)

    setCachedSearch(cacheKey, newSearch)

    if (hasStructuredBlock) {
      appendSearchBlock(newSearch)
      if (newSearch.final_answer && newSearch.final_answer !== "Ho completato l’analisi della richiesta.") {
        appendAssistantMessage(newSearch.final_answer)
      }
    } else {
      appendAssistantMessage(newSearch.final_answer ?? "Ho completato l’analisi della richiesta.")
    }

    setLoadingQuery(null)

  }, [finalPayload, running])

  // Ensure an active session is set on mount if missing
  useEffect(() => {
    if (!activeSessionId && sessions.length > 0) {
      switchSession(sessions[0].id)
    }
  }, [activeSessionId, sessions, switchSession])

  const hasSearches = useMemo(
    () => chat.some((entry: ChatEntry) => entry.type === "search"),
    [chat]
  )

  const resetChat = () => {
    reset()
    resetConversation()
  }

  const handleSend = async (text: string, image?: string) => {
    if (!text.trim() && !image) return

    const query = text.trim()
    const cacheKey = (query + (image ? "_img" : "")).toLowerCase()

    const userMessage: Message = {
      role: "user",
      content: query || "Analizza questa immagine",
      image: image // Assicurati che il tipo Message supporti image
    }

    appendMessage(userMessage)

    const cached = cache[cacheKey]

    if (cached) {
      appendAssistantMessage(
        cached.final_answer || "Ho recuperato la risposta dalla cache."
      )
      appendSearchBlock(cached)
      return
    }

    setLoadingQuery(query || "Analisi immagine")
    run(query, image)
  }

  return {
    chat,
    activeSessionId: activeSession?.id,
    steps,
    running,
    finalPayload,
    plannedTasks,
    loadingQuery,
    hasSearches,
    handleSend,
    resetChat
  }
}
