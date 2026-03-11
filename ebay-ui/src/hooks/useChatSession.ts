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

  const resetConversation = useChatStore((state) => state.resetConversation)
  const setLoadingQuery = useChatStore((state) => state.setLoadingQuery)
  const appendMessage = useChatStore((state) => state.appendMessage)
  const appendAssistantMessage = useChatStore(
    (state) => state.appendAssistantMessage
  )
  const appendSearchBlock = useChatStore((state) => state.appendSearchBlock)
  const switchSession = useChatStore((state) => state.switchSession)

  const { steps, running, finalPayload, plannedTasks, run, reset } = useAgentStream({
    onDone: (payload, query) => {
      const cacheKey = query.toLowerCase()

      const sellerSummary = payload.sellerSummary || null
      const results = payload.results || []
      const hasComparison = !!payload.comparison

      const mode = detectMode(results.length, !!sellerSummary?.seller_name, hasComparison)

      const newSearch: SearchBlock = {
        query,
        results,
        analysis: payload.analysis,
        metrics: payload.metrics,
        rag_context: payload.ragContext,
        timings: undefined,
        agent_trace: payload.trace?.length ? payload.trace : [], // Fixed to use the reliable payload trace directly
        seller_summary: sellerSummary,
        comparison: payload.comparison || null,
        final_answer:
          payload.finalAnswer ||
          "Ho completato l’analisi della richiesta.",
        mode,
        errors: payload.errors
      }

      const hasStructuredBlock =
        results.length > 0 ||
        !!sellerSummary?.seller_name ||
        !!newSearch.analysis ||
        !!newSearch.agent_trace?.length ||
        !!newSearch.errors?.length ||
        !!newSearch.comparison ||
        !!payload.plannedTasks?.length ||
        !!payload.toolStates &&
        Object.keys(payload.toolStates).length > 0

      useChatStore.getState().setCachedSearch(cacheKey, newSearch)

      useChatStore.getState().appendAssistantMessage(
        newSearch.final_answer ?? "Ho completato l’analisi della richiesta."
      )

      if (hasStructuredBlock) {
        useChatStore.getState().appendSearchBlock(newSearch)
      }

      useChatStore.getState().setLoadingQuery(null)
    }
  })

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

  const handleSend = async (text: string) => {
    if (!text.trim()) return

    const query = text.trim()
    const cacheKey = query.toLowerCase()

    const userMessage: Message = {
      role: "user",
      content: query
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

    setLoadingQuery(query)
    run(query)
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
