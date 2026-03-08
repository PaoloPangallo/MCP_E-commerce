import { useEffect, useMemo } from "react"

import type {
  ChatEntry,
  Message,
  SearchBlock
} from "../types/searchTypes.ts"

import { useAgentStream } from "../features/agent/hooks/useAgentStream.ts"
import { useChatStore } from "../features/chat/store/chatStore.ts"
import { useSidebarStore } from "../features/chat/store/sidebarStore.ts"

function detectMode(
  resultsCount: number,
  hasSeller: boolean
): "search" | "seller" | "hybrid" {
  if (resultsCount > 0 && hasSeller) return "hybrid"
  if (hasSeller) return "seller"
  return "search"
}

export function useChatSession() {
  const chat = useChatStore((state) => state.chat)
  const loadingQuery = useChatStore((state) => state.loadingQuery)
  const cache = useChatStore((state) => state.cache)

  const resetConversation = useChatStore((state) => state.resetConversation)
  const setLoadingQuery = useChatStore((state) => state.setLoadingQuery)
  const appendMessage = useChatStore((state) => state.appendMessage)
  const appendAssistantMessage = useChatStore(
    (state) => state.appendAssistantMessage
  )
  const appendSearchBlock = useChatStore((state) => state.appendSearchBlock)
  const setCachedSearch = useChatStore((state) => state.setCachedSearch)

  const addHistory = useSidebarStore((state) => state.addHistory)

  const { steps, running, finalPayload, plannedTasks, run, reset } = useAgentStream()

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
      addHistory({ query, results: cached.results.length })

      appendAssistantMessage(
        cached.final_answer || "Ho recuperato la risposta dalla cache."
      )
      appendSearchBlock(cached)
      return
    }

    setLoadingQuery(query)
    run(query)
  }

  useEffect(() => {
    if (!finalPayload || !loadingQuery) return

    const query = loadingQuery
    const cacheKey = query.toLowerCase()

    const sellerSummary = finalPayload.sellerSummary || null
    const results = finalPayload.results || []

    const mode = detectMode(results.length, !!sellerSummary?.seller_name)

    const newSearch: SearchBlock = {
      query,
      results,
      analysis: finalPayload.analysis,
      metrics: finalPayload.metrics,
      rag_context: finalPayload.ragContext,
      timings: undefined,
      agent_trace: finalPayload.trace?.length ? finalPayload.trace : steps,
      seller_summary: sellerSummary,
      final_answer:
        finalPayload.finalAnswer ||
        "Ho completato l’analisi della richiesta.",
      mode,
      errors: finalPayload.errors
    }

    const hasStructuredBlock =
      results.length > 0 ||
      !!sellerSummary?.seller_name ||
      !!newSearch.analysis ||
      !!newSearch.agent_trace?.length ||
      !!newSearch.errors?.length ||
      !!finalPayload.plannedTasks?.length ||
      !!finalPayload.toolStates &&
        Object.keys(finalPayload.toolStates).length > 0

    setCachedSearch(cacheKey, newSearch)
    addHistory({ query, results: results.length })

    appendAssistantMessage(
      newSearch.final_answer ?? "Ho completato l’analisi della richiesta."
    )

    if (hasStructuredBlock) {
      appendSearchBlock(newSearch)
    }

    setLoadingQuery(null)
  }, [
    finalPayload,
    loadingQuery,
    steps,
    setCachedSearch,
    addHistory,
    appendAssistantMessage,
    appendSearchBlock,
    setLoadingQuery
  ])

  return {
  chat,
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