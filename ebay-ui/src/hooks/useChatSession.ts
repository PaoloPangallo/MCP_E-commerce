import { useEffect, useMemo, useState } from "react"
import { useAgentStream } from "./useAgentStream"

import type {
  ChatEntry,
  Message,
  SearchBlock
} from "../component/searchTypes"

const HISTORY_KEY = "search_history"

type HistoryItem = {
  query: string
  results: number
}

function getWelcomeMessage(): ChatEntry {
  return {
    type: "message",
    msg: {
      role: "assistant",
      content:
        "Ciao! Sono ebayGPT. Posso cercare prodotti, confrontare risultati, spiegare il ranking e analizzare l’affidabilità di un venditore eBay."
    }
  }
}

function safeParseHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    const parsed = raw ? JSON.parse(raw) : []

    if (!Array.isArray(parsed)) return []

    return parsed.filter(
      (item) =>
        item &&
        typeof item.query === "string" &&
        typeof item.results === "number"
    )
  } catch {
    return []
  }
}

function saveSearchHistory(query: string, resultsCount = 0) {
  const history = safeParseHistory()

  const updated = [
    { query, results: resultsCount },
    ...history.filter(
      (item) => item.query.toLowerCase() !== query.toLowerCase()
    )
  ].slice(0, 20)

  localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
  window.dispatchEvent(new Event("search_history_updated"))
}

function detectMode(
  resultsCount: number,
  hasSeller: boolean
): "search" | "seller" | "hybrid" {
  if (resultsCount > 0 && hasSeller) return "hybrid"
  if (hasSeller) return "seller"
  return "search"
}

export function useChatSession() {
  const [chat, setChat] = useState<ChatEntry[]>([getWelcomeMessage()])
  const [loadingQuery, setLoadingQuery] = useState<string | null>(null)
  const [cache, setCache] = useState<Record<string, SearchBlock>>({})

  const {
    steps,
    running,
    finalPayload,
    run,
    reset
  } = useAgentStream()

  const hasSearches = useMemo(
    () => chat.some((entry) => entry.type === "search"),
    [chat]
  )

  const resetChat = () => {
    reset()
    setChat([getWelcomeMessage()])
    setLoadingQuery(null)
  }

  const handleSend = async (text: string) => {
    if (!text.trim()) return

    const query = text.trim()
    const cacheKey = query.toLowerCase()

    const userMessage: Message = {
      role: "user",
      content: query
    }

    setChat((prev) => [...prev, { type: "message", msg: userMessage }])

    if (cache[cacheKey]) {
      const cached = cache[cacheKey]

      saveSearchHistory(query, cached.results.length)

      setChat((prev) => [
        ...prev,
        {
          type: "message",
          msg: {
            role: "assistant",
            content:
              cached.final_answer || "Ho recuperato la risposta dalla cache."
          }
        },
        { type: "search", search: cached }
      ])

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
      !!newSearch.errors?.length

    setCache((prev) => ({
      ...prev,
      [cacheKey]: newSearch
    }))

    saveSearchHistory(query, results.length)

    if (hasStructuredBlock) {
      setChat((prev) => [
        ...prev,
        {
          type: "message",
          msg: {
            role: "assistant",
            content: newSearch.final_answer ?? "Analisi completata."
          }
        },
        { type: "search", search: newSearch }
      ])
    } else {
      setChat((prev) => [
        ...prev,
        {
          type: "message",
          msg: {
            role: "assistant",
            content:
              newSearch.final_answer ??
              "Ho completato l’analisi della richiesta."
          }
        }
      ])
    }

    setLoadingQuery(null)
  }, [finalPayload, loadingQuery, steps])

  return {
    chat,
    steps,
    running,
    finalPayload,
    loadingQuery,
    hasSearches,
    handleSend,
    resetChat
  }
}