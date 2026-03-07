import { useEffect, useMemo, useRef, useState } from "react"
import { Box, Typography } from "@mui/material"

import ChatLayout from "./component/ChatLayout"
import ChatInput from "./component/ChatInput"
import MessageBubble from "./component/MessageBubble"
import SearchResultList from "./component/SearchResultList"
import AIAnalysisCard from "./component/AIAnalysisCard"
import AIThinkingPipeline from "./component/AIThinkingPipeline"

import { searchProducts } from "./api/searchApi"

import type {
  ChatEntry,
  Message,
  SearchBlock,
  IRMetrics,
  RagContext
} from "./component/searchTypes"

const HISTORY_KEY = "search_history"

function getWelcomeMessage(): ChatEntry {
  return {
    type: "message",
    msg: {
      role: "assistant",
      content:
        "Ciao! Dimmi cosa vuoi cercare su eBay e ti aiuto a trovare i risultati migliori."
    }
  }
}

function safeParseHistory(): Array<{ query: string; results: number }> {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    const parsed = raw ? JSON.parse(raw) : []

    if (!Array.isArray(parsed)) {
      return []
    }

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

function normalizeRagContext(value: unknown): RagContext {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string")
  }

  if (typeof value === "string") {
    return value
  }

  return undefined
}

function normalizeMetrics(value: unknown): IRMetrics | undefined {
  if (!value || typeof value !== "object") {
    return undefined
  }

  return value as IRMetrics
}

export default function App() {
  const [chat, setChat] = useState<ChatEntry[]>([getWelcomeMessage()])
  const [loading, setLoading] = useState(false)
  const [loadingQuery, setLoadingQuery] = useState<string | null>(null)
  const [cache, setCache] = useState<Record<string, SearchBlock>>({})
  const [searchContext, setSearchContext] = useState<Record<string, any>>({})
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end"
    })
  }, [chat, loading])

  const hasSearches = useMemo(
    () => chat.some((entry) => entry.type === "search"),
    [chat]
  )

  const resetChat = () => {
    setChat([getWelcomeMessage()])
    setSearchContext({})
    setLoading(false)
    setLoadingQuery(null)
  }

  const handleSend = async (text: string) => {
    if (loading || !text.trim()) {
      return
    }

    const query = text.trim()
    const cacheKey = query.toLowerCase()

    const userMessage: Message = {
      role: "user",
      content: query
    }

    setChat((prev) => [...prev, { type: "message", msg: userMessage }])

    if (cache[cacheKey]) {
      const cachedSearch = cache[cacheKey]

      saveSearchHistory(query, cachedSearch.results.length)

      setChat((prev) => [
        ...prev,
        {
          type: "message",
          msg: {
            role: "assistant",
            content: `Ho recuperato ${cachedSearch.results.length} risultati dalla cache per "${query}".`
          }
        },
        { type: "search", search: cachedSearch }
      ])

      return
    }

    setLoading(true)
    setLoadingQuery(query)

    try {
      const chatHistory = chat
        .map(entry => {
          if (entry.type === "message") {
            return { role: entry.msg.role, content: entry.msg.content };
          } else if (entry.type === "search" && entry.search.analysis) {
            return { role: "assistant", content: entry.search.analysis };
          }
          return null;
        })
        .filter((msg): msg is { role: string, content: string } => msg !== null);

      const data = await searchProducts(query, chatHistory, searchContext)

      if (data.parsed_query) {
        setSearchContext(data.parsed_query)
      }

      const results = Array.isArray(data.results) ? data.results : []
      const analysis =
        typeof data.analysis === "string" ? data.analysis : null
      const metrics = normalizeMetrics(data.metrics)
      const ragContext = normalizeRagContext(data.rag_context)
      const timings =
        data._timings && typeof data._timings === "object"
          ? (data._timings as Record<string, number>)
          : undefined

      const newSearch: SearchBlock = {
        query,
        results,
        analysis,
        metrics,
        rag_context: ragContext,
        timings,
        thinking_trace: data.thinking_trace
      }

      setCache((prev) => ({
        ...prev,
        [cacheKey]: newSearch
      }))

      saveSearchHistory(query, results.length)

      const isConversational = results.length === 0 && (!data.thinking_trace || data.thinking_trace.length === 0);

      if (isConversational) {
        setChat((prev) => [
          ...prev,
          {
            type: "message",
            msg: {
              role: "assistant",
              content: analysis || "Nessuna risposta."
            }
          }
        ]);
      } else {
        setChat((prev) => [
          ...prev,
          { type: "search", search: newSearch }
        ]);
      }
    } catch (error) {
      console.error("Search error:", error)

      setChat((prev) => [
        ...prev,
        {
          type: "message",
          msg: {
            role: "assistant",
            content:
              "C'è stato un errore durante la ricerca. Controlla backend e endpoint, poi riprova."
          }
        }
      ])
    } finally {
      setLoading(false)
      setLoadingQuery(null)
    }
  }

  return (
    <ChatLayout onSearch={handleSend} onNewChat={resetChat}>
      <Box
        sx={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
          width: "100%",
          px: { xs: 2, md: 4 },
          py: 4,
          position: "relative"
        }}
      >
        <Box sx={{ width: "100%", maxWidth: 1000 }}>
          {!hasSearches && chat.length <= 1 && (
            <Box sx={{ px: 2, py: 4 }}>
              <Typography
                sx={{
                  fontSize: 28,
                  fontWeight: 700,
                  mb: 1.5,
                  color: "#202123"
                }}
              >
                Ricerca conversazionale per eBay
              </Typography>

              <Typography
                sx={{
                  color: "#6e6e80",
                  fontSize: 15
                }}
              >
                Prova con query come “iPhone 13 massimo 700 euro” oppure
                “notebook Lenovo business con ottima affidabilità”.
              </Typography>
            </Box>
          )}

          {chat.map((entry, index) => {
            if (entry.type === "message") {
              return (
                <Box key={`msg-${index}`} mb={2}>
                  <MessageBubble
                    role={entry.msg.role}
                    timestamp={entry.msg.timestamp}
                  >
                    {entry.msg.content}
                  </MessageBubble>
                </Box>
              )
            }

            const search = entry.search

            return (
              <Box key={`search-${index}`} mt={2} mb={4}>
                {search.thinking_trace && search.thinking_trace.length > 0 && (
                  <Box mb={2}>
                    <MessageBubble role="assistant">
                      <AIThinkingPipeline trace={search.thinking_trace} />
                    </MessageBubble>
                  </Box>
                )}
                {(search.analysis || search.metrics || search.rag_context) && (
                  <AIAnalysisCard
                    text={search.analysis ?? undefined}
                    metrics={search.metrics}
                    rag_context={search.rag_context}
                  />
                )}

                {search.results.length > 0 && (
                  <SearchResultList results={search.results} />
                )}
              </Box>
            )
          })}

          {loading && (
            <Box mt={2} mb={3}>
              <MessageBubble role="assistant">
                <AIThinkingPipeline
                  loading
                  query={loadingQuery ?? undefined}
                />
              </MessageBubble>
            </Box>
          )}

          <div ref={bottomRef} />
        </Box>
      </Box>

      <ChatInput onSend={handleSend} disabled={loading} />
    </ChatLayout>
  )
}