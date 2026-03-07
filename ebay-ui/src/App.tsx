import { useEffect, useMemo, useRef, useState } from "react"
import { Box, Typography } from "@mui/material"

import ChatLayout from "./component/ChatLayout"
import ChatInput from "./component/ChatInput"
import MessageBubble from "./component/MessageBubble"
import SearchResultList from "./component/SearchResultList"
import AIAnalysisCard from "./component/AIAnalysisCard"
import AIThinkingPipeline from "./component/AIThinkingPipeline"
import SellerFeedbackPanel from "./component/SellerFeedbackPanel"

import { useAgentStream } from "./hooks/useAgentStream"

import type {
  ChatEntry,
  Message,
  SearchBlock
} from "./component/searchTypes"

const HISTORY_KEY = "search_history"

function getWelcomeMessage(): ChatEntry {
  return {
    type: "message",
    msg: {
      role: "assistant",
      content:
        "Ciao! Sono ebayGPT. Posso cercare prodotti, spiegarti il ranking e controllare se il venditore del risultato migliore è affidabile."
    }
  }
}

function safeParseHistory(): Array<{ query: string; results: number }> {
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

export default function App() {
  const [chat, setChat] = useState<ChatEntry[]>([getWelcomeMessage()])
  const [loadingQuery, setLoadingQuery] = useState<string | null>(null)
  const [cache, setCache] = useState<Record<string, SearchBlock>>({})

  const bottomRef = useRef<HTMLDivElement | null>(null)

  const {
    steps,
    results,
    running,
    run
  } = useAgentStream()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end"
    })
  }, [chat, steps])

  const hasSearches = useMemo(
    () => chat.some((entry) => entry.type === "search"),
    [chat]
  )

  const resetChat = () => {
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
            content: cached.final_answer || "Risultati recuperati dalla cache."
          }
        },
        { type: "search", search: cached }
      ])

      return
    }

    setLoadingQuery(query)

    run(query)
  }

  /**
   * Quando arrivano risultati finali dallo stream
   */

  useEffect(() => {

    if (!results || results.length === 0) return

    const query = loadingQuery || "query"
    const cacheKey = query.toLowerCase()

    const newSearch: SearchBlock = {
      query,
      results,
      analysis: null,
      metrics: undefined,
      rag_context: undefined,
      timings: undefined,
      agent_trace: steps,
      seller_summary: null,
      final_answer: `Ho trovato ${results.length} risultati per "${query}".`
    }

    setCache((prev) => ({
      ...prev,
      [cacheKey]: newSearch
    }))

    saveSearchHistory(query, results.length)

    setChat((prev) => [
      ...prev,
      {
        type: "message",
        msg: {
          role: "assistant",
          content: newSearch.final_answer ?? ""
        }
      },
      { type: "search", search: newSearch }
    ])

  }, [results])

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
          py: 4
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
                ebayGPT · ricerca conversazionale agentica
              </Typography>

              <Typography sx={{ color: "#6e6e80", fontSize: 15 }}>
                Prova con query come “iPhone 13 massimo 700 euro e controlla il venditore”.
              </Typography>
            </Box>
          )}

          {chat.map((entry, index) => {

            if (entry.type === "message") {
              return (
                <Box key={`msg-${index}`} mb={2}>
                  <MessageBubble role={entry.msg.role}>
                    {entry.msg.content}
                  </MessageBubble>
                </Box>
              )
            }

            const search = entry.search

            return (
              <Box key={`search-${index}`} mt={2} mb={4}>

                {search.agent_trace?.length ? (
                  <AIThinkingPipeline
                    agentTrace={search.agent_trace}
                    query={search.query}
                  />
                ) : null}

                {search.analysis && (
                  <AIAnalysisCard text={search.analysis} />
                )}

                {search.results.length > 0 ? (
                  <SearchResultList results={search.results} />
                ) : null}

                {search.seller_summary?.seller_name && (
                  <Box
                    sx={{
                      mt: 3,
                      p: 3,
                      borderRadius: "16px",
                      bgcolor: "#fff",
                      border: "1px solid #e5e5e5"
                    }}
                  >
                    <Typography sx={{ fontSize: 16, fontWeight: 700 }}>
                      Seller deep dive
                    </Typography>

                    <SellerFeedbackPanel
                      seller={search.seller_summary.seller_name}
                    />
                  </Box>
                )}

              </Box>
            )
          })}

          {running && (
            <Box mt={2} mb={3}>
              <MessageBubble role="assistant">
                <AIThinkingPipeline
                  agentTrace={steps}
                  query={loadingQuery ?? undefined}
                />
              </MessageBubble>
            </Box>
          )}

          <div ref={bottomRef} />
        </Box>
      </Box>

      <ChatInput onSend={handleSend} disabled={running} />
    </ChatLayout>
  )
}