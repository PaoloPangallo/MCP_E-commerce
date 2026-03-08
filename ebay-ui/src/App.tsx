import { useEffect, useMemo, useRef, useState } from "react"
import { Box, Typography, Paper, Chip } from "@mui/material"

import ChatLayout from "./component/ChatLayout"
import ChatInput from "./component/ChatInput"
import MessageBubble from "./component/MessageBubble"
import SearchResultList from "./component/SearchResultList"
import AIThinkingPipeline from "./component/AIThinkingPipeline"
import SellerFeedbackPanel from "./component/SellerFeedbackPanel"
import SellerTrustGauge from "./component/SellerTrustGauge"

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
        "Ciao! Sono ebayGPT. Posso cercare prodotti, confrontare risultati, spiegare il ranking e analizzare l’affidabilità di un venditore eBay."
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

function detectMode(resultsCount: number, hasSeller: boolean): "search" | "seller" | "hybrid" {
  if (resultsCount > 0 && hasSeller) return "hybrid"
  if (hasSeller) return "seller"
  return "search"
}

function SellerSummaryCard({
  sellerName,
  trustScore,
  sentimentScore,
  count
}: {
  sellerName?: string
  trustScore?: number
  sentimentScore?: number
  count?: number
}) {
  if (!sellerName) return null

  return (
    <Paper
      elevation={0}
      sx={{
        mt: 3,
        p: 3,
        borderRadius: "16px",
        bgcolor: "#fff",
        border: "1px solid #e5e5e5"
      }}
    >
      <Box display="flex" alignItems="center" gap={1} mb={1.25} flexWrap="wrap">
        <Typography sx={{ fontSize: 18, fontWeight: 700, color: "#202123" }}>
          Seller deep dive
        </Typography>

        <Chip
          label={sellerName}
          size="small"
          sx={{
            bgcolor: "#f5f5f5",
            border: "1px solid #e5e5e5"
          }}
        />
      </Box>

      {typeof trustScore === "number" && (
        <SellerTrustGauge score={trustScore} />
      )}

      <Box display="flex" gap={1} flexWrap="wrap" mt={1}>
        {typeof sentimentScore === "number" && (
          <Chip
            label={`Sentiment ${Math.round(sentimentScore * 100)}%`}
            size="small"
            sx={{ bgcolor: "#f5f5f5", border: "1px solid #e5e5e5" }}
          />
        )}

        {typeof count === "number" && (
          <Chip
            label={`${count} feedback analizzati`}
            size="small"
            sx={{ bgcolor: "#f5f5f5", border: "1px solid #e5e5e5" }}
          />
        )}
      </Box>

      <SellerFeedbackPanel seller={sellerName} />
    </Paper>
  )
}

export default function App() {
  const [chat, setChat] = useState<ChatEntry[]>([getWelcomeMessage()])
  const [loadingQuery, setLoadingQuery] = useState<string | null>(null)
  const [cache, setCache] = useState<Record<string, SearchBlock>>({})

  const bottomRef = useRef<HTMLDivElement | null>(null)

  const {
    steps,
    running,
    finalPayload,
    run,
    reset
  } = useAgentStream()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end"
    })
  }, [chat, steps, finalPayload, running])

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
            content: cached.final_answer || "Ho recuperato la risposta dalla cache."
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
          content: newSearch.final_answer ?? "Analisi completata."
        }
      },
      { type: "search", search: newSearch }
    ])

    setLoadingQuery(null)
  }, [finalPayload, loadingQuery, steps])

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
                  fontSize: 30,
                  fontWeight: 800,
                  mb: 1.5,
                  color: "#202123"
                }}
              >
                ebayGPT
              </Typography>

              <Typography sx={{ color: "#6e6e80", fontSize: 15, mb: 2 }}>
                Ricerca conversazionale agentica per prodotti e venditori eBay.
              </Typography>

              <Box display="flex" gap={1} flexWrap="wrap">
                <Chip label="product search" size="small" />
                <Chip label="seller trust" size="small" />
                <Chip label="agent trace" size="small" />
                <Chip label="ranking explanation" size="small" />
              </Box>
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

                {search.mode !== "seller" && search.results.length > 0 ? (
                  <SearchResultList results={search.results} />
                ) : null}

                {search.mode === "seller" && search.seller_summary?.seller_name ? (
                  <SellerSummaryCard
                    sellerName={search.seller_summary.seller_name}
                    trustScore={search.seller_summary.trust_score}
                    sentimentScore={search.seller_summary.sentiment_score}
                    count={search.seller_summary.count}
                  />
                ) : null}

                {search.mode === "hybrid" && search.seller_summary?.seller_name ? (
                  <SellerSummaryCard
                    sellerName={search.seller_summary.seller_name}
                    trustScore={search.seller_summary.trust_score}
                    sentimentScore={search.seller_summary.sentiment_score}
                    count={search.seller_summary.count}
                  />
                ) : null}

                {search.errors && search.errors.length > 0 && (
                  <Paper
                    elevation={0}
                    sx={{
                      mt: 2,
                      p: 2,
                      borderRadius: 2,
                      bgcolor: "#fff7f7",
                      border: "1px solid #f1cccc"
                    }}
                  >
                    <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#a33" }}>
                      Errori o segnali backend
                    </Typography>

                    {search.errors.map((err, idx) => (
                      <Typography
                        key={`${err}-${idx}`}
                        sx={{ fontSize: 12.5, color: "#7a4b4b", mt: 0.5 }}
                      >
                        {err}
                      </Typography>
                    ))}
                  </Paper>
                )}
              </Box>
            )
          })}

          {running && (
            <Box mt={2} mb={3}>
              <MessageBubble role="assistant" isTyping={false}>
                <AIThinkingPipeline
                  agentTrace={steps}
                  loading
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