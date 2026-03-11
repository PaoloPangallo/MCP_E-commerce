import { useEffect, useMemo, useRef } from "react"
import { Box, Chip, Paper, Typography, Avatar, Divider } from "@mui/material"
import VerifiedIcon from "@mui/icons-material/Verified"
import StarIcon from "@mui/icons-material/Star"

import { useChatSession } from "../../hooks/useChatSession.ts"
import type { SearchBlock } from "../search/types.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx"
import AIAnalysisCard from "../agent/components/AIAnalysisCard.tsx"
import ChatLayout from "./ChatLayout.tsx"
import SearchResultList from "../search/components/SearchResultList.tsx"
import ChatInput from "./ChatInput.tsx"
import SellerFeedbackPanel from "../seller/component/SellerFeedbackPanel.tsx"
import ComparisonDisplay from "../search/components/ComparisonDisplay.tsx"
import MessageBubble from "./MessageBubble.tsx";

function SellerSummaryCard({
  sellerName,
  trustScore,
  sentimentScore,
  count,
  feedbacks = []
}: {
  sellerName?: string
  trustScore?: number
  sentimentScore?: number
  count?: number
  feedbacks?: any[]
}) {
  if (!sellerName) return null

  // Theme logic based on trust
  const isTopTier = (trustScore ?? 0) >= 0.85
  const isGoodTier = (trustScore ?? 0) >= 0.70

  const themeColor = isTopTier ? "#f59e0b" : isGoodTier ? "#3b82f6" : "#64748b"
  const gradient = isTopTier
    ? "radial-gradient(circle at 20% 20%, #fffbeb 0%, #fef3c7 50%, #fff7ed 100%)"
    : isGoodTier
      ? "radial-gradient(circle at 20% 20%, #eff6ff 0%, #dbeafe 50%, #f8fafc 100%)"
      : "radial-gradient(circle at 20% 20%, #f8fafc 0%, #f1f5f9 50%, #ffffff 100%)"

  return (
    <Paper
      elevation={0}
      sx={{
        mt: 3,
        borderRadius: 5,
        overflow: "hidden",
        bgcolor: "#ffffff",
        border: "1px solid #e2e8f0",
        boxShadow: "0 4px 20px -5px rgba(0,0,0,0.05)"
      }}
    >
      {/* Hero Section */}
      <Box sx={{
        p: { xs: 3, md: 4 },
        background: gradient,
        position: "relative",
        borderBottom: "1px solid rgba(226, 232, 240, 0.8)"
      }}>
        <Box display="flex" alignItems="center" gap={3} flexWrap="wrap">
          <Box sx={{ position: "relative" }}>
            <Avatar
              sx={{
                width: 80,
                height: 80,
                fontSize: 32,
                fontWeight: 900,
                bgcolor: themeColor,
                boxShadow: `0 8px 16px -4px ${themeColor}44`,
                border: "4px solid #ffffff"
              }}
            >
              {sellerName.charAt(0).toUpperCase()}
            </Avatar>
            {isTopTier && (
              <Box sx={{
                position: "absolute",
                bottom: -4,
                right: -4,
                bgcolor: "#ffffff",
                borderRadius: "50%",
                display: "flex",
                p: 0.5,
                boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
              }}>
                <VerifiedIcon sx={{ color: "#f59e0b", fontSize: 22 }} />
              </Box>
            )}
          </Box>

          <Box flex={1} minWidth={200}>
            <Box display="flex" alignItems="center" gap={1} mb={0.5}>
              <Typography sx={{ fontSize: 24, fontWeight: 900, color: "#0f172a", letterSpacing: "-0.02em" }}>
                {sellerName}
              </Typography>
              {isTopTier && (
                <Chip
                  label="TOP SELLER"
                  size="small"
                  sx={{
                    bgcolor: "#fef3c7",
                    color: "#92400e",
                    fontWeight: 800,
                    fontSize: 10,
                    height: 20
                  }}
                />
              )}
            </Box>
            <Typography sx={{ fontSize: 14, color: "#475569", fontWeight: 500 }}>
              Analisi completa dell'affidabilità commerciale
            </Typography>

            <Box display="flex" alignItems="center" gap={2} mt={2}>
              <Box>
                <Typography sx={{ fontSize: 10, fontWeight: 800, color: themeColor, textTransform: "uppercase", letterSpacing: 1 }}>
                  Trust Score
                </Typography>
                <Typography sx={{ fontSize: 20, fontWeight: 900, color: "#0f172a" }}>
                  {Math.round((trustScore ?? 0) * 100)}%
                </Typography>
              </Box>
              <Divider orientation="vertical" flexItem sx={{ height: 24, alignSelf: "center", mx: 1 }} />
              <Box>
                <Typography sx={{ fontSize: 10, fontWeight: 800, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>
                  Sentiment
                </Typography>
                <Typography sx={{ fontSize: 20, fontWeight: 900, color: "#0f172a" }}>
                  {Math.round((sentimentScore ?? 0) * 100)}%
                </Typography>
              </Box>
              <Divider orientation="vertical" flexItem sx={{ height: 24, alignSelf: "center", mx: 1 }} />
              <Box>
                <Typography sx={{ fontSize: 10, fontWeight: 800, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>
                  Analizzati
                </Typography>
                <Typography sx={{ fontSize: 20, fontWeight: 900, color: "#0f172a" }}>
                  {count}
                </Typography>
              </Box>
            </Box>
          </Box>

          {/* Quick trust indicator */}
          <Box sx={{
            display: { xs: "none", lg: "flex" },
            flexDirection: "column",
            alignItems: "center",
            p: 2,
            borderRadius: 4,
            bgcolor: "rgba(255,255,255,0.6)",
            backdropFilter: "blur(8px)",
            border: "1px solid rgba(255,255,255,0.4)"
          }}>
            <StarIcon sx={{ color: themeColor, fontSize: 32, mb: 1 }} />
            <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#0f172a" }}>
              {isTopTier ? "MOLTO AFFIDABILE" : isGoodTier ? "AFFIDABILE" : "DA VERIFICARE"}
            </Typography>
          </Box>
        </Box>

        {/* Golden Feedback Previews & Highlights */}
        {feedbacks.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography sx={{ fontSize: 10, fontWeight: 800, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, mb: 1.5 }}>
              Analisi Punti Chiave (Top 50)
            </Typography>
            <Box sx={{ display: "flex", gap: 2, overflowX: "auto", pb: 1, "&::-webkit-scrollbar": { display: "none" } }}>
              {(() => {
                // Sort feedbacks: prioritizza sentiment, poi rating
                const sorted = [...feedbacks].sort((a, b) => {
                  if (typeof a.sentiment === "number" && typeof b.sentiment === "number") {
                    return b.sentiment - a.sentiment
                  }
                  return (b.rating ?? 0) - (a.rating ?? 0)
                })

                const best = sorted[0]
                const worst = sorted[sorted.length - 1]
                const hasContrast = sorted.length > 1 && best !== worst

                return (
                  <>
                    {/* MIGLIORE */}
                    <Box sx={{
                      minWidth: 280,
                      p: 2,
                      borderRadius: 3,
                      bgcolor: "rgba(236, 253, 245, 0.5)",
                      backdropFilter: "blur(12px)",
                      border: "1px solid #10b98144",
                      boxShadow: "0 4px 12px rgba(0,0,0,0.02)"
                    }}>
                      <Box display="flex" alignItems="center" gap={1} mb={1}>
                        <VerifiedIcon sx={{ color: "#10b981", fontSize: 14 }} />
                        <Typography sx={{ fontSize: 11, fontWeight: 800, color: "#065f46" }}>IL MEGLIO</Typography>
                      </Box>
                      <Typography sx={{
                        fontSize: 12,
                        color: "#065f46",
                        fontStyle: "italic",
                        lineHeight: 1.5,
                        display: "-webkit-box",
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden"
                      }}>
                        "{best.comment}"
                      </Typography>
                    </Box>

                    {/* PEGGIORE (solo se c'è contrasto sufficiente o se è negativo) */}
                    {hasContrast && (
                      <Box sx={{
                        minWidth: 280,
                        p: 2,
                        borderRadius: 3,
                        bgcolor: "rgba(255, 241, 242, 0.5)",
                        backdropFilter: "blur(12px)",
                        border: "1px solid #f43f5e44",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.02)"
                      }}>
                        <Box display="flex" alignItems="center" gap={1} mb={1}>
                          <StarIcon sx={{ color: "#f43f5e", fontSize: 14 }} />
                          <Typography sx={{ fontSize: 11, fontWeight: 800, color: "#9f1239" }}>DA ATTENZIONARE</Typography>
                        </Box>
                        <Typography sx={{
                          fontSize: 12,
                          color: "#9f1239",
                          fontStyle: "italic",
                          lineHeight: 1.5,
                          display: "-webkit-box",
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden"
                        }}>
                          "{worst.comment}"
                        </Typography>
                      </Box>
                    )}
                  </>
                )
              })()}
            </Box>
          </Box>
        )}
      </Box>

      {/* Main Gauges & Detailed Panel */}
      <Box sx={{ p: 3 }}>
        <SellerFeedbackPanel seller={sellerName} />
      </Box>
    </Paper>
  )
}

function SearchBlockView({ search }: { search: SearchBlock }) {
  const hasSeller = !!search.seller_summary?.seller_name
  const hasResults = Array.isArray(search.results) && search.results.length > 0
  const hasAnalysis = !!search.analysis || !!search.metrics || !!search.rag_context
  const hasComparison = !!search.comparison && Array.isArray(search.comparison.comparison_matrix) && search.comparison.comparison_matrix.length > 0
  const hasTrace = Array.isArray(search.agent_trace) && search.agent_trace.length > 0

  const showSellerCard =
    hasSeller && (search.mode === "seller" || search.mode === "hybrid")

  return (
    <Box sx={{ mt: 1.5, mb: 4 }}>
      {hasTrace ? (
        <Paper
          elevation={0}
          sx={{
            p: 2.5,
            borderRadius: 4,
            border: "1px solid #e5e7eb",
            bgcolor: "#ffffff"
          }}
        >
          <AIThinkingPipeline
            agentTrace={search.agent_trace}
            query={search.query}
          />
        </Paper>
      ) : null}

      {hasAnalysis ? (
        <Box mt={2.5}>
          <AIAnalysisCard
            text={search.analysis ?? undefined}
            metrics={search.metrics}
            rag_context={search.rag_context}
          />
        </Box>
      ) : null}

      {search.mode !== "seller" && hasResults ? (
        <Box mt={2.5}>
          <SearchResultList results={search.results} />
        </Box>
      ) : null}

      {showSellerCard ? (
        <SellerSummaryCard
          sellerName={search.seller_summary?.seller_name}
          trustScore={search.seller_summary?.trust_score}
          sentimentScore={search.seller_summary?.sentiment_score}
          count={search.seller_summary?.count}
          feedbacks={search.seller_summary?.feedbacks}
        />
      ) : null}

      {hasComparison ? (
        <Box mt={2.5}>
          <ComparisonDisplay data={search.comparison!} />
        </Box>
      ) : null}

      {search.errors && search.errors.length > 0 ? (
        <Paper
          elevation={0}
          sx={{
            mt: 2.5,
            p: 2.25,
            borderRadius: 3,
            bgcolor: "#fff7f7",
            border: "1px solid #f2d6d6"
          }}
        >
          <Typography
            sx={{
              fontSize: 13,
              fontWeight: 700,
              color: "#9f2d2d",
              mb: 0.75
            }}
          >
            Errori o segnali backend
          </Typography>

          {search.errors.map((err, idx) => (
            <Typography
              key={`${err}-${idx}`}
              sx={{ fontSize: 13, color: "#7a4b4b", lineHeight: 1.6 }}
            >
              {err}
            </Typography>
          ))}
        </Paper>
      ) : null}
    </Box>
  )
}

function WelcomePanel() {
  const suggestions = useMemo(
    () => [
      "iPhone 13 massimo 700 euro",
      "cerca una maglia Inter e controlla il venditore",
      "analizza il seller pegaso_italia",
      "fammi un confronto tra i migliori risultati per Nintendo Switch"
    ],
    []
  )

  return (
    <Box
      sx={{
        maxWidth: 760,
        mx: "auto",
        px: { xs: 2, md: 3 },
        pt: { xs: 8, md: 12 },
        pb: 6
      }}
    >
      <Typography
        sx={{
          fontSize: { xs: 30, md: 38 },
          fontWeight: 700,
          color: "#111827",
          letterSpacing: "-0.02em",
          mb: 1
        }}
      >
        Cosa vuoi cercare oggi?
      </Typography>

      <Typography
        sx={{
          fontSize: 15,
          color: "#6b7280",
          lineHeight: 1.75,
          maxWidth: 700
        }}
      >
        ebayGPT unisce product search, ranking explanation e seller trust analysis
        in un flusso conversazionale unico.
      </Typography>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 3 }}>
        {suggestions.map((item) => (
          <Chip
            key={item}
            label={item}
            sx={{
              bgcolor: "#ffffff",
              border: "1px solid #e5e7eb",
              fontSize: 13
            }}
          />
        ))}
      </Box>
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
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end"
    })
  }, [chat, steps, running, loadingQuery])

  const showWelcome = !hasSearches && chat.length <= 1

  return (
    <ChatLayout
      onSearch={handleSend}
      onNewChat={resetChat}
      composer={<ChatInput onSend={handleSend} disabled={running} />}
    >
      <Box
        sx={{
          width: "100%",
          maxWidth: 880,
          mx: "auto",
          px: { xs: 2, md: 3 },
          pt: { xs: 3, md: 4 },
          pb: 2
        }}
      >
        {showWelcome ? <WelcomePanel /> : null}

        {chat.map((entry, index) =>
          entry.type === "message" ? (
            <Box key={`msg-${index}`} mb={2.5}>
              <MessageBubble role={entry.msg.role}>
                {entry.msg.content}
              </MessageBubble>
            </Box>
          ) : (
            <Box key={`search-${index}`}>
              <SearchBlockView search={entry.search} />
            </Box>
          )
        )}

        {running ? (
          <Box mt={1.5} mb={3}>
            <MessageBubble role="assistant">
              <Paper
                elevation={0}
                sx={{
                  p: 2.25,
                  borderRadius: 4,
                  border: "1px solid #e5e7eb",
                  bgcolor: "#ffffff"
                }}
              >
                <AIThinkingPipeline
                  agentTrace={steps}
                  loading
                  query={loadingQuery ?? undefined}
                  plannedTasks={plannedTasks}
                />
              </Paper>
            </MessageBubble>
          </Box>
        ) : null}

        {!running &&
          finalPayload &&
          !finalPayload.results?.length &&
          !finalPayload.sellerSummary?.seller_name &&
          !finalPayload.analysis &&
          !chat.some((entry) => entry.type === "search") ? (
          <Box mt={1.5} mb={3}>
            <MessageBubble role="assistant">
              {finalPayload.finalAnswer || "Ho completato l’analisi della richiesta."}
            </MessageBubble>
          </Box>
        ) : null}

        <div ref={bottomRef} />
      </Box>
    </ChatLayout>
  )
}