import { useEffect, useMemo, useRef } from "react"
import { Box, Chip, Paper, Typography } from "@mui/material"



import { useChatSession } from "../../hooks/useChatSession.ts"
import type { SearchBlock } from "../search/types.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx";
import AIAnalysisCard from "../agent/components/AIAnalysisCard.tsx";
import ChatLayout from "./ChatLayout.tsx";
import SearchResultList from "../search/components/SearchResultList.tsx";
import ChatInput from "./ChatInput.tsx";
import MessageBubble from "./MessageBubble.tsx";
import SellerTrustGauge from "../seller/component/SellerTrustGauge.tsx";
import SellerFeedbackPanel from "../seller/component/SellerFeedbackPanel.tsx";

function SellerSummaryCard({ sellerName, trustScore, sentimentScore, count }: { sellerName?: string; trustScore?: number; sentimentScore?: number; count?: number }) {
  if (!sellerName) return null

  return (
    <Paper elevation={0} sx={{ mt: 2.5, p: 3, borderRadius: 4, bgcolor: "#ffffff", border: "1px solid #e5e7eb" }}>
      <Box display="flex" alignItems="center" justifyContent="space-between" gap={2} flexWrap="wrap" mb={1.5}>
        <Box>
          <Typography sx={{ fontSize: 17, fontWeight: 700, color: "#111827" }}>Seller deep dive</Typography>
          <Typography sx={{ fontSize: 13, color: "#6b7280", mt: 0.5 }}>Analisi affidabilità e segnali sintetici del venditore.</Typography>
        </Box>
        <Chip label={sellerName} size="small" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb" }} />
      </Box>

      {typeof trustScore === "number" ? <SellerTrustGauge score={trustScore} /> : null}

      <Box display="flex" gap={1} flexWrap="wrap" mt={1.5}>
        {typeof sentimentScore === "number" ? <Chip label={`Sentiment ${Math.round(sentimentScore * 100)}%`} size="small" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb" }} /> : null}
        {typeof count === "number" ? <Chip label={`${count} feedback analizzati`} size="small" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb" }} /> : null}
      </Box>

      <SellerFeedbackPanel seller={sellerName} />
    </Paper>
  )
}

function SearchBlockView({ search }: { search: SearchBlock }) {
  const showSellerCard = !!search.seller_summary?.seller_name && (search.mode === "seller" || search.mode === "hybrid")

  return (
    <Box sx={{ mt: 1.5, mb: 4 }}>
      {search.agent_trace?.length ? (
        <Paper elevation={0} sx={{ p: 2.5, borderRadius: 4, border: "1px solid #e5e7eb", bgcolor: "#ffffff" }}>
          <AIThinkingPipeline agentTrace={search.agent_trace} query={search.query} />
        </Paper>
      ) : null}

      {search.analysis || search.metrics || search.rag_context ? (
        <Box mt={2.5}>
          <AIAnalysisCard text={search.analysis ?? undefined} metrics={search.metrics} rag_context={search.rag_context} />
        </Box>
      ) : null}

      {search.mode !== "seller" && search.results.length > 0 ? (
        <Box mt={2.5}>
          <SearchResultList results={search.results} />
        </Box>
      ) : null}

      {showSellerCard ? <SellerSummaryCard sellerName={search.seller_summary?.seller_name} trustScore={search.seller_summary?.trust_score} sentimentScore={search.seller_summary?.sentiment_score} count={search.seller_summary?.count} /> : null}

      {search.errors && search.errors.length > 0 ? (
        <Paper elevation={0} sx={{ mt: 2.5, p: 2.25, borderRadius: 3, bgcolor: "#fff7f7", border: "1px solid #f2d6d6" }}>
          <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#9f2d2d", mb: 0.75 }}>Errori o segnali backend</Typography>
          {search.errors.map((err, idx) => <Typography key={`${err}-${idx}`} sx={{ fontSize: 13, color: "#7a4b4b", lineHeight: 1.6 }}>{err}</Typography>)}
        </Paper>
      ) : null}
    </Box>
  )
}

function WelcomePanel() {
  const suggestions = useMemo(() => [
    "iPhone 13 massimo 700 euro",
    "cerca una maglia Inter e controlla il venditore",
    "analizza il seller pegaso_italia",
    "fammi un confronto tra i migliori risultati per Nintendo Switch"
  ], [])

  return (
    <Box sx={{ maxWidth: 760, mx: "auto", px: { xs: 2, md: 3 }, pt: { xs: 8, md: 12 }, pb: 6 }}>
      <Typography sx={{ fontSize: { xs: 30, md: 38 }, fontWeight: 700, color: "#111827", letterSpacing: "-0.02em", mb: 1 }}>Cosa vuoi cercare oggi?</Typography>
      <Typography sx={{ fontSize: 15, color: "#6b7280", lineHeight: 1.75, maxWidth: 700 }}>
        ebayGPT unisce product search, ranking explanation e seller trust analysis in un flusso conversazionale unico.
      </Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 3 }}>
        {suggestions.map((item) => <Chip key={item} label={item} sx={{ bgcolor: "#ffffff", border: "1px solid #e5e7eb", fontSize: 13 }} />)}
      </Box>
    </Box>
  )
}

export default function ChatPage() {
  const { chat, steps, running, loadingQuery, hasSearches, handleSend, resetChat } = useChatSession()
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [chat, steps, running, loadingQuery])

  return (
    <ChatLayout onSearch={handleSend} onNewChat={resetChat} composer={<ChatInput onSend={handleSend} disabled={running} />}>
      <Box sx={{ width: "100%", maxWidth: 880, mx: "auto", px: { xs: 2, md: 3 }, pt: { xs: 3, md: 4 }, pb: 2 }}>
        {!hasSearches && chat.length <= 1 ? <WelcomePanel /> : null}

        {chat.map((entry, index) => entry.type === "message" ? (
          <Box key={`msg-${index}`} mb={2.5}><MessageBubble role={entry.msg.role}>{entry.msg.content}</MessageBubble></Box>
        ) : (
          <Box key={`search-${index}`}><SearchBlockView search={entry.search} /></Box>
        ))}

        {running ? (
          <Box mt={1.5} mb={3}>
            <MessageBubble role="assistant">
              <Paper elevation={0} sx={{ p: 2.25, borderRadius: 4, border: "1px solid #e5e7eb", bgcolor: "#ffffff" }}>
                <AIThinkingPipeline agentTrace={steps} loading query={loadingQuery ?? undefined} />
              </Paper>
            </MessageBubble>
          </Box>
        ) : null}

        <div ref={bottomRef} />
      </Box>
    </ChatLayout>
  )
}
