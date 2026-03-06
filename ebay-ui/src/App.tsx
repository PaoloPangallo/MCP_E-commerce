import { useState, useRef, useEffect } from "react";
import { Box, CircularProgress } from "@mui/material";

import ChatLayout from "./component/ChatLayout";
import ChatInput from "./component/ChatInput";
import MessageBubble from "./component/MessageBubble";
import SearchResultList from "./component/SearchResultList";
import AIAnalysisCard from "./component/AIAnalysisCard";


// --------------------------------------------------
// TYPES
// --------------------------------------------------

interface Message {
  role: "user" | "assistant"
  content: string
}

interface SearchItem {
  ebay_id?: string
  title?: string
  price?: number
  currency?: string
  condition?: string
  seller_name?: string
  seller_rating?: number
  url?: string
  image_url?: string

  trust_score?: number
  ranking_score?: number
  explanations?: string[]

  _already_in_db?: boolean
}

interface IRMetrics {
  "precision@5"?: number
  "precision@10"?: number
  "recall@10"?: number
  "ndcg@10"?: number
}

interface SearchBlock {
  query: string
  results: SearchItem[]
  analysis: string | null
  metrics?: IRMetrics
  rag_context?: string
}


// --------------------------------------------------
// APP
// --------------------------------------------------

type ChatEntry =
  | { type: "message", msg: Message }
  | { type: "search", search: SearchBlock }

export default function App() {

  const [chat, setChat] = useState<ChatEntry[]>([
    { type: "message", msg: { role: "assistant", content: "Ciao! Dimmi cosa vuoi cercare su eBay." } }
  ])

  const [loading, setLoading] = useState(false)

  const [cache, setCache] = useState<Record<string, SearchBlock>>({})

  const bottomRef = useRef<HTMLDivElement | null>(null)


  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chat])


  // --------------------------------------------------
  // SAVE SEARCH HISTORY
  // --------------------------------------------------

  const saveSearch = (query: string, resultsCount: number = 0) => {

    const history = JSON.parse(
      localStorage.getItem("search_history") || "[]"
    )

    const updated = [
      { query, results: resultsCount },
      ...history.filter((h: any) => h.query !== query)
    ]

    const newHistory = updated.slice(0, 20)

    localStorage.setItem(
      "search_history",
      JSON.stringify(newHistory)
    )

    window.dispatchEvent(new Event("search_history_updated"))
  }


  // --------------------------------------------------
  // HANDLE SEARCH
  // --------------------------------------------------

  const handleSend = async (text: string) => {

    if (!text || !text.trim()) return

    const query = text.trim()
    const key = query.toLowerCase()

    const userMessage: Message = {
      role: "user",
      content: query
    }

    setChat(prev => [...prev, { type: "message", msg: userMessage }])


    // --------------------------------------------------
    // CACHE HIT
    // --------------------------------------------------

    if (cache[key]) {

      const cached = cache[key]

      saveSearch(query, cached.results.length)

      const assistantMessage: Message = {
        role: "assistant",
        content: `Ho trovato ${cached.results.length} risultati per "${query}" (cache).`
      }

      setChat(prev => [
        ...prev,
        { type: "message", msg: assistantMessage },
        { type: "search", search: cached }
      ])

      return
    }


    setLoading(true)

    try {

      const res = await fetch("http://127.0.0.1:8030/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query,
          llm_engine: "ollama"
        })
      })

      if (!res.ok) {
        throw new Error("Backend error")
      }

      const data = await res.json()

      const newResults: SearchItem[] = data.results || []
      const newAnalysis: string | null = data.analysis || null

      const newMetrics: IRMetrics | undefined = data.metrics || undefined
      const newRagContext: string | undefined = data.rag_context || undefined


      const newSearch: SearchBlock = {
        query,
        results: newResults,
        analysis: newAnalysis,
        metrics: newMetrics,
        rag_context: newRagContext
      }


      // --------------------------------------------------
      // CACHE
      // --------------------------------------------------

      setCache(prev => ({
        ...prev,
        [key]: newSearch
      }))


      // --------------------------------------------------
      // SAVE HISTORY
      // --------------------------------------------------

      saveSearch(query, newResults.length)


      const assistantMessage: Message = {
        role: "assistant",
        content: `Ho trovato ${data.results_count || newResults.length} risultati per "${query}".`
      }

      setChat(prev => [
        ...prev,
        { type: "message", msg: assistantMessage },
        { type: "search", search: newSearch }
      ])

    }
    catch (err) {

      console.error("Search error:", err)

      const errorMessage: Message = {
        role: "assistant",
        content: "Errore durante la ricerca."
      }

      setChat(prev => [...prev, { type: "message", msg: errorMessage }])

    }
    finally {
      setLoading(false)
    }

  }


  // --------------------------------------------------
  // RENDER
  // --------------------------------------------------

  return (

    <ChatLayout onSearch={handleSend}>

      <Box
        sx={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
          width: "100%",
          px: 4,
          py: 4,
          position: "relative"
        }}
      >

        <Box sx={{ width: "100%", maxWidth: 1000 }}>


          {/* CHAT THREAD */}

          {chat.map((entry, i) => {

            if (entry.type === "message") {
              return (
                <Box key={`msg-${i}`} mb={2}>
                  <MessageBubble role={entry.msg.role}>
                    {entry.msg.content}
                  </MessageBubble>
                </Box>
              )
            }

            if (entry.type === "search") {
              const search = entry.search;
              return (
                <Box key={`search-${i}`} mt={3} mb={4}>

                  {search.analysis && (
                    <AIAnalysisCard
                      text={search.analysis}
                      metrics={search.metrics}
                      rag_context={search.rag_context}
                    />
                  )}

                  {search.results.length > 0 && (
                    <SearchResultList
                      results={search.results}
                    />
                  )}

                </Box>
              )
            }

            return null;
          })}


          <div ref={bottomRef} />

        </Box>


        {loading && (

          <Box
            sx={{
              position: "fixed",
              top: 20,
              right: 20,
              zIndex: 1000
            }}
          >
            <CircularProgress size={24} />
          </Box>

        )}

      </Box>


      <ChatInput
        onSend={handleSend}
        disabled={loading}
      />

    </ChatLayout>

  )

}