import { useState, useRef, useEffect } from "react"
import { Box } from "@mui/material"

import ChatLayout from "./component/ChatLayout"
import ChatInput from "./component/ChatInput"
import MessageBubble from "./component/MessageBubble"
import SearchResultList from "./component/SearchResultList"
import AIAnalysisCard from "./component/AIAnalysisCard"
import AIThinkingPipeline from "./component/AIThinkingPipeline"


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
  timings?: Record<string, number>
}


// --------------------------------------------------
// APP
// --------------------------------------------------

export default function App() {

  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Ciao! Dimmi cosa vuoi cercare su eBay."
    }
  ])

  const [searches, setSearches] = useState<SearchBlock[]>([])
  const [loading, setLoading] = useState(false)

  const [cache, setCache] = useState<Record<string, SearchBlock>>({})

  const bottomRef = useRef<HTMLDivElement | null>(null)


  // --------------------------------------------------
  // AUTO SCROLL
  // --------------------------------------------------

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, searches, loading])


  // --------------------------------------------------
  // SAVE SEARCH HISTORY
  // --------------------------------------------------

  const saveSearch = (query: string, resultsCount: number = 0) => {

    try {

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

    } catch (err) {

      console.error("History save error:", err)

    }

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

    setMessages(prev => [...prev, userMessage])


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

      setMessages(prev => [...prev, assistantMessage])
      setSearches(prev => [...prev, cached])

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
        throw new Error(`Backend error: ${res.status}`)
      }

      const data = await res.json()

      const newResults: SearchItem[] = data.results || []
      const newAnalysis: string | null = data.analysis || null
      const newMetrics: IRMetrics | undefined = data.metrics || undefined
      const newRagContext: string | undefined = data.rag_context || undefined
      const newTimings: Record<string, number> | undefined = data._timings || undefined


      const newSearch: SearchBlock = {
        query,
        results: newResults,
        analysis: newAnalysis,
        metrics: newMetrics,
        rag_context: newRagContext,
        timings: newTimings
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

      setMessages(prev => [...prev, assistantMessage])
      setSearches(prev => [...prev, newSearch])

    }
    catch (err) {

      console.error("Search error:", err)

      const errorMessage: Message = {
        role: "assistant",
        content: "Errore durante la ricerca. Riprova tra poco."
      }

      setMessages(prev => [...prev, errorMessage])

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
          py: 4
        }}
      >

        <Box sx={{ width: "100%", maxWidth: 1000 }}>


          {/* CHAT MESSAGES */}

          {messages.map((msg, i) => (

            <Box key={i} mb={2}>
              <MessageBubble role={msg.role}>
                {msg.content}
              </MessageBubble>
            </Box>

          ))}


          {/* THINKING PIPELINE */}

          {loading && (

            <AIThinkingPipeline
              loading={true}
            />

          )}


          {/* SEARCH RESULTS */}

          {searches.map((search, i) => (

            <Box key={search.query + i} mt={3}>

              {search.analysis && (

                <>
                  <AIThinkingPipeline
                    loading={false}
                    timings={search.timings}
                  />

                  <AIAnalysisCard
                    text={search.analysis}
                    metrics={search.metrics}
                    rag_context={search.rag_context}
                  />
                </>

              )}

              {search.results.length > 0 && (

                <SearchResultList
                  results={search.results}
                />

              )}

            </Box>

          ))}


          <div ref={bottomRef} />

        </Box>

      </Box>


      <ChatInput
        onSend={handleSend}
        disabled={loading}
      />

    </ChatLayout>

  )

}