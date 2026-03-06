import { useState } from "react"
import AIAnalysisCard from "./AIAnalysisCard.tsx";
import SearchResultList from "./SearchResultList.tsx";
import ChatInput from "./ChatInput.tsx";
import AIThinkingPipeline from "./AIThinkingPipeline.tsx";

export default function SearchPage() {

  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any[]>([])
  const [analysis, setAnalysis] = useState("")
  const [timings, setTimings] = useState<any>(null)
  const [metrics, setMetrics] = useState<any>(null)

  const handleSearch = async (query: string) => {

    setLoading(true)
    setResults([])
    setAnalysis("")
    setTimings(null)

    const res = await fetch("http://127.0.0.1:8030/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query })
    })

    const data = await res.json()

    setResults(data.results || [])
    setAnalysis(data.analysis || "")
    setTimings(data._timings || {})
    setMetrics(data.metrics || {})

    setLoading(false)

  }

  return (

    <>
      <AIThinkingPipeline
        loading={loading}
        timings={timings}
      />

      <AIAnalysisCard
        text={analysis}
        metrics={metrics}
      />

      <SearchResultList results={results} />

      <ChatInput
        onSend={handleSearch}
        disabled={loading}
      />
    </>

  )

}