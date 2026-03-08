import { useCallback, useRef, useState } from "react"
import { streamAgent, type AgentEvent } from "../api/agentStreamApi"

import type {
  AgentStep,
  SearchItem,
  SellerSummaryBlock,
  IRMetrics,
  RagContext
} from "../component/searchTypes"


type FinalPayload = {
  finalAnswer: string | null
  results: SearchItem[]
  analysis: string | null
  metrics?: IRMetrics
  ragContext?: RagContext
  sellerSummary?: SellerSummaryBlock | null
  trace: AgentStep[]
  errors?: string[]
}


function normalizeTraceStep(event: AgentEvent): AgentStep | null {

  if (event.type === "thinking") {

    return {
      step: event.step ?? 1,
      thought: event.thought ?? event.message ?? "",
      action: event.action,
      status: "thinking"
    }

  }

  if (event.type === "tool_start") {

    return {
      step: event.step ?? 1,
      thought: `Avvio tool ${event.tool ?? ""}`,
      action: event.tool,
      action_input: event.input,
      status: "running"
    }

  }

  if (event.type === "tool_result") {

    return {
      step: event.step ?? 1,
      action: event.tool,
      observation_summary: event.summary,
      status: event.ok ? "ok" : "error"
    }

  }

  return null
}



export function useAgentStream() {

  const sourceRef = useRef<EventSource | null>(null)

  const [steps, setSteps] = useState<AgentStep[]>([])
  const [results, setResults] = useState<SearchItem[]>([])
  const [running, setRunning] = useState(false)
  const [finalPayload, setFinalPayload] = useState<FinalPayload | null>(null)


  const reset = useCallback(() => {

    if (sourceRef.current) {
      sourceRef.current.close()
      sourceRef.current = null
    }

    setSteps([])
    setResults([])
    setFinalPayload(null)
    setRunning(false)

  }, [])



  const run = useCallback((query: string) => {

    if (!query.trim()) return


    // reset stato
    reset()

    setRunning(true)


    const nextSource = streamAgent(query, (event) => {

      // ignoriamo eventi non utili
      if (event.type === "heartbeat" || event.type === "start") {
        return
      }


      if (event.type === "error") {

        setRunning(false)

        setFinalPayload({
          finalAnswer:
            event.message ||
            "Si è verificato un errore nello stream agentico.",
          results: [],
          analysis: null,
          sellerSummary: null,
          trace: [],
          errors: [event.message || "Unknown stream error"]
        })

        return
      }


      const traceStep = normalizeTraceStep(event)

      if (traceStep) {

        setSteps((prev) => [...prev, traceStep])

      }


      if (event.type === "final") {

        const finalData = event.final_data || {}
        const search = finalData.search || {}
        const seller = finalData.seller || null

        const finalResults = Array.isArray(search.results)
          ? search.results
          : []

        setResults(finalResults)

        setFinalPayload({
          finalAnswer: event.final_answer || null,
          results: finalResults,
          analysis: search.analysis || finalData.search_analysis || null,
          metrics: search.metrics || finalData.metrics,
          ragContext: search.rag_context,
          sellerSummary: seller,
          trace: Array.isArray(event.agent_trace)
            ? event.agent_trace
            : [],
          errors: Array.isArray(finalData.errors)
            ? finalData.errors
            : []
        })

        setRunning(false)

      }


      if (event.type === "done") {

        setRunning(false)

      }

    })


    sourceRef.current = nextSource

  }, [reset])


  return {
    steps,
    results,
    running,
    finalPayload,
    run,
    reset
  }

}