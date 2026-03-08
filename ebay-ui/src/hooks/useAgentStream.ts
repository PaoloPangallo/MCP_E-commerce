import { useCallback, useEffect, useRef, useState } from "react"
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
  const runIdRef = useRef(0)

  const [steps, setSteps] = useState<AgentStep[]>([])
  const [results, setResults] = useState<SearchItem[]>([])
  const [running, setRunning] = useState(false)
  const [finalPayload, setFinalPayload] = useState<FinalPayload | null>(null)

  const reset = useCallback(() => {
    runIdRef.current += 1

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

    reset()
    setRunning(true)

    const currentRunId = runIdRef.current
    let localTrace: AgentStep[] = []

    const nextSource = streamAgent(query, (event) => {
      if (currentRunId !== runIdRef.current) {
        return
      }

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
          trace: localTrace,
          errors: [event.message || "Unknown stream error"]
        })

        if (sourceRef.current) {
          sourceRef.current.close()
          sourceRef.current = null
        }

        return
      }

      const traceStep = normalizeTraceStep(event)

      if (traceStep) {
        localTrace = [...localTrace, traceStep]
        setSteps(localTrace)
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
          trace:
            Array.isArray(event.agent_trace) && event.agent_trace.length > 0
              ? event.agent_trace
              : localTrace,
          errors: Array.isArray(finalData.errors)
            ? finalData.errors
            : []
        })

        setRunning(false)

        if (sourceRef.current) {
          sourceRef.current.close()
          sourceRef.current = null
        }

        return
      }

      if (event.type === "done") {
        setRunning(false)

        if (sourceRef.current) {
          sourceRef.current.close()
          sourceRef.current = null
        }
      }
    })

    sourceRef.current = nextSource
  }, [reset])

  useEffect(() => {
    return () => {
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
    }
  }, [])

  return {
    steps,
    results,
    running,
    finalPayload,
    run,
    reset
  }
}