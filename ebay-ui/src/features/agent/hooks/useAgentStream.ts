import { useCallback, useEffect, useRef, useState } from "react"
import { streamAgent } from "../api/stream"

import type {
  AgentEvent,
  AgentStep,
  FinalPayload,
  PlannedTask,
  ToolStatePayload
} from "../types"

function upsertStep(
  previous: AgentStep[],
  incoming: Partial<AgentStep> & { step: number }
): AgentStep[] {
  const index = previous.findIndex((s) => {
    if (s.step !== incoming.step) return false
    if (incoming.action && s.action && s.action !== incoming.action) return false

    if (incoming.status === "running") {
      return s.status === "thinking" && !s.action_input
    }

    if (incoming.status === "ok" || incoming.status === "error") {
      return s.status === "running" || s.status === "thinking"
    }

    return true
  })

  if (index === -1) {
    return [
      ...previous,
      {
        step: incoming.step,
        thought: incoming.thought ?? "",
        action: incoming.action,
        action_input: incoming.action_input,
        observation_summary: incoming.observation_summary,
        status: incoming.status ?? "thinking"
      }
    ]
  }

  const current = previous[index]

  const updated: AgentStep = {
    ...current,
    ...incoming,
    thought: incoming.thought ?? current.thought,
    action: incoming.action ?? current.action,
    action_input: incoming.action_input ?? current.action_input,
    observation_summary:
      incoming.observation_summary ?? current.observation_summary,
    status: incoming.status ?? current.status
  }

  const next = [...previous]
  next[index] = updated
  return next
}

function normalizeFinalTrace(trace: AgentStep[] | undefined, fallback: AgentStep[]) {
  if (Array.isArray(trace) && trace.length > 0) return trace
  return fallback
}

export function useAgentStream(options?: {
  onDone?: (payload: FinalPayload, originalQuery: string) => void
}) {
  const sourceRef = useRef<EventSource | null>(null)
  const runIdRef = useRef(0)

  const [steps, setSteps] = useState<AgentStep[]>([])
  const [results, setResults] = useState<import("../../search/types").SearchItem[]>([])
  const [running, setRunning] = useState(false)
  const [finalPayload, setFinalPayload] = useState<FinalPayload | null>(null)
  const [plannedTasks, setPlannedTasks] = useState<PlannedTask[]>([])

  const reset = useCallback(() => {
    runIdRef.current += 1

    if (sourceRef.current) {
      sourceRef.current.close()
      sourceRef.current = null
    }

    setSteps([])
    setResults([])
    setFinalPayload(null)
    setPlannedTasks([])
    setRunning(false)
  }, [])

  const run = useCallback((query: string) => {
    if (!query.trim()) return

    reset()
    setRunning(true)

    const currentRunId = runIdRef.current
    let localTrace: AgentStep[] = []
    let localPlannedTasks: PlannedTask[] = []

    const nextSource = streamAgent(query, (event: AgentEvent) => {
      if (currentRunId !== runIdRef.current) return

      if (event.type === "heartbeat") return

      if (event.type === "start") {
        localPlannedTasks = Array.isArray(event.planned_tasks)
          ? event.planned_tasks
          : []
        setPlannedTasks(localPlannedTasks)
        return
      }

      if (event.type === "error") {
        setRunning(false)
        const errorPayload: FinalPayload = {
          finalAnswer:
            event.message || "Si è verificato un errore nello stream agentico.",
          results: [],
          analysis: null,
          sellerSummary: null,
          trace: localTrace,
          errors: [event.message || "Unknown stream error"],
          plannedTasks: localPlannedTasks,
          pendingTasks: [],
          toolStates: {},
          toolCalls: {},
          finalData: null
        }
        setFinalPayload(errorPayload)
        options?.onDone?.(errorPayload, query)

        if (sourceRef.current) {
          sourceRef.current.close()
          sourceRef.current = null
        }
        return
      }

      if (event.type === "thinking") {
        const nextStep = {
          step: event.step ?? 1,
          thought: event.thought ?? event.message ?? "",
          action: event.action,
          status: "thinking" as const
        }

        localTrace = upsertStep(localTrace, nextStep)
        setSteps(localTrace)
        return
      }

      if (event.type === "tool_start") {
        const nextStep = {
          step: event.step ?? 1,
          action: event.tool,
          action_input: event.input,
          status: "running" as const
        }

        localTrace = upsertStep(localTrace, nextStep)
        setSteps(localTrace)
        return
      }

      if (event.type === "tool_result") {
        const nextStep = {
          step: event.step ?? 1,
          action: event.tool,
          observation_summary: event.summary,
          status: event.ok ? ("ok" as const) : ("error" as const)
        }

        localTrace = upsertStep(localTrace, nextStep)
        setSteps(localTrace)
        return
      }

      if (event.type === "final") {
        const finalData = event.final_data || {}
        const search = finalData.search || {}
        const seller = finalData.seller || null
        const finalResults = Array.isArray(search.results) ? search.results : []
        const finalTrace = normalizeFinalTrace(event.agent_trace, localTrace)

        const toolStates = finalData.tool_states || {}
        const toolCalls = finalData.tool_calls || {}
        const pendingTasks = Array.isArray(finalData.pending_tasks)
          ? finalData.pending_tasks
          : []

        const payload: FinalPayload = {
          finalAnswer: event.final_answer || null,
          results: finalResults,
          analysis: search.analysis || finalData.search_analysis || null,
          metrics: search.metrics || finalData.metrics,
          ragContext: search.rag_context,
          sellerSummary: seller,
          comparison: finalData.compare || null,
          itemDetails: finalData.item_details || null,
          shippingCosts: finalData.shipping_costs || null,
          trace: finalTrace,
          errors: Array.isArray(finalData.errors) ? finalData.errors : [],
          plannedTasks: localPlannedTasks,
          pendingTasks,
          toolStates: toolStates as Record<string, ToolStatePayload>,
          toolCalls,
          finalData
        }

        setResults(finalResults)
        setFinalPayload(payload)
        options?.onDone?.(payload, query)

        setSteps(finalTrace)
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
    plannedTasks,
    run,
    reset
  }
}