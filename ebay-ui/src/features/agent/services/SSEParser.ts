import { mapToolData, buildFinalPayload } from "../hooks/payloadMapper"
import type { AgentEvent, AgentStep, PlannedTask, FinalPayload } from "../types"

export interface AgentState {
  steps: AgentStep[]
  plannedTasks: PlannedTask[]
  running: boolean
  finalPayload: any | null 
  results: any[]
}

export function upsertStep(
  previous: AgentStep[],
  incoming: Partial<AgentStep> & { step: number }
): AgentStep[] {
  const index = previous.findIndex((s) => s.step === incoming.step)

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

export function markStepsAsDone(prev: AgentStep[]): AgentStep[] {
  if (prev.length === 0) return []
  let changed = false
  const next = prev.map(s => {
    if (s.status === "thinking" || s.status === "running") {
      changed = true
      return { ...s, status: "ok" as const, thought: s.thought || "Completato." }
    }
    return s
  })
  return changed ? next : prev
}

function normalizeFinalTrace(trace: AgentStep[] | undefined, fallback: AgentStep[]) {
  if (Array.isArray(trace) && trace.length > 0) return trace
  return fallback
}

/**
 * Pure parsing logic for SSE events.
 * Returns a partial state update.
 */
export function parseSSEEvent(
  event: AgentEvent, 
  currentState: AgentState,
  streamingAnswer: string
): Partial<AgentState> | null {
  
  switch (event.type) {
    case "start":
      return { 
        plannedTasks: Array.isArray(event.planned_tasks) ? event.planned_tasks : [],
        running: true 
      }

    case "thinking":
      return {
        steps: upsertStep(currentState.steps, {
          step: event.step ?? 1,
          thought: event.thought ?? event.message ?? "",
          action: event.action,
          status: "thinking"
        })
      }

    case "tool_start":
      return {
        steps: upsertStep(currentState.steps, {
          step: event.step ?? 1,
          action: event.tool,
          action_input: event.input,
          status: "running"
        })
      }

    case "tool_result": {
      const toolStatus = event.ok ? "ok" : "error"
      const nextSteps = upsertStep(currentState.steps, {
        step: event.step ?? 1,
        action: event.tool!,
        status: toolStatus,
        observation_summary: event.summary
      })

      if (event.ok && (event as any).data) {
        const toolDataPatch = mapToolData(event.tool!, (event as any).data)
        const basePayload = currentState.finalPayload || {
          finalAnswer: null,
          results: currentState.results || [],
          analysis: null,
          trace: nextSteps,
          plannedTasks: currentState.plannedTasks
        }

        const patch: Partial<AgentState> = {
          steps: nextSteps,
          finalPayload: { ...basePayload, ...toolDataPatch }
        }

        // Sync top-level results if it's a search_products tool
        if (event.tool === "search_products" && toolDataPatch.results) {
          patch.results = toolDataPatch.results
        }

        return patch
      }

      return { steps: nextSteps }
    }

    case "answer_chunk": {
      const nextSteps = markStepsAsDone(currentState.steps)
      const basePayload = currentState.finalPayload || {
          finalAnswer: streamingAnswer,
          results: [],
          analysis: null,
          trace: nextSteps,
          plannedTasks: currentState.plannedTasks
      }
      
      const nextPayload = {
        ...basePayload,
        finalAnswer: streamingAnswer,
        trace: markStepsAsDone(basePayload.trace || [])
      }

      return {
        steps: nextSteps,
        finalPayload: nextPayload
      }
    }

    case "vision_analysis":
      return {
        finalPayload: {
          ...currentState.finalPayload,
          visionAnalysis: {
            description: event.description ?? "",
            tags: event.tags ?? [],
            confidence: event.confidence ?? 1.0
          }
        }
      }

    case "final": {
      const fallbackTrace = currentState.steps.map(s =>
        (s.status === "thinking" || s.status === "running") ? { ...s, status: "ok" as const } : s
      )
      const mappedAgentTrace = (event.agent_trace || []).map((s: any) =>
        (s.status === "thinking" || s.status === "running") ? { ...s, status: "ok" as const } : s
      )
      const finalTrace = normalizeFinalTrace(mappedAgentTrace, fallbackTrace)
      const payload = buildFinalPayload(event, streamingAnswer, finalTrace, currentState.plannedTasks)
      
      // Smart merge
      const prev = currentState.finalPayload as FinalPayload | null
      let finalMerged = payload
      if (prev) {
        finalMerged = { ...prev, ...payload }
        for (const key of Object.keys(payload)) {
          const k = key as keyof FinalPayload
          if ((payload[k] === null || payload[k] === undefined) && prev[k] !== undefined && prev[k] !== null) {
            (finalMerged as any)[k] = prev[k]
          } else if (Array.isArray(payload[k]) && (payload[k] as any[]).length === 0 && Array.isArray(prev[k]) && (prev[k] as any[]).length > 0) {
            (finalMerged as any)[k] = prev[k]
          }
        }
      }

      return {
        running: false,
        steps: finalTrace,
        results: finalMerged.results || [],
        finalPayload: finalMerged
      }
    }

    case "error":
      return {
        running: false,
        finalPayload: {
          ...currentState.finalPayload,
          finalAnswer: event.message || "Errore nello stream.",
          errors: [event.message || "Unknown error"]
        }
      }

    case "done":
      return { running: false }

    default:
      return null
  }
}
