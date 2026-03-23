import { useCallback, useEffect, useRef, useReducer } from "react"
import { streamAgent } from "../api/stream"
import { parseSSEEvent, type AgentState } from "../services/SSEParser"
import type {
  AgentEvent,
  FinalPayload
} from "../types"

// ── RAF throttle helper ─────────────────────────────────────────────
// Coalesces multiple SSE chunks into a single React render per frame.
function createRAFThrottle() {
  let rafId: number | null = null
  let pending: (() => void) | null = null

  function schedule(fn: () => void) {
    pending = fn
    if (rafId === null) {
      rafId = requestAnimationFrame(() => {
        rafId = null
        pending?.()
        pending = null
      })
    }
  }

  function cancel() {
    if (rafId !== null) {
      cancelAnimationFrame(rafId)
      rafId = null
    }
    pending?.()
    pending = null
  }

  return { schedule, cancel }
}

type AgentAction = 
  | { type: 'RESET' }
  | { type: 'UPDATE', payload: Partial<AgentState> }

const initialState: AgentState = {
  steps: [],
  results: [],
  running: false,
  finalPayload: null,
  plannedTasks: []
}

function agentReducer(state: AgentState, action: AgentAction): AgentState {
  switch (action.type) {
    case 'RESET':
      return initialState
    case 'UPDATE':
      return { ...state, ...action.payload }
    default:
      return state
  }
}

export function useAgentStream(options?: {
  sessionId?: string | null
  onDone?: (payload: FinalPayload, originalQuery: string) => void
}) {
  const [state, dispatch] = useReducer(agentReducer, initialState)
  
  // We use a Ref to track the current state because SSE events arrive asynchronously
  // and frequently, and we need the LATEST state for the parser to perform correct upserts.
  const stateRef = useRef(state)
  useEffect(() => {
    stateRef.current = state
  }, [state])

  const sourceRef = useRef<{ close: () => void } | null>(null)
  const runIdRef = useRef(0)
  const sessionIdRef = useRef<string | null>(options?.sessionId || null)
  const answerThrottle = useRef(createRAFThrottle())

  // Cleanup on unmount or session change
  useEffect(() => {
    if (options?.sessionId !== sessionIdRef.current) {
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
      dispatch({ type: 'UPDATE', payload: { running: false } })
      sessionIdRef.current = options?.sessionId || null
    }
  }, [options?.sessionId])

  const reset = useCallback(() => {
    runIdRef.current += 1
    if (sourceRef.current) {
      sourceRef.current.close()
      sourceRef.current = null
    }
    dispatch({ type: 'RESET' })
  }, [])

  const run = useCallback((query: string, image?: string) => {
    if (!query.trim() && !image) return

    reset()
    dispatch({ type: 'UPDATE', payload: { running: true } })

    const currentRunId = runIdRef.current
    let streamingAnswer: string = ""

    const nextSource = streamAgent(query, image, (event: AgentEvent) => {
      // Abort if the run is outdated
      if (currentRunId !== runIdRef.current) return
      if (options?.sessionId && options.sessionId !== sessionIdRef.current) return

      if (event.type === "heartbeat") return

      // Accumulate text stream
      if (event.type === "answer_chunk") {
        streamingAnswer += (typeof event.chunk === "string" ? event.chunk : "")
      }

      // Delegate parsing to pure service
      const patch = parseSSEEvent(event, stateRef.current, streamingAnswer)
      
      if (patch) {
        if (event.type === "answer_chunk") {
          // Throttled update for text chunks to preserve frame rate
          answerThrottle.current.schedule(() => {
             dispatch({ type: 'UPDATE', payload: patch })
          })
        } else {
          // Immediate update for technical tool steps/results
          dispatch({ type: 'UPDATE', payload: patch })
        }
      }

      // Special handling for the onDone callback when final event arrives
      if (event.type === "final") {
        const finalMerged = patch?.finalPayload
        if (finalMerged && options?.onDone) {
           // We use setTimeout to ensure the React render cycle (dispatch) finishes 
           // before the callback (which might trigger external state changes) runs.
           setTimeout(() => {
             options.onDone?.(finalMerged, query)
           }, 0)
        }
      }
    })

    sourceRef.current = nextSource
  }, [reset, options?.sessionId])

  useEffect(() => {
    return () => {
      answerThrottle.current.cancel()
      if (sourceRef.current) {
        sourceRef.current.close()
        sourceRef.current = null
      }
    }
  }, [])

  return {
    ...state,
    run,
    reset
  }
}