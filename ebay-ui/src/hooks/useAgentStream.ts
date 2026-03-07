import { useState } from "react"
import { streamAgent } from "../api/agentStreamApi"

import type { AgentStep, SearchItem } from "../component/searchTypes"

export function useAgentStream() {

  const [steps, setSteps] = useState<AgentStep[]>([])
  const [results, setResults] = useState<SearchItem[]>([])
  const [running, setRunning] = useState(false)

  const run = (query: string) => {

    setSteps([])
    setResults([])
    setRunning(true)

    streamAgent(query, (event) => {

      if (event.type === "thinking") {

  const thinkingStep: AgentStep = {
    step: event.step,
    thought: event.thought,
    action: event.action,
    action_input: event.action_input
  }

  setSteps(prev => [...prev, thinkingStep])
}

if (event.type === "step" && event.step) {

  setSteps(prev => [
    ...prev,
    event.step as AgentStep
  ])

}

      if (event.type === "final") {

        const finalResults =
          event.final_data?.search?.results || []

        setResults(finalResults)

        if (event.agent_trace) {
          setSteps(event.agent_trace)
        }

        setRunning(false)

      }

    })

  }

  return {
    steps,
    results,
    running,
    run
  }

}