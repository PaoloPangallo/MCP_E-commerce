export interface AgentStep {
  step: number
  thought?: string
  action?: string
  action_input?: any
  observation_summary?: string
  status?: "thinking" | "running" | "ok" | "error" | "final"
}

export interface AgentEvent {
  type:
    | "start"
    | "thinking"
    | "tool_start"
    | "tool_result"
    | "final"
    | "error"
    | "heartbeat"
    | "done"

  step?: number
  message?: string
  thought?: string
  action?: string
  tool?: string
  input?: Record<string, any>
  ok?: boolean
  summary?: string
  final_answer?: string
  final_data?: any
  agent_trace?: AgentStep[]
  steps_used?: number
}

export type FinalPayload = {
  finalAnswer: string | null
  results: import("../search/types").SearchItem[]
  analysis: string | null
  metrics?: import("../search/types").IRMetrics
  ragContext?: import("../search/types").RagContext
  sellerSummary?: import("../seller/types").SellerSummaryBlock | null
  trace: AgentStep[]
  errors?: string[]
}
