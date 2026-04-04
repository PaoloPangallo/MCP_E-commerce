export interface AgentMessage {
  role: "user" | "assistant"
  content: string
  image?: string
}

export interface PlannedTask {
  tool: string
  input: Record<string, any>
}

export interface ToolStatePayload {
  status?: "ok" | "no_data" | "error"
  quality?: "empty" | "partial" | "good"
  terminal?: boolean
  payload?: Record<string, any>
  tool?: string
  attempts?: number
}

export interface AgentStep {
  step: number
  thought?: string
  action?: string
  tool?: string
  action_input?: Record<string, any>
  observation_summary?: string
  observation_data?: any
  status?: "thinking" | "running" | "ok" | "error" | "final"
  ebay_query?: string
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
  | "answer_chunk"
  | "vision_analysis"

  step?: number
  message?: string
  thought?: string
  action?: string
  tool?: string
  input?: Record<string, any>
  ok?: boolean
  summary?: string
  chunk?: string

  description?: string
  tags?: string[]
  brand?: string | null
  condition_clues?: string | null
  confidence?: number

  query?: string
  llm_engine?: string
  max_steps?: number
  planned_tasks?: PlannedTask[]

  final_answer?: string
  final_data?: any
  agent_trace?: AgentStep[]
  steps_used?: number
}

export interface ContactSellerResult {
  status: "ok" | "error"
  success: boolean
  contact_status: string
  detail: string
  message_sent?: string | null
  product_url?: string
}

export type FinalPayload = {
  finalAnswer: string | null
  results: import("../search/types").SearchItem[]
  analysis: string | null
  metrics?: import("../search/types").IRMetrics
  ragContext?: import("../search/types").RagContext
  sellerSummary?: import("../seller/types").SellerSummaryBlock | null
  comparison?: import("../search/types").ComparisonData | null
  itemDetails?: import("../search/types").ItemDetailsData | null
  shippingCosts?: import("../search/types").ShippingCostsData | null
  metadata?: any | null
  deals?: import("../search/types").DealsData | null
  contactSeller?: ContactSellerResult | null
  trace: AgentStep[]
  errors?: string[]

  plannedTasks?: PlannedTask[]
  pendingTasks?: PlannedTask[]
  toolStates?: Record<string, ToolStatePayload>
  toolCalls?: Record<string, number>
  finalData?: Record<string, any> | null
  visionAnalysis?: {
    description: string
    tags: string[]
    brand?: string | null
    condition_clues?: string | null
    confidence: number
  } | null
  marketTrends?: {
    status: string
    query: string
    shopping_data: any
    trends_data: any
  } | null
}