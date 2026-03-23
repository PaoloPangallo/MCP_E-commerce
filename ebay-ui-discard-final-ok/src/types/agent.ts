export interface AgentStep {
  step: number
  thought?: string
  action?: string
  action_input?: any
  observation_summary?: string
  status?: "thinking" | "running" | "ok" | "error" | "final"
}