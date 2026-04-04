import type { FinalPayload, AgentStep, PlannedTask, ToolStatePayload } from "../types"

// ── Tool-to-Payload Mapping ──────────────────────────────────────────
// Single source of truth for mapping tool results to FinalPayload fields.
// Used by both the `tool_result` (incremental) and `final` (complete) handlers.

interface ToolDataPatch {
  results?: any[]
  analysis?: string | null
  metrics?: any
  ragContext?: any
  sellerSummary?: any
  comparison?: any
  itemDetails?: any
  shippingCosts?: any
  marketTrends?: any
  metadata?: any
  deals?: any
  contactSeller?: any
  visionAnalysis?: { description: string; tags: string[]; brand?: string | null; condition_clues?: string | null; confidence: number }
}

/** Maps a tool name + its data to the corresponding FinalPayload fields */
export function mapToolData(toolName: string, data: any): ToolDataPatch {
  switch (toolName) {
    case "get_ebay_deals":
      if (data && Array.isArray(data.deals)) return { deals: data }
      if (data && Array.isArray(data.items)) return { deals: { ...data, deals: data.items } }
      return { deals: data }
    case "search_products":
      return {
        results: data.results || [],
        analysis: data.analysis || null,
        metrics: data.metrics || data.ir_metrics,
        ragContext: data.rag_context
      }
    case "analyze_seller":
      return { sellerSummary: data }
    case "compare_products":
      return { comparison: data }
    case "get_item_details":
      return { itemDetails: (data && data.data) ? data.data : data }
    case "get_shipping_costs":
      return { shippingCosts: (data && data.data) ? data.data : data }
    case "market_trends":
      return { marketTrends: data }
    case "get_marketplace_metadata":
      return { metadata: data }
    case "contact_seller":
    case "contact_seller_playwright":
      return { contactSeller: data }
    default:
      return {}
  }
}

/** Builds a complete FinalPayload from the `final` SSE event */
export function buildFinalPayload(
  event: any,
  streamingAnswer: string,
  finalTrace: AgentStep[],
  plannedTasks: PlannedTask[]
): FinalPayload {
  const finalData = event.final_data || {}
  const search = finalData.search || {}
  const seller = finalData.seller || null
  const finalResults = Array.isArray(search.results) ? search.results : []

  const toolStates = finalData.tool_states || {}
  const toolCalls = finalData.tool_calls || {}
  const pendingTasks = Array.isArray(finalData.pending_tasks)
    ? finalData.pending_tasks
    : []

  return {
    finalAnswer: event.final_answer || streamingAnswer || null,
    results: finalResults,
    analysis: search.analysis || finalData.search_analysis || null,
    metrics: search.metrics || finalData.metrics,
    ragContext: search.rag_context,
    sellerSummary: seller,
    comparison: finalData.compare || null,
    itemDetails: finalData.item_details || null,
    shippingCosts: finalData.shipping_costs || finalData.shippingCosts || null,
    metadata: finalData.metadata || null,
    marketTrends: finalData.market_trends || finalData.marketTrends || null,
    deals: finalData.deals || null,
    contactSeller: finalData.contact_seller_playwright || finalData.contact_seller || null,
    trace: finalTrace,
    errors: Array.isArray(finalData.errors) ? finalData.errors : [],
    plannedTasks,
    pendingTasks,
    toolStates: toolStates as Record<string, ToolStatePayload>,
    toolCalls,
    finalData
  }
}
