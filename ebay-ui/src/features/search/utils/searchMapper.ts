import type { SearchBlock } from "../types"
import type { FinalPayload } from "../../agent/types"

export function detectMode(
  resultsCount: number,
  hasSeller: boolean,
  hasComparison: boolean
): "search" | "seller" | "hybrid" | "comparison" {
  if (hasComparison) return "comparison"
  if (resultsCount > 0 && hasSeller) return "hybrid"
  if (hasSeller) return "seller"
  return "search"
}

export function mapPayloadToSearchBlock(query: string, payload: FinalPayload): SearchBlock {
  const sellerSummary = payload.sellerSummary || null
  const results = payload.results || []
  const hasComparison = !!payload.comparison
  const mode = detectMode(results.length, !!sellerSummary?.seller_name, hasComparison)

  return {
    query,
    results,
    analysis: payload.analysis,
    metrics: payload.metrics,
    rag_context: payload.ragContext,
    timings: undefined,
    agent_trace: payload.trace?.length ? payload.trace : [],
    seller_summary: sellerSummary,
    comparison: payload.comparison || null,
    item_details: payload.itemDetails || null,
    shipping_costs: payload.shippingCosts || null,
    metadata: payload.metadata || null,
    market_trends: payload.marketTrends || null,
    deals: payload.deals || null,
    // Senza questo la VisionAnalysisCard esiste solo durante lo stream: il blocco
    // persistito perde l'analisi e a fine risposta la card sparisce dalla chat.
    vision_analysis: payload.visionAnalysis || null,
    final_answer: payload.finalAnswer || "Ho completato l’analisi della richiesta.",
    mode,
    errors: payload.errors
  }
}
