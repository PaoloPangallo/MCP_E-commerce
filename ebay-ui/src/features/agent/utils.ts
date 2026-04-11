export function humanizeToolName(action?: string, input?: any) {
  const val = (action || "").toLowerCase()
  const q = input?.query || input?.product || ""
  const seller = input?.seller_name || ""

  switch (val) {
    case "search_products":
      return q ? `Cerco "${q}" su eBay` : "Ricerca prodotti"
    case "compare_products":
      return q ? `Confronto "${q}"` : "Confronto prodotti"
    case "profile_query":
      return "Ricerca potenziata da Profilo"
    case "analyze_seller":
      return seller ? `Analizzo venditore ${seller}` : "Analisi venditore"
    case "get_item_details":
      return "Estrazione dettagli oggetto"
    case "get_similar_items":
      return "Ricerca oggetti simili"
    case "get_shipping_costs":
      return "Calcolo spedizione"
    case "get_marketplace_metadata":
      return "Ricerca metadati marketplace"
    case "get_ebay_deals":
      return "Ricerca offerte eBay"
    case "market_trends":
      return "Analisi trend di mercato"
    case "conversation":
      return "Azione conversazionale"
    case "finish":
      return "Sintesi risultati"
    // Legacy fallbacks
    case "search_pipeline":
      return q ? `Cerco "${q}" su eBay` : "Ricerca prodotti"
    case "seller_pipeline":
      return seller ? `Analizzo ${seller}` : "Analisi venditore"
    case "price_history_tool":
      return "Storico prezzi"
    default:
      return action || "Elaborazione"
  }
}
