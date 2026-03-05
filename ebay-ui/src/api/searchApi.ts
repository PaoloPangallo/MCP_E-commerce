export interface SearchItem {
  ebay_id: string
  title: string
  price: number
  currency: string
  condition: string
  seller_name: string
  seller_rating: number
  url: string
  image_url: string

  trust_score?: number
  ranking_score?: number
  explanations?: string[]

  _already_in_db?: boolean
}


export interface IRMetrics {
  "precision@5": number
  "precision@10": number
  "recall@10": number
  "ndcg@10": number
}

export interface SearchResponse {

  parsed_query?: any

  ebay_query_used?: string

  results: SearchItem[]

  results_count?: number

  saved_new_count?: number

  analysis?: string

  rag_context?: string

  metrics?: IRMetrics

  _timings?: Record<string, number>
}

export async function searchProducts(query: string): Promise<SearchResponse> {

  const response = await fetch("http://localhost:8020/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      llm_engine: "ollama",
    }),
  });

  if (!response.ok) {

    const text = await response.text()

    throw new Error(`Search API error: ${text}`)
  }

  return await response.json()
}