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
}

export interface SearchResponse {
  results: SearchItem[]
  analysis?: string
  results_count?: number
}

export async function searchProducts(query: string): Promise<SearchResponse> {

  const response = await fetch("http://localhost:8000/search", {
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
    throw new Error("Errore search API");
  }

  const data = await response.json();

  return data;
}