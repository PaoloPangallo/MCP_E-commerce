export async function searchProducts(query: string) {
  const response = await fetch("http://localhost:8000/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      llm_engine: "gemini",
    }),
  });

  if (!response.ok) {
    throw new Error("Errore search API");
  }

  const data = await res.json()
setResults(data.results)
}