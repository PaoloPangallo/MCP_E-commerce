import { useState } from "react"
import { searchProducts } from "../api/searchApi"

import type {
  SearchItem
} from "../component/searchTypes"

export function useSearch() {

  const [results, setResults] = useState<SearchItem[]>([])
  const [analysis, setAnalysis] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function runSearch(query: string) {

    setLoading(true)

    try {

      const data = await searchProducts(query)

      const search = data.final_data?.search

      setResults(search?.results || [])
      setAnalysis(search?.analysis || null)

    } catch (err) {

      console.error("Search error", err)

    } finally {

      setLoading(false)

    }

  }

  return {
    results,
    analysis,
    loading,
    runSearch
  }

}