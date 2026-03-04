import { useState } from "react";
import { searchProducts, type SearchItem } from "../api/searchApi";

interface SearchResponse {
  results: SearchItem[];
  analysis: string | null;
}

export function useSearch() {

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = async (query: string): Promise<SearchResponse> => {

    setLoading(true);
    setError(null);

    try {

      const data = await searchProducts(query);

      return {
        results: data.results ?? [],
        analysis: data.analysis ?? null
      };

    } catch (err) {

      console.error("Search error:", err);

      setError("Errore durante la ricerca");

      return {
        results: [],
        analysis: null
      };

    } finally {

      setLoading(false);

    }
  };

  return {
    search,
    loading,
    error
  };
}