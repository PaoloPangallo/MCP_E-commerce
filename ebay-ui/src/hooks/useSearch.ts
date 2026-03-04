import { useState } from "react";
import {type SearchItem, searchProducts} from "../api/searchApi";

export function useSearch() {
  const [loading, setLoading] = useState(false);

  const search = async (query: string): Promise<SearchItem[]> => {
    setLoading(true);
    try {
      const results = await searchProducts(query);
      return results;
    } finally {
      setLoading(false);
    }
  };

  return { search, loading };
}