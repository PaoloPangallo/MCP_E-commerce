import { useState } from "react";
import {type Feedback, fetchSellerFeedback} from "../api/sellerApi";

export function useSellerFeedback() {

  const [loading, setLoading] = useState(false);

  const getFeedback = async (seller: string): Promise<Feedback[]> => {
    setLoading(true);
    try {
      const data = await fetchSellerFeedback(seller);
      return data;
    } finally {
      setLoading(false);
    }
  };

  return { getFeedback, loading };
}