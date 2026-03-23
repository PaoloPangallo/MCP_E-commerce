import type { SellerFeedbackResponse } from "../types"
import {apiFetch} from "../../../api/apiClient.ts";

export async function fetchSellerFeedback(
  seller: string,
  page = 1,
  limit = 5
): Promise<SellerFeedbackResponse> {
  const safeSeller = encodeURIComponent(seller)
  return apiFetch<SellerFeedbackResponse>(`/seller/${safeSeller}/feedback?page=${page}&limit=${limit}`)
}
