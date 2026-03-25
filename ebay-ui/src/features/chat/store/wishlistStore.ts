import { create } from "zustand"
import { apiFetch } from "../../../api/apiClient"

export interface WishlistItem {
  id: number
  ebay_id: string
  title: string | null
  price: number | null
  currency: string | null
  condition: string | null
  image_url: string | null
  url: string | null
  seller_name: string | null
  added_at: string | null
}

interface WishlistState {
  items: WishlistItem[]
  isLoading: boolean

  loadItems: () => Promise<void>
  addItem: (item: Omit<WishlistItem, "id" | "added_at">) => Promise<void>
  removeItem: (ebayId: string) => Promise<void>
  isInWishlist: (ebayId: string) => boolean
}

export const useWishlistStore = create<WishlistState>((set, get) => ({
  items: [],
  isLoading: false,

  loadItems: async () => {
    set({ isLoading: true })
    try {
      const resp = await apiFetch<{ count: number; items: WishlistItem[] }>("/wishlist")
      set({ items: resp.items ?? [], isLoading: false })
    } catch {
      set({ isLoading: false })
    }
  },

  addItem: async (item) => {
    try {
      await apiFetch<{ status: string; id: number }>("/wishlist", {
        method: "POST",
        body: JSON.stringify({
          ebay_id: item.ebay_id,
          title: item.title,
          price: item.price,
          currency: item.currency,
          condition: item.condition,
          image_url: item.image_url,
          url: item.url,
          seller_name: item.seller_name,
        }),
      })
      // Refresh to get the full persisted item
      await get().loadItems()
    } catch (err) {
      console.error("Failed to add to wishlist:", err)
    }
  },

  removeItem: async (ebayId) => {
    try {
      await apiFetch<{ status: string }>(`/wishlist/${encodeURIComponent(ebayId)}`, {
        method: "DELETE",
      })
      set((state) => ({ items: state.items.filter((i) => i.ebay_id !== ebayId) }))
    } catch (err) {
      console.error("Failed to remove from wishlist:", err)
    }
  },

  isInWishlist: (ebayId) => {
    return get().items.some((i) => i.ebay_id === ebayId)
  },
}))
