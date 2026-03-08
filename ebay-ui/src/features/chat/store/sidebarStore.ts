import { create } from "zustand"
import { createJSONStorage, persist } from "zustand/middleware"

export type HistoryItem = {
  query: string
  results: number
}

type SidebarStore = {
  mobileOpen: boolean
  history: HistoryItem[]
  pinned: string[]

  setMobileOpen: (open: boolean) => void
  addHistory: (item: HistoryItem) => void
  clearHistory: () => void
  pinSearch: (query: string) => void
  unpinSearch: (query: string) => void
}

function normalizeQuery(query: string) {
  return query.trim()
}

export const useSidebarStore = create<SidebarStore>()(
  persist(
    (set) => ({
      mobileOpen: false,
      history: [],
      pinned: [],

      setMobileOpen: (open) => set({ mobileOpen: open }),

      addHistory: (item) =>
        set((state) => {
          const query = normalizeQuery(item.query)
          if (!query) return state

          const updated = [
            { query, results: item.results },
            ...state.history.filter(
              (entry) => entry.query.toLowerCase() !== query.toLowerCase()
            )
          ].slice(0, 20)

          return { history: updated }
        }),

      clearHistory: () => set({ history: [] }),

      pinSearch: (query) =>
        set((state) => {
          const normalized = normalizeQuery(query)
          if (!normalized) return state

          const updated = [
            normalized,
            ...state.pinned.filter(
              (entry) => entry.toLowerCase() !== normalized.toLowerCase()
            )
          ].slice(0, 10)

          return { pinned: updated }
        }),

      unpinSearch: (query) =>
        set((state) => {
          const normalized = normalizeQuery(query)

          return {
            pinned: state.pinned.filter(
              (entry) => entry.toLowerCase() !== normalized.toLowerCase()
            )
          }
        })
    }),
    {
      name: "ebay-ui-sidebar",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        history: state.history,
        pinned: state.pinned
      })
    }
  )
)