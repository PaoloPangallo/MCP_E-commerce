import { create } from "zustand"

import type {
  ChatEntry,
  Message,
  SearchBlock
} from "../../../types/searchTypes.ts"

function getWelcomeMessage(): ChatEntry {
  return {
    type: "message",
    msg: {
      role: "assistant",
      content:
        "Ciao! Sono ebayGPT. Posso cercare prodotti, confrontare risultati, spiegare il ranking e analizzare l’affidabilità di un venditore eBay."
    }
  }
}

type ChatStore = {
  chat: ChatEntry[]
  loadingQuery: string | null
  cache: Record<string, SearchBlock>

  resetConversation: () => void
  setLoadingQuery: (query: string | null) => void

  appendMessage: (msg: Message) => void
  appendAssistantMessage: (content: string) => void
  appendSearchBlock: (search: SearchBlock) => void

  setCachedSearch: (key: string, value: SearchBlock) => void
}

export const useChatStore = create<ChatStore>((set) => ({
  chat: [getWelcomeMessage()],
  loadingQuery: null,
  cache: {},

  resetConversation: () =>
    set((state) => ({
      chat: [getWelcomeMessage()],
      loadingQuery: null,
      cache: state.cache
    })),

  setLoadingQuery: (query) => set({ loadingQuery: query }),

  appendMessage: (msg) =>
    set((state) => ({
      chat: [...state.chat, { type: "message", msg }]
    })),

  appendAssistantMessage: (content) =>
    set((state) => ({
      chat: [
        ...state.chat,
        {
          type: "message",
          msg: {
            role: "assistant",
            content
          }
        }
      ]
    })),

  appendSearchBlock: (search) =>
    set((state) => ({
      chat: [...state.chat, { type: "search", search }]
    })),

  setCachedSearch: (key, value) =>
    set((state) => ({
      cache: {
        ...state.cache,
        [key]: value
      }
    }))
}))