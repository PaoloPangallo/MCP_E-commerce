import { create } from "zustand"
import { persist, createJSONStorage } from "zustand/middleware"


import type {
  ChatEntry,
  Message,
  SearchBlock
} from "../../../types/searchTypes.ts"
import {getToken} from "../../../auth/authStore.ts";

export type ChatSession = {
  id: string
  title: string
  chat: ChatEntry[]
  createdAt: number
}

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
  sessions: ChatSession[]
  activeSessionId: string | null
  loadingQuery: string | null
  cache: Record<string, SearchBlock>

  // Actions
  createSession: (firstQuery?: string) => string
  deleteSession: (id: string) => void
  switchSession: (id: string) => void
  resetConversation: () => void // Resets current session

  setLoadingQuery: (query: string | null) => void
  appendMessage: (msg: Message) => void
  appendAssistantMessage: (content: string) => void
  appendSearchBlock: (search: SearchBlock) => void
  setCachedSearch: (key: string, value: SearchBlock) => void
}

const createNewSession = (title: string = "Nuova Ricerca"): ChatSession => ({
  id: crypto.randomUUID(),
  title,
  chat: [getWelcomeMessage()],
  createdAt: Date.now()
})

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      sessions: [createNewSession()],
      activeSessionId: null,
      loadingQuery: null,
      cache: {},

      createSession: (firstQuery) => {
        const newSess = createNewSession(firstQuery ? (firstQuery.length > 25 ? firstQuery.slice(0, 25) + "..." : firstQuery) : undefined)
        set((state) => ({
          sessions: [newSess, ...state.sessions],
          activeSessionId: newSess.id
        }))
        return newSess.id
      },

      deleteSession: (id) => set((state) => {
        const updated = state.sessions.filter(s => s.id !== id)
        let nextId = state.activeSessionId
        let nextSessions = updated
        if (state.activeSessionId === id) {
          if (updated.length > 0) {
            nextId = updated[0].id
          } else {
            const fresh = createNewSession()
            nextSessions = [fresh]
            nextId = fresh.id
          }
        }
        return { sessions: nextSessions, activeSessionId: nextId }
      }),

      switchSession: (id) => set({ activeSessionId: id }),

      resetConversation: () => set((state) => {
        const sid = state.activeSessionId || (state.sessions[0]?.id)
        if (!sid) return state

        return {
          sessions: state.sessions.map(s =>
            s.id === sid ? { ...s, chat: [getWelcomeMessage()], title: "Nuova Ricerca" } : s
          )
        }
      }),

      setLoadingQuery: (query) => set({ loadingQuery: query }),

      appendMessage: (msg) => set((state) => {
        const sid = state.activeSessionId || (state.sessions[0]?.id)
        if (!sid) return state

        return {
          sessions: state.sessions.map(s =>
            s.id === sid ? {
              ...s,
              chat: [...s.chat, { type: "message", msg }],
              title: s.chat.length === 1 && msg.role === "user" ? (msg.content.slice(0, 30)) : s.title
            } : s
          )
        }
      }),

      appendAssistantMessage: (content) => set((state) => {
        const sid = state.activeSessionId || (state.sessions[0]?.id)
        if (!sid) return state

        return {
          sessions: state.sessions.map(s =>
            s.id === sid ? {
              ...s,
              chat: [...s.chat, { type: "message", msg: { role: "assistant", content } }]
            } : s
          )
        }
      }),

      appendSearchBlock: (search) => set((state) => {
        const sid = state.activeSessionId || (state.sessions[0]?.id)
        if (!sid) return state

        return {
          sessions: state.sessions.map(s =>
            s.id === sid ? { ...s, chat: [...s.chat, { type: "search", search }] } : s
          )
        }
      }),

      setCachedSearch: (key, value) =>
        set((state) => ({
          cache: { ...state.cache, [key]: value }
        }))
    }),
    {
      name: "ebay-gpt-sessions",
      storage: createJSONStorage(() => ({
        getItem: (name) => {
          if (!getToken()) return null
          return localStorage.getItem(name)
        },
        setItem: (name, value) => {
          if (getToken()) {
            localStorage.setItem(name, value)
          }
        },
        removeItem: (name) => localStorage.removeItem(name)
      })),
      partialize: (state) => {
        const recentSessions = state.sessions.slice(0, 5)

        const trimmedSessions = recentSessions.map(sess => {
          const trimmedChat = sess.chat.slice(-20).map(entry => {
            if (entry.type === "search" && entry.search) {
              return {
                ...entry,
                search: {
                  ...entry.search,
                  agent_trace: [], // strip trace from disk
                  rag_context: undefined // strip contexts
                }
              }
            }
            return entry
          })

          return { ...sess, chat: trimmedChat }
        })

        return {
          sessions: trimmedSessions,
          activeSessionId: state.activeSessionId
        }
      }
    }
  )
)