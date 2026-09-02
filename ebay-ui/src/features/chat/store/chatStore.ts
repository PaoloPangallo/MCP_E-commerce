import { create } from "zustand"
import { persist, createJSONStorage } from "zustand/middleware"

import type { SearchBlock } from "../../../features/search/types.ts"
import type { AgentMessage, FinalPayload } from "../../agent/types" 
import { getToken } from "../../../auth/authStore.ts";
import { API_BASE } from "../../../api/apiClient.ts"
import { mapPayloadToSearchBlock } from "../../search/utils/searchMapper"

export type Message = AgentMessage

export type ChatEntry =
  | { type: "message"; msg: Message }
  | { type: "search"; search: SearchBlock }

export type ChatSession = {
  id: string
  title: string
  chat: ChatEntry[]
  createdAt: number
}

function getWelcomeMessage(options?: { memoryCleared?: boolean }): ChatEntry {
  const isCleared = options?.memoryCleared
  return {
    type: "message",
    msg: {
      role: "assistant",
      content: isCleared 
        ? "Memoria di sessione ripristinata. Sono pronto per una nuova ricerca partendo da un contesto pulito!" 
        : "Ciao! Sono ebayGPT. Posso cercare prodotti, confrontare risultati, spiegare il ranking e analizzare l’affidabilità di un venditore eBay."
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
  clearMemory: () => Promise<void> // Clears MCP redis memory and resets current session

  setLoadingQuery: (query: string | null) => void
  setSessionTitle: (title: string) => void
  appendMessage: (msg: AgentMessage) => void
  appendAssistantMessage: (content: string) => void
  appendSearchBlock: (search: SearchBlock) => void
  setCachedSearch: (key: string, value: SearchBlock) => void
  saveAgentResponse: (query: string, payload: FinalPayload, extractedTitle?: string, cacheable?: boolean) => void
}

const createNewSession = (title: string = "Nuova chat"): ChatSession => ({
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
            s.id === sid ? { ...s, chat: [getWelcomeMessage()], title: "Nuova chat" } : s
          )
        }
      }),

      clearMemory: async () => {
        try {
          const token = getToken()
          await fetch(`${API_BASE}/agent/memory`, {
            method: "DELETE",
            headers: token ? { "Authorization": `Bearer ${token}` } : {}
          })
        } catch (e) {
          console.error("Failed to clear memory on server", e)
        }
        set((state) => {
          const sid = state.activeSessionId || (state.sessions[0]?.id)
          if (!sid) return state

          return {
            sessions: state.sessions.map(s =>
              s.id === sid ? { ...s, chat: [getWelcomeMessage({ memoryCleared: true })], title: "Nuova chat" } : s
            )
          }
        })
      },

      setLoadingQuery: (query) => set({ loadingQuery: query }),

      // Update the active session title only if it hasn't been set by a real search yet.
      // This is called separately from saveAgentResponse to keep cache keys intact.
      setSessionTitle: (title) => set((state) => {
        const sid = state.activeSessionId || (state.sessions[0]?.id)
        if (!sid) return state
        return {
          sessions: state.sessions.map(s => {
            if (s.id !== sid) return s
            // Only update if no search block exists yet (first search of this session)
            const hasSearch = s.chat.some(e => e.type === "search")
            if (hasSearch) return s
            const trimmed = title.length > 35 ? title.slice(0, 35) + "\u2026" : title
            return { ...s, title: trimmed }
          })
        }
      }),

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
        })),

    saveAgentResponse: (query: string, payload: FinalPayload, extractedTitle?: string, cacheable: boolean = true) => set((state) => {
        const sid = state.activeSessionId || (state.sessions[0]?.id)
        if (!sid) return state

        const cacheKey = query.toLowerCase()
        const newSearch = mapPayloadToSearchBlock(query, payload)

        const hasStructuredBlock =
          newSearch.results.length > 0 ||
          !!newSearch.seller_summary?.seller_name ||
          !!newSearch.analysis ||
          !!newSearch.agent_trace?.length ||
          !!newSearch.errors?.length ||
          !!newSearch.comparison ||
          !!newSearch.item_details ||
          !!newSearch.shipping_costs ||
          !!newSearch.vision_analysis ||
          !!newSearch.market_trends ||
          !!newSearch.deals ||
          !!payload.plannedTasks?.length ||
          (!!payload.toolStates && Object.keys(payload.toolStates).length > 0)

        // I turni con immagine non sono cacheabili: la chiave è puramente testuale
        // e non può rappresentare i byte della foto.
        const nextCache = cacheable ? { ...state.cache, [cacheKey]: newSearch } : state.cache

        return {
          cache: nextCache,
          loadingQuery: null,
          sessions: state.sessions.map(s => {
            if (s.id !== sid) return s

            const nextChat = [...s.chat]

            // NB: la deduplica avviene a monte in useChatSession, confrontando
            // l'identità dell'oggetto finalPayload. Confrontarla qui su
            // (query, analysis) scartava i turni legittimi con immagine, che
            // condividono sempre query "Analisi immagine" e analysis null.

            const hasSearchBefore = nextChat.some(entry => entry.type === "search")

            if (hasStructuredBlock) {
              nextChat.push({ type: "search", search: newSearch })
              if (payload.finalAnswer && payload.finalAnswer !== "Ho completato l’analisi della richiesta.") {
                nextChat.push({ type: "message", msg: { role: "assistant", content: payload.finalAnswer } })
              }
            } else if (payload.finalAnswer) {
              nextChat.push({ type: "message", msg: { role: "assistant", content: payload.finalAnswer } })
            }

            const rawComputedTitle = 
              payload.finalData?.search?.parsed_product || 
              payload.finalData?.search?.query || 
              payload.finalData?.search?.ebay_query_used || 
              extractedTitle;
              
            let finalTitle = s.title;
            if (!hasSearchBefore && rawComputedTitle) {
              // Proper casing for the title
              const words = rawComputedTitle.toLowerCase().split(' ');
              const capitalized = words.map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
              finalTitle = capitalized.length > 35 ? capitalized.slice(0, 35) + "\u2026" : capitalized;
            }

            return { ...s, chat: nextChat, title: finalTitle }
          })
        }
      }),
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