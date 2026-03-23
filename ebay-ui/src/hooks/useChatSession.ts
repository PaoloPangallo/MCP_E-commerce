import { useEffect, useMemo } from "react"
import { useShallow } from "zustand/react/shallow"

import type { ChatEntry, Message } from "../features/chat/store/chatStore.ts"

import { useAgentStream } from "../features/agent/hooks/useAgentStream.ts"
import { useChatStore } from "../features/chat/store/chatStore.ts"

export function useChatSession() {
  // Data selectors — useShallow prevents re-renders when unrelated state changes
  const { sessions, activeSessionId, loadingQuery, cache } = useChatStore(
    useShallow((state) => ({
      sessions: state.sessions,
      activeSessionId: state.activeSessionId,
      loadingQuery: state.loadingQuery,
      cache: state.cache
    }))
  )

  // Action selectors — functions are stable references, batch into one selector
  const {
    resetConversation, setLoadingQuery, appendMessage,
    appendAssistantMessage, appendSearchBlock, switchSession, saveAgentResponse
  } = useChatStore(
    useShallow((state) => ({
      resetConversation: state.resetConversation,
      setLoadingQuery: state.setLoadingQuery,
      appendMessage: state.appendMessage,
      appendAssistantMessage: state.appendAssistantMessage,
      appendSearchBlock: state.appendSearchBlock,
      switchSession: state.switchSession,
      saveAgentResponse: state.saveAgentResponse
    }))
  )

  const activeSession = useMemo(() => {
    const sid = activeSessionId || sessions[0]?.id
    return sessions.find(s => s.id === sid) || sessions[0]
  }, [sessions, activeSessionId])

  const chat = activeSession?.chat || []

  const { steps, running, finalPayload, plannedTasks, run, reset } = useAgentStream({
    sessionId: activeSessionId
  })

  // Watch for payload completion
  useEffect(() => {
    if (!finalPayload || running || !loadingQuery) return
    saveAgentResponse(loadingQuery, finalPayload)
  }, [finalPayload, running, loadingQuery, saveAgentResponse])

  // Ensure an active session is set on mount if missing
  useEffect(() => {
    if (!activeSessionId && sessions.length > 0) {
      switchSession(sessions[0].id)
    }
  }, [activeSessionId, sessions, switchSession])

  const hasSearches = useMemo(
    () => chat.some((entry: ChatEntry) => entry.type === "search"),
    [chat]
  )

  const resetChat = () => {
    reset()
    resetConversation()
  }

  const handleSend = async (text: string, image?: string) => {
    if (!text.trim() && !image) return

    const query = text.trim()
    const cacheKey = (query + (image ? "_img" : "")).toLowerCase()

    const userMessage: Message = {
      role: "user",
      content: query || "Analizza questa immagine",
      image: image // Assicurati che il tipo Message supporti image
    }

    appendMessage(userMessage)

    const cached = cache[cacheKey]

    if (cached) {
      appendAssistantMessage(
        cached.final_answer || "Ho recuperato la risposta dalla cache."
      )
      appendSearchBlock(cached)
      return
    }

    setLoadingQuery(query || "Analisi immagine")
    run(query, image)
  }

  return {
    chat,
    activeSessionId: activeSession?.id,
    steps,
    running,
    finalPayload,
    plannedTasks,
    loadingQuery,
    hasSearches,
    handleSend,
    resetChat
  }
}
