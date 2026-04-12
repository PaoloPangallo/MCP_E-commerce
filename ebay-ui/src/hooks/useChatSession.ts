import { useCallback, useEffect, useMemo } from "react"
import { useShallow } from "zustand/react/shallow"

import type { ChatEntry, Message } from "../features/chat/store/chatStore.ts"

import { useAgentStream } from "../features/agent/hooks/useAgentStream.ts"
import { useChatStore } from "../features/chat/store/chatStore.ts"
import { useSettingsStore } from "../features/chat/store/settingsStore.ts"

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
    appendAssistantMessage, appendSearchBlock, switchSession, saveAgentResponse, setSessionTitle
  } = useChatStore(
    useShallow((state) => ({
      resetConversation: state.resetConversation,
      setLoadingQuery: state.setLoadingQuery,
      appendMessage: state.appendMessage,
      appendAssistantMessage: state.appendAssistantMessage,
      appendSearchBlock: state.appendSearchBlock,
      switchSession: state.switchSession,
      saveAgentResponse: state.saveAgentResponse,
      setSessionTitle: state.setSessionTitle
    }))
  )

  const activeSession = useMemo(() => {
    const sid = activeSessionId || sessions[0]?.id
    return sessions.find(s => s.id === sid) || sessions[0]
  }, [sessions, activeSessionId])

  const chat = activeSession?.chat || []

  const { steps, running, finalPayload, plannedTasks, ebayQuery, run, reset } = useAgentStream({
    sessionId: activeSessionId
  })

  const refreshSettings = useSettingsStore(s => s.refreshSettings)

  // Watch for payload completion
  useEffect(() => {
    if (!finalPayload || running || !loadingQuery) return
    
    // setSessionTitle MUST run before saveAgentResponse:
    // saveAgentResponse adds the search block (hasSearch → true), after which
    // setSessionTitle's guard would reject the update. Order matters here.
    if (ebayQuery) {
      setSessionTitle(ebayQuery)
    }
    saveAgentResponse(loadingQuery, finalPayload)
    
    // Auto-refresh settings if the agent potentially updated them
    const hasProfileUpdate = steps.some((s: any) => 
      s.tool_calls?.some((tc: any) => tc.name === "update_user_preferences")
    )
    if (hasProfileUpdate) {
      refreshSettings()
    }
  }, [finalPayload, running, loadingQuery, saveAgentResponse, setSessionTitle, ebayQuery, steps, refreshSettings])

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

  const handleSend = useCallback(async (text: string, image?: string) => {
    if ((!text.trim() && !image) || running) return

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
  }, [running, appendMessage, appendAssistantMessage, appendSearchBlock, cache, setLoadingQuery, run])

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
