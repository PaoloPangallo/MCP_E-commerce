import { createContext, useContext, useRef, type RefObject } from "react"

const ScrollContainerCtx = createContext<RefObject<HTMLDivElement | null>>({ current: null })

/** Provides the scroll container ref to child components */
export function useScrollContainer() {
  return useContext(ScrollContainerCtx)
}

/** Hook used ONLY by ChatLayout to create and provide the ref */
export function useScrollContainerProvider() {
  const ref = useRef<HTMLDivElement | null>(null)
  return { ref, Provider: ScrollContainerCtx.Provider }
}
