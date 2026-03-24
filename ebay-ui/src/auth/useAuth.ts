import { useEffect, useState, useCallback, useRef } from "react"
import {
  getToken,
  subscribe,
  setToken
} from "./authStore"
import {
  login as authServiceLogin,
  logout as authServiceLogout,
  register as authServiceRegister,
  getCurrentUser
} from "./authService"
import { useSettingsStore } from "../features/chat/store/settingsStore"

export interface AuthUser {
  email: string
  favorite_brands?: string | null
  price_preference?: string | null
  custom_instructions?: string | null
  theme?: string | null
  conversation_tone?: string | null
}

export function useAuth() {
  const [token, setTokenState] = useState<string | null>(getToken())
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loadingUser, setLoadingUser] = useState(true) // Start true to check session
  const loadingRef = useRef(false)

  useEffect(() => {
    return subscribe(setTokenState)
  }, [])

  const handleLogout = useCallback(() => {
    authServiceLogout()
    setToken(null)
    setUser(null)
    setLoadingUser(false)
  }, [])

  const loadUser = useCallback(async () => {
    if (!token || loadingRef.current) {
      if (!token) {
        setUser(null)
        setLoadingUser(false)
      }
      return
    }

    loadingRef.current = true
    setLoadingUser(true)
    let isMounted = true
    
    // Safety timeout for loadUser
    const safetyTimeout = setTimeout(() => {
      if (isMounted) setLoadingUser(false);
    }, 8000);

    try {
      const data = await getCurrentUser()
      if (isMounted) {
        if (data) {
          setUser(data)
          // Sincronizza lo store delle impostazioni
          useSettingsStore.getState().loadSettingsFromAuth(
            (data as any).theme,
            (data as any).conversation_tone,
            (data as any).custom_instructions
          )
        } else {
          handleLogout()
        }
      }
    } catch (err) {
      if (isMounted) handleLogout()
    } finally {
      if (isMounted) {
        setLoadingUser(false)
        clearTimeout(safetyTimeout)
        loadingRef.current = false
      } else {
        loadingRef.current = false
      }
    }
  }, [token, handleLogout])

  useEffect(() => {
    loadUser()
  }, [loadUser])

  const login = async (email: string, pass: string) => {
    setLoadingUser(true)
    try {
      const res = await authServiceLogin(email, pass)
      setToken(res.access_token)
      setUser({
        email: res.email,
        favorite_brands: res.favorite_brands,
        price_preference: res.price_preference,
        custom_instructions: res.custom_instructions,
        theme: (res as any).theme,
        conversation_tone: (res as any).conversation_tone
      })

      useSettingsStore.getState().loadSettingsFromAuth(
        (res as any).theme,
        (res as any).conversation_tone,
        (res as any).custom_instructions
      )

      setLoadingUser(false)
    } catch (err: any) {
      setLoadingUser(false)
      throw err
    }
  }

  const register = async (email: string, pass: string) => {
    setLoadingUser(true)
    try {
      const res = await authServiceRegister(email, pass)
      setToken(res.access_token)
      setUser({
        email: res.email,
        favorite_brands: res.favorite_brands,
        price_preference: res.price_preference,
        custom_instructions: res.custom_instructions,
        theme: (res as any).theme,
        conversation_tone: (res as any).conversation_tone
      })

      useSettingsStore.getState().loadSettingsFromAuth(
        (res as any).theme,
        (res as any).conversation_tone,
        (res as any).custom_instructions
      )

      setLoadingUser(false)
    } catch (err: any) {
      setLoadingUser(false)
      throw err
    }
  }

  return {
    token,
    loggedIn: !!token,
    user,
    loadingUser,
    login,
    register,
    logout: handleLogout
  }
}