import { useEffect, useState } from "react"

import {
  getToken,
  subscribe
} from "./authStore"

import {
  login,
  logout,
  register,
  getCurrentUser
} from "./authService"

export interface AuthUser {
  email: string
  favorite_brands?: string | null
  price_preference?: string | null
  custom_instructions?: string | null
}

export function useAuth() {
  const [token, setToken] = useState<string | null>(getToken())
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loadingUser, setLoadingUser] = useState(false)

  useEffect(() => {
    return subscribe(setToken)
  }, [])

  useEffect(() => {
    if (!token) {
      setUser(null)
      setLoadingUser(false)
      return
    }

    async function loadUser() {
      try {
        setLoadingUser(true)
        const data = await getCurrentUser()
        setUser(data)
      } catch {
        setUser(null)
      } finally {
        setLoadingUser(false)
      }
    }

    loadUser()
  }, [token])

  async function handleLogin(email: string, password: string) {
    const data = await login(email, password)
    setToken(data.access_token)
  }

  async function handleRegister(email: string, password: string) {
    const data = await register(email, password)
    setToken(data.access_token)
  }

  function handleLogout() {
    logout()
    setToken(null)
    setUser(null)
  }

  return {
    token,
    loggedIn: !!token,
    user,
    loadingUser,
    login: handleLogin,
    register: handleRegister,
    logout: handleLogout
  }
}