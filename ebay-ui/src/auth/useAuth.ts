import { useState, useEffect } from "react"

import {
  getToken
} from "./authStore"

import {
  login,
  logout,
  getCurrentUser
} from "./authService"

export function useAuth() {

  const [tokenState, setTokenState] = useState(getToken())

  const [user, setUser] = useState<any>(null)

  useEffect(() => {

    async function loadUser() {

      if (!tokenState) return

      try {

        const data = await getCurrentUser()

        setUser(data)

      } catch {

        setUser(null)

      }

    }

    loadUser()

  }, [tokenState])

  async function handleLogin(email: string, password: string) {

    const data = await login(email, password)

    setTokenState(data.access_token)

  }

  function handleLogout() {

    logout()

    setTokenState(null)

    setUser(null)

  }

  return {

    token: tokenState,

    loggedIn: !!tokenState,

    user,

    login: handleLogin,

    logout: handleLogout

  }

}