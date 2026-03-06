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


export function useAuth() {

  const [token, setToken] = useState<string | null>(getToken())
  const [user, setUser] = useState<any>(null)


  // sync token
  useEffect(() => {
    return subscribe(setToken)
  }, [])


  // load user
  useEffect(() => {

    if (!token) {
      setUser(null)
      return
    }

    async function loadUser() {

      try {

        const data = await getCurrentUser()
        setUser(data)

      } catch {

        setUser(null)

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

    login: handleLogin,
    register: handleRegister,   // 👈 QUESTA ERA LA COSA MANCANTE
    logout: handleLogout

  }

}