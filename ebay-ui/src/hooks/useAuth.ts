import { useState } from "react"
import {getToken} from "../auth/authStore.ts";
import {login, logout} from "../auth/authService.ts";



export function useAuth() {

  const [tokenState, setTokenState] = useState(
    getToken()
  )

  async function handleLogin(
    email: string,
    password: string
  ) {

    const data = await login(email, password)

    setTokenState(data.access_token)

  }

  function handleLogout() {

    logout()

    setTokenState(null)

  }

  return {

    token: tokenState,

    loggedIn: !!tokenState,

    login: handleLogin,

    logout: handleLogout

  }
}