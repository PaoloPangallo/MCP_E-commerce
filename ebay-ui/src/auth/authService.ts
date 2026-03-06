import { apiFetch } from "../api/apiClient"
import { setToken } from "./authStore"

export interface AuthResponse {

  access_token: string
  token_type: string
  user_id: number

}


export async function login(
  email: string,
  password: string
): Promise<AuthResponse> {

  const data = await apiFetch("/auth/login", {

    method: "POST",

    body: JSON.stringify({
      email,
      password
    })

  })

  setToken(data.access_token)

  return data

}


export async function register(
  email: string,
  password: string
): Promise<AuthResponse> {

  const data = await apiFetch("/auth/register", {

    method: "POST",

    body: JSON.stringify({
      email,
      password
    })

  })

  // auto login
  setToken(data.access_token)

  return data

}


export function logout() {

  setToken(null)

}


export async function getCurrentUser() {

  return apiFetch("/auth/me", {
    method: "GET"
  })

}