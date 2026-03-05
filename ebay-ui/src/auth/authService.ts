import { apiFetch } from "../api/apiClient"
import { setToken } from "./authStore"

export async function login(
  email: string,
  password: string
) {

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
) {

  const data = await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify({
      email,
      password
    })
  })

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