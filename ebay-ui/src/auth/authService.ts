import { apiFetch } from "../api/apiClient"
import { setToken } from "./authStore"
import type { AuthUser } from "./useAuth"

export interface AuthResponse {
  access_token: string
  token_type: string
  user_id: number
}

export async function login(
  email: string,
  password: string
): Promise<AuthResponse> {
  const data = await apiFetch<AuthResponse>("/auth/login", {
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
  const data = await apiFetch<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({
      email,
      password
    })
  })

  setToken(data.access_token)
  return data
}

export function logout() {
  setToken(null)
}

export async function getCurrentUser(): Promise<AuthUser> {
  return apiFetch<AuthUser>("/auth/me", {
    method: "GET"
  })
}

export async function updateCustomInstructions(instructions: string): Promise<{ status: string, custom_instructions: string }> {
  return apiFetch<{ status: string, custom_instructions: string }>("/auth/me/instructions", {
    method: "PATCH",
    body: JSON.stringify({ custom_instructions: instructions })
  })
}