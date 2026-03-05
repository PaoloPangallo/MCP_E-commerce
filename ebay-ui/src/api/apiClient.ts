import { getToken } from "../auth/authStore"

const API_BASE = "http://localhost:8030"

export async function apiFetch(
  path: string,
  options: RequestInit = {}
) {

  const token = getToken()

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {})
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers
  })

  if (!res.ok) {

    let message = "API error"

    try {
      const data = await res.json()
      message = data.detail || message
    } catch {}

    throw new Error(message)
  }

  return res.json()
}