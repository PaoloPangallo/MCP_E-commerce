import { getToken } from "../auth/authStore"

const API_BASE = "http://localhost:8030"

type ApiOptions = RequestInit & {
  timeout?: number
}

export async function apiFetch<T = any>(
  path: string,
  options: ApiOptions = {}
): Promise<T> {
  const token = getToken()

  const controller = new AbortController()
  const timeout = options.timeout ?? 600000
  const id = setTimeout(() => controller.abort(), timeout)

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {})
  }

  if (token) {
    headers.Authorization = `Bearer ${token} `
  }

  try {
    const res = await fetch(`${API_BASE}${path} `, {
      ...options,
      headers,
      signal: controller.signal
    })

    if (!res.ok) {
      let message = "API error"

      try {
        const data = await res.json()
        message = data.detail || message
      } catch { }

      throw new Error(message)
    }

    return await res.json()
  } catch (err: any) {
    if (err.name === "AbortError") {
      throw new Error("Request timeout")
    }

    throw err
  } finally {
    clearTimeout(id)
  }
}