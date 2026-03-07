import { getToken } from "../auth/authStore"

const API_BASE = "http://localhost:8040"

type ApiOptions = RequestInit & {
  timeout?: number
}

export async function apiFetch<T = unknown>(
  path: string,
  options: ApiOptions = {}
): Promise<T> {
  const token = getToken()

  const controller = new AbortController()
  const timeout = options.timeout ?? 90000
  const timeoutId = window.setTimeout(() => controller.abort(), timeout)

  const mergedHeaders: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {})
  }

  if (token) {
    mergedHeaders.Authorization = `Bearer ${token}`
  }

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: mergedHeaders,
      signal: controller.signal
    })

    if (!response.ok) {
      let message = "API error"

      try {
        const data = await response.json()
        message = data.detail || message
      } catch {
        // ignore invalid json body
      }

      throw new Error(message)
    }

    return (await response.json()) as T
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error("Request timeout")
    }

    throw err
  } finally {
    clearTimeout(timeoutId)
  }
}