import { getToken } from "../auth/authStore"

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8050"

type ApiOptions = RequestInit & {
  timeout?: number
}

function buildHeaders(
  token: string | null,
  headers?: HeadersInit
): HeadersInit {
  const merged = new Headers(headers || {})

  if (!merged.has("Content-Type")) {
    merged.set("Content-Type", "application/json")
  }

  if (token) {
    merged.set("Authorization", `Bearer ${token}`)
  }

  return merged
}

export async function apiFetch<T = unknown>(
  path: string,
  options: ApiOptions = {}
): Promise<T> {
  const token = getToken()
  const controller = new AbortController()
  const timeout = options.timeout ?? 90000

  const timeoutId = window.setTimeout(() => {
    controller.abort()
  }, timeout)

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: buildHeaders(token, options.headers),
      signal: controller.signal
    })

    if (!response.ok) {
      let message = "API error"

      try {
        const data = await response.json()
        message = data?.detail || data?.message || message
      } catch {
        // ignore non-json body
      }

      throw new Error(message)
    }

    const contentType = response.headers.get("content-type") || ""

    if (!contentType.includes("application/json")) {
      return undefined as T
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