import { getToken } from "../auth/authStore"

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://127.0.0.1:8050"

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
  
  // Stricter timeouts for better UX
  const isAuthMe = path === "/auth/me"
  const defaultTimeout = isAuthMe ? 8000 : 15000
  const timeout = options.timeout ?? defaultTimeout

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
      if (response.status !== 401 || path !== "/auth/me") {
        console.warn(`API Error [${response.status}] for ${path}`);
      }

      let message = "API error"

      try {
        const data = await response.json()
        message = data?.detail || data?.message || message
      } catch {
        // ignore non-json body
      }

      // Bypass early debugger pause for /auth/me 401 (not logged in)
      if (response.status === 401 && path === "/auth/me") {
        return null as unknown as T
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
      console.error(`Request timeout for ${path} after ${timeout}ms`);
      throw new Error("Il server impiega troppo tempo a rispondere. Verifica la connessione.")
    }

    throw err
  } finally {
    clearTimeout(timeoutId)
  }
}