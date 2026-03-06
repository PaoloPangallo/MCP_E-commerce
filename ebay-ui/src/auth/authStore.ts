let token: string | null = localStorage.getItem("token")

const listeners: Array<(token: string | null) => void> = []

export function getToken(): string | null {
  return token
}

export function setToken(newToken: string | null) {
  token = newToken

  if (newToken) {
    localStorage.setItem("token", newToken)
  } else {
    localStorage.removeItem("token")
  }

  listeners.forEach(listener => listener(token))
}

export function subscribe(callback: (token: string | null) => void) {
  listeners.push(callback)

  return () => {
    const index = listeners.indexOf(callback)
    if (index >= 0) {
      listeners.splice(index, 1)
    }
  }
}

export function isLoggedIn(): boolean {
  return !!token
}