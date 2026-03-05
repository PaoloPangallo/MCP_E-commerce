let token: string | null = localStorage.getItem("token")

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
}

export function isLoggedIn(): boolean {
  return !!token
}