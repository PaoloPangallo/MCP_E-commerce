import { Button } from "@mui/material"
import { useAuth } from "./useAuth"

export default function LoginButton() {

  const { loggedIn, login, logout } = useAuth()

  async function handleLogin() {

    const email = prompt("Email")
    const password = prompt("Password")

    if (!email || !password) return

    try {

      await login(email, password)

      alert("Login successful")

    } catch (err: any) {

      alert(err.message || "Login error")

    }

  }

  if (loggedIn) {

    return (

      <Button
        size="small"
        onClick={logout}
      >

        Logout

      </Button>

    )
  }

  return (

    <Button
      size="small"
      onClick={handleLogin}
    >

      Login

    </Button>

  )
}