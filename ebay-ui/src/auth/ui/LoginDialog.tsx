import {
  Dialog,
  DialogTitle,
  DialogContent,
  TextField,
  Button,
  Stack
} from "@mui/material"

import { useState } from "react"
import {useAuth} from "../useAuth.ts";

export default function LoginDialog({
  open,
  onClose,
  onRegister
}: {
  open: boolean
  onClose: () => void
  onRegister: () => void
}) {

  const { login } = useAuth()

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  async function handleLogin() {

    try {

      await login(email, password)

      onClose()

    } catch (err: any) {

      alert(err.message || "Login failed")

    }

  }

  return (

    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xs"
      fullWidth
    >

      <DialogTitle>
        Accedi
      </DialogTitle>

      <DialogContent>

        <Stack spacing={2} mt={1}>

          <TextField
            label="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            fullWidth
          />

          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            fullWidth
          />

          <Button
            variant="contained"
            onClick={handleLogin}
          >
            Login
          </Button>

          <Button
            onClick={onRegister}
          >
            Non hai un account? Registrati
          </Button>

        </Stack>

      </DialogContent>

    </Dialog>

  )

}