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

export default function RegisterDialog({
  open,
  onClose
}: {
  open: boolean
  onClose: () => void
}) {

  const { register } = useAuth()

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")

  async function handleRegister() {

    try {

      await register(email, password)

      onClose()

    } catch (err: any) {

      alert(err.message || "Register failed")

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
        Registrati
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
            onClick={handleRegister}
          >
            Registrati
          </Button>

        </Stack>

      </DialogContent>

    </Dialog>

  )

}