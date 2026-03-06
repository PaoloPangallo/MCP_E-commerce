import { Box, Typography, Button, Chip } from "@mui/material"
import { useState } from "react"

import LoginDialog from "./LoginDialog"
import RegisterDialog from "./RegisterDialog"
import {useAuth} from "../useAuth.ts";

export default function AuthPanel() {

  const { user, loggedIn, logout } = useAuth()

  const [loginOpen, setLoginOpen] = useState(false)
  const [registerOpen, setRegisterOpen] = useState(false)

  if (loggedIn && user) {

    return (

      <Box>

        <Typography fontSize={13}>
          {user.email}
        </Typography>

        <Box mt={1} display="flex" gap={1} flexWrap="wrap">

          {user.favorite_brands && (

            <Chip
              size="small"
              label={`Brand: ${user.favorite_brands}`}
            />

          )}

          {user.price_preference && (

            <Chip
              size="small"
              label={`Budget: ${user.price_preference}`}
            />

          )}

        </Box>

        <Button
          size="small"
          sx={{ mt: 1 }}
          onClick={logout}
        >
          Logout
        </Button>

      </Box>

    )

  }

  return (

    <Box>

      <Typography
        fontSize={13}
        color="text.secondary"
        mb={1}
      >
        Accedi per risultati personalizzati
      </Typography>

      <Button
        fullWidth
        variant="contained"
        size="small"
        onClick={() => setLoginOpen(true)}
      >
        Login
      </Button>

      <Button
        fullWidth
        size="small"
        sx={{ mt: 1 }}
        onClick={() => setRegisterOpen(true)}
      >
        Registrati
      </Button>

      <Typography
        fontSize={11}
        color="text.secondary"
        mt={1}
      >
        Puoi usare la ricerca anche senza account.
      </Typography>

      <LoginDialog
        open={loginOpen}
        onClose={() => setLoginOpen(false)}
        onRegister={() => {
          setLoginOpen(false)
          setRegisterOpen(true)
        }}
      />

      <RegisterDialog
        open={registerOpen}
        onClose={() => setRegisterOpen(false)}
      />

    </Box>

  )

}