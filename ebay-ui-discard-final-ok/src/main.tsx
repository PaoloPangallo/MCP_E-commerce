import React from "react"
import ReactDOM from "react-dom/client"
import { CssBaseline, ThemeProvider } from "@mui/material"
import App from "./App"
import { appTheme, appComponents } from "./app/theme"
import "./index.css"

const theme = {
  ...appTheme,
  components: {
    ...appTheme.components,
    ...appComponents
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
)