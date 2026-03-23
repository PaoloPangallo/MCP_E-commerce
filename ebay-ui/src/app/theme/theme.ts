import { createTheme } from "@mui/material/styles"

export const appTheme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#111827"
    },
    secondary: {
      main: "#10b981"
    },
    background: {
      default: "#f7f8fb",
      paper: "#ffffff"
    },
    text: {
      primary: "#111827",
      secondary: "#6b7280"
    },
    divider: "#e5e7eb"
  },
  shape: {
    borderRadius: 16
  },
  typography: {
    fontFamily: [
      "Inter",
      "ui-sans-serif",
      "system-ui",
      "-apple-system",
      "BlinkMacSystemFont",
      "\"Segoe UI\"",
      "sans-serif"
    ].join(","),
    h1: {
      fontSize: "2rem",
      fontWeight: 800,
      letterSpacing: "-0.02em"
    },
    h2: {
      fontSize: "1.5rem",
      fontWeight: 700,
      letterSpacing: "-0.02em"
    },
    h3: {
      fontSize: "1.125rem",
      fontWeight: 700
    },
    body1: {
      fontSize: "0.95rem",
      lineHeight: 1.7
    },
    body2: {
      fontSize: "0.875rem",
      lineHeight: 1.6
    },
    button: {
      textTransform: "none",
      fontWeight: 600
    }
  },
  shadows: [
    "none",
    "0 1px 2px rgba(16, 24, 40, 0.05)",
    "0 2px 6px rgba(16, 24, 40, 0.06)",
    "0 8px 20px rgba(16, 24, 40, 0.08)",
    "0 12px 28px rgba(16, 24, 40, 0.10)",
    ...Array(20).fill("0 12px 28px rgba(16, 24, 40, 0.10)")
  ] as any
})