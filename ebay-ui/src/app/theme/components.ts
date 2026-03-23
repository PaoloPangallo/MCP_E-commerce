import type {Components, Theme} from "@mui/material/styles"

export const appComponents: Components<Theme> = {
  MuiCssBaseline: {
    styleOverrides: {
      html: {
        height: "100%"
      },
      body: {
        height: "100%",
        margin: 0,
        backgroundColor: "#f7f8fb"
      },
      "#root": {
        minHeight: "100vh"
      },
      "*": {
        boxSizing: "border-box"
      },
      "::-webkit-scrollbar": {
        width: 10,
        height: 10
      },
      "::-webkit-scrollbar-thumb": {
        background: "#d1d5db",
        borderRadius: 999
      },
      "::-webkit-scrollbar-track": {
        background: "transparent"
      }
    }
  },

  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: "none",
        border: "1px solid #e5e7eb"
      }
    }
  },

  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: 999,
        fontWeight: 600
      }
    }
  },

  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        paddingInline: 14
      }
    }
  },

  MuiDrawer: {
    styleOverrides: {
      paper: {
        borderRight: "1px solid #e5e7eb"
      }
    }
  }
}