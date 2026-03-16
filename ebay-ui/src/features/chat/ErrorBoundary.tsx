import { Component, type ErrorInfo, type ReactNode } from "react";
import { Box, Typography, Button } from "@mui/material";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";

interface Props {
  children?: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error in UI:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Box
          sx={{
            p: 4,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            textAlign: "center",
            gap: 2,
            bgcolor: "#fff5f5",
            borderRadius: 4,
            border: "1px solid #feb2b2"
          }}
        >
          <ErrorOutlineIcon sx={{ fontSize: 48, color: "#f56565" }} />
          <Typography variant="h6" fontWeight={700} color="#c53030">
            Qualcosa è andato storto
          </Typography>
          <Typography variant="body2" color="#742a2a">
            Si è verificato un errore durante la visualizzazione di questa parte dell'interfaccia.
          </Typography>
          <Button
            variant="outlined"
            size="small"
            onClick={() => this.setState({ hasError: false })}
            sx={{ mt: 1 }}
          >
            Riprova
          </Button>
        </Box>
      );
    }

    return this.props.children;
  }
}
