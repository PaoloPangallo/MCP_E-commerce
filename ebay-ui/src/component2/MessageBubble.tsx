import { Avatar, Box, Typography } from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

interface MessageBubbleProps {
  role: "user" | "assistant"
  children: React.ReactNode
  timestamp?: string
}

export default function MessageBubble({ role, children, timestamp }: MessageBubbleProps) {
  const isUser = role === "user"

  return (
    <Box
      sx={{
        width: "100%",
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        gap: 2,
        px: 2,
        mb: 3
      }}
    >
      {!isUser && (
        <Avatar
          sx={{
            width: 30,
            height: 30,
            bgcolor: "#fff",
            border: "1px solid #e5e5e5",
            color: "#000",
            flexShrink: 0,
            mt: 0.5
          }}
        >
          <AutoAwesomeIcon sx={{ fontSize: 18 }} />
        </Avatar>
      )}

      <Box
        sx={{
          maxWidth: isUser ? "72%" : "calc(100% - 46px)",
          display: "flex",
          flexDirection: "column",
          alignItems: isUser ? "flex-end" : "flex-start"
        }}
      >
        <Box
          sx={{
            px: isUser ? 2.2 : 0,
            py: isUser ? 1.6 : 0.5,
            bgcolor: isUser ? "#f4f4f4" : "transparent",
            borderRadius: isUser ? "24px" : 0,
            maxWidth: "100%"
          }}
        >
          {typeof children === "string" ? (
            <Typography sx={{ fontSize: 15, lineHeight: 1.7, color: "#202123", whiteSpace: "pre-wrap" }}>
              {children}
            </Typography>
          ) : (
            children
          )}
        </Box>

        {timestamp && (
          <Typography variant="caption" sx={{ mt: 0.6, color: "#8e8ea0", fontSize: 11 }}>
            {timestamp}
          </Typography>
        )}
      </Box>
    </Box>
  )
}
