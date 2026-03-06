import { Box, Avatar, Typography } from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

interface MessageBubbleProps {
  role: "user" | "assistant"
  children: React.ReactNode
  timestamp?: string
  isTyping?: boolean
}


// -----------------------------
// Typing indicator
// -----------------------------

function TypingIndicator() {

  return (

    <Box
      display="flex"
      alignItems="center"
      gap={0.8}
      height={24}
    >

      {[0, 0.2, 0.4].map((delay, i) => (

        <Box
          key={i}
          sx={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            bgcolor: "#a3a3a3",
            animation: "bubble 1.4s infinite ease-in-out",
            animationDelay: `${delay}s`,
            "@keyframes bubble": {
              "0%, 80%, 100%": { transform: "scale(0)" },
              "40%": { transform: "scale(1)" }
            }
          }}
        />

      ))}

      <Typography
        sx={{
          fontSize: 12,
          color: "#888",
          ml: 1
        }}
      >
        AI is thinking...
      </Typography>

    </Box>

  )

}


// -----------------------------
// Component
// -----------------------------

export default function MessageBubble({
  role,
  children,
  timestamp,
  isTyping = false
}: MessageBubbleProps) {

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

      {/* ASSISTANT AVATAR */}

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

      {/* MESSAGE CONTENT */}

      <Box
        sx={{
          maxWidth: isUser ? "70%" : "calc(100% - 46px)",
          display: "flex",
          flexDirection: "column",
          alignItems: isUser ? "flex-end" : "flex-start"
        }}
      >

        {/* BUBBLE */}

        <Box
          sx={{
            px: isUser ? 2.2 : 0,
            py: isUser ? 1.6 : 0.5,
            bgcolor: isUser ? "#f4f4f4" : "transparent",
            borderRadius: isUser ? "24px" : 0,
            maxWidth: "100%"
          }}
        >

          {isTyping
            ? <TypingIndicator />
            : children}

        </Box>

        {/* TIMESTAMP */}

        {timestamp && (

          <Typography
            variant="caption"
            sx={{
              mt: 0.6,
              color: "#8e8ea0",
              fontSize: 11
            }}
          >
            {timestamp}
          </Typography>

        )}

      </Box>

    </Box>

  )

}