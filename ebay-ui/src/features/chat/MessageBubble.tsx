import { Box, Typography } from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

interface MessageBubbleProps {
  role: "user" | "assistant"
  children: React.ReactNode
  timestamp?: string
  isTyping?: boolean
}

function TypingIndicator() {
  return (
    <Box display="flex" alignItems="center" gap={1} minHeight={26}>
      <Box display="flex" alignItems="center" gap={0.6}>
        {[0, 0.18, 0.36].map((delay, index) => (
          <Box
            key={index}
            sx={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              bgcolor: "#9ca3af",
              animation: "chatgptPulse 1.4s infinite ease-in-out",
              animationDelay: `${delay}s`,
              "@keyframes chatgptPulse": {
                "0%, 80%, 100%": {
                  transform: "scale(0.55)",
                  opacity: 0.45
                },
                "40%": {
                  transform: "scale(1)",
                  opacity: 1
                }
              }
            }}
          />
        ))}
      </Box>

      <Typography
        sx={{
          fontSize: 13,
          color: "#6b7280"
        }}
      >
        L’agente sta ragionando…
      </Typography>
    </Box>
  )
}

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
        mb: 4,
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start"
      }}
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: isUser ? "row-reverse" : "row",
          alignItems: "flex-start",
          gap: 2,
          width: "100%",
          maxWidth: isUser ? "85%" : "100%"
        }}
      >
        {/* AVATAR */}
        <Box
          sx={{
            width: 32,
            height: 32,
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            bgcolor: isUser ? "#111827" : "#10a37f", // ChatGPT Green for assistant
            color: "#ffffff",
            fontSize: 14,
            fontWeight: 600,
            mt: 0.5
          }}
        >
          {isUser ? "U" : <AutoAwesomeIcon sx={{ fontSize: 18 }} />}
        </Box>

        {/* CONTENT */}
        <Box
          sx={{
            minWidth: 0,
            flex: 1,
            pt: 0.5
          }}
        >
          <Typography
            sx={{
              fontSize: 12,
              fontWeight: 700,
              color: "#6b7280",
              mb: 0.5,
              textTransform: "uppercase",
              letterSpacing: 0.5
            }}
          >
            {isUser ? "Tu" : "ebayGPT"}
          </Typography>

          <Box
            sx={{
              fontSize: 16,
              lineHeight: 1.6,
              color: "#374151",
              wordBreak: "break-word"
            }}
          >
            {isTyping && !children ? (
              <TypingIndicator />
            ) : (
              <Box
                sx={{
                  "& p": { my: 0 },
                  "& > *:not(:last-child)": { mb: 2 }
                }}
              >
                {typeof children === "string" ? (
                  <Typography
                    component="div"
                    sx={{
                      fontSize: 16,
                      lineHeight: 1.6,
                      color: "#374151",
                      whiteSpace: "pre-wrap"
                    }}
                  >
                    {children}
                  </Typography>
                ) : (
                  children
                )}
              </Box>
            )}
          </Box>
        </Box>
      </Box>

      {timestamp && (
        <Typography
          sx={{
            fontSize: 11,
            color: "#9ca3af",
            mt: 1,
            ml: isUser ? 0 : 6,
            mr: isUser ? 6 : 0
          }}
        >
          {timestamp}
        </Typography>
      )}
    </Box>
  )
}