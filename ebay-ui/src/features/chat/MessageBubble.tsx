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
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start"
      }}
    >
      <Box
        sx={{
          width: "100%",
          maxWidth: isUser ? "78%" : "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: isUser ? "flex-end" : "flex-start"
        }}
      >
        {!isUser ? (
          <Box
            sx={{
              display: "flex",
              alignItems: "flex-start",
              gap: 1.25,
              width: "100%"
            }}
          >
            <Box
              sx={{
                width: 28,
                height: 28,
                mt: 0.15,
                borderRadius: 2,
                bgcolor: "#ffffff",
                border: "1px solid #e5e7eb",
                color: "#111827",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0
              }}
            >
              <AutoAwesomeIcon sx={{ fontSize: 16 }} />
            </Box>

            <Box
              sx={{
                minWidth: 0,
                width: "100%",
                color: "#111827"
              }}
            >
              {isTyping ? (
                <TypingIndicator />
              ) : (
                <Box
                  sx={{
                    fontSize: 15,
                    lineHeight: 1.8,
                    color: "#111827",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word"
                  }}
                >
                  {typeof children === "string" ? (
                    <Typography
                      component="div"
                      sx={{
                        fontSize: 15,
                        lineHeight: 1.8,
                        color: "#111827",
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
        ) : (
          <Box
            sx={{
              px: 2,
              py: 1.4,
              borderRadius: "24px",
              bgcolor: "#eef2f7",
              color: "#111827",
              border: "1px solid #e5e7eb",
              maxWidth: "100%"
            }}
          >
            {typeof children === "string" ? (
              <Typography
                sx={{
                  fontSize: 14.5,
                  lineHeight: 1.7,
                  color: "#111827",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word"
                }}
              >
                {children}
              </Typography>
            ) : (
              children
            )}
          </Box>
        )}

        {timestamp ? (
          <Typography
            sx={{
              mt: 0.7,
              fontSize: 11.5,
              color: "#9ca3af"
            }}
          >
            {timestamp}
          </Typography>
        ) : null}
      </Box>
    </Box>
  )
}