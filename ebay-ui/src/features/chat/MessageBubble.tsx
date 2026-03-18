import { Box, Typography } from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

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
              letterSpacing: 0.5,
              textAlign: isUser ? "right" : "left"
            }}
          >
            {isUser ? "Tu" : "ebayGPT"}
          </Typography>

          <Box
            sx={{
              fontSize: 17,
              lineHeight: 1.6,
              color: "#374151",
              wordBreak: "break-word",
              textAlign: "left",
              display: "flex",
              flexDirection: "column",
              alignItems: isUser ? "flex-end" : "flex-start"
            }}
          >
            {isUser ? (
              <Box
                sx={{
                  bgcolor: "#f3f4f6",
                  p: 1.5,
                  borderRadius: "20px 20px 4px 20px",
                  color: "#111827",
                  textAlign: "left", // Text wrap starts from left
                  maxWidth: "fit-content",
                  alignSelf: "flex-end"
                }}
              >
                {children}
              </Box>
            ) : isTyping && !children ? (
              <TypingIndicator />
            ) : (
              <Box
                sx={{
                  "& > *:not(:last-child)": { mb: 2 }
                }}
              >
                {typeof children === "string" ? (
                  <Box
                    sx={{
                      fontSize: "1rem",
                      lineHeight: 1.6,
                      "& h1, & h2, & h3": { 
                        display: "block",
                        width: "100%",
                        mt: 4, 
                        mb: 2, 
                        fontWeight: 800, 
                        color: "#111827", 
                        lineHeight: 1.3 
                      },
                      "& h1": { fontSize: "1.6rem" },
                      "& h2": { fontSize: "1.4rem", borderBottom: "1px solid #f0f0f0", pb: 1 },
                      "& h3": { fontSize: "1.2rem" },
                      "& p": { my: 1.25 },
                      "& ul, & ol": { pl: 3, my: 1.25 },
                      "& li": { mb: 0.75 },
                      "& strong": { fontWeight: 700, color: "#000000" },
                      "& code": { bgcolor: "rgba(0,0,0,0.05)", px: 0.6, py: 0.2, borderRadius: 1, fontSize: "0.9em", fontWeight: 600, fontFamily: "'Fira Code', 'Roboto Mono', monospace" },
                      "& pre": { bgcolor: "#1e293b", p: 2, borderRadius: 2, overflowX: "auto", my: 2, "& code": { bgcolor: "transparent", color: "#f8fafc", p: 0, fontSize: "14px", fontWeight: 400 } },
                      "& blockquote": { borderLeft: "4px solid #10a37f", pl: 2.5, py: 0.5, my: 2.5, color: "#4b5563", fontStyle: "italic", bgcolor: "rgba(16, 163, 127, 0.03)" },
                      "& hr": { border: 0, borderTop: "1px solid #e5e7eb", my: 3 },
                      "& table": { width: "100%", borderCollapse: "collapse", my: 2.5, fontSize: "14px", borderRadius: "8px", overflow: "hidden", border: "1px solid #e5e7eb" },
                      "& th, & td": { p: 1.5, textAlign: "left", borderBottom: "1px solid #e5e7eb" },
                      "& th": { bgcolor: "#f8fafb", fontWeight: 700, color: "#475569" },
                      "& a": { color: "#10a37f", textDecoration: "none", borderBottom: "1px dotted #10a37f", "&:hover": { borderBottomStyle: "solid" } },
                    }}
                  >
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        table: ({node, ...props}) => (
                          <Box sx={{ overflowX: "auto", my: 2, borderRadius: "8px", border: "1px solid #e5e7eb", bgcolor: "#fff" }}>
                            <Box component="table" {...props} sx={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", tableLayout: "auto" }} />
                          </Box>
                        ),
                        th: ({node, ...props}) => (
                          <Box component="th" {...props} sx={{ bgcolor: "#f8fafb", p: 1.5, borderBottom: "1px solid #e5e7eb", textAlign: "left", fontWeight: 700, color: "#475569", whiteSpace: "nowrap" }} />
                        ),
                        td: ({node, ...props}) => (
                          <Box component="td" {...props} sx={{ p: 1.5, borderBottom: "1px solid #e5e7eb", textAlign: "left", minWidth: "80px" }} />
                        ),
                        a: ({node, ...props}) => (
                          <a {...props} target="_blank" rel="noopener noreferrer" />
                        )
                      }}
                    >
                      {children as string}
                    </ReactMarkdown>
                  </Box>
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