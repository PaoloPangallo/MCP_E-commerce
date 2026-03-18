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
    <Box display="flex" alignItems="center" gap={0.75} sx={{ py: 0.5 }}>
      {[0, 0.18, 0.36].map((delay, index) => (
        <Box
          key={index}
          sx={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            bgcolor: "#9ca3af",
            animation: "chatPulse 1.4s infinite ease-in-out",
            animationDelay: `${delay}s`,
            "@keyframes chatPulse": {
              "0%, 80%, 100%": { transform: "scale(0.55)", opacity: 0.4 },
              "40%": { transform: "scale(1)", opacity: 1 }
            }
          }}
        />
      ))}
    </Box>
  )
}

const markdownSx = {
  fontSize: "0.9375rem",
  lineHeight: 1.7,
  color: "#111827",
  "& h1, & h2, & h3": {
    mt: 3,
    mb: 1.25,
    fontWeight: 700,
    color: "#111827",
    lineHeight: 1.3
  },
  "& h1": { fontSize: "1.45rem" },
  "& h2": { fontSize: "1.2rem", borderBottom: "1px solid #f0f0f0", pb: 0.75 },
  "& h3": { fontSize: "1.05rem" },
  "& p": { my: 1 },
  "& ul, & ol": { pl: 2.5, my: 1 },
  "& li": { mb: 0.5 },
  "& li + li": { mt: 0.25 },
  "& strong": { fontWeight: 600, color: "#111827" },
  "& code": {
    bgcolor: "rgba(0,0,0,0.06)",
    px: 0.75,
    py: 0.2,
    borderRadius: 1,
    fontSize: "0.85em",
    fontFamily: "'Fira Code', 'Roboto Mono', monospace"
  },
  "& pre": {
    bgcolor: "#1e293b",
    p: 2,
    borderRadius: 2,
    overflowX: "auto",
    my: 2,
    "& code": { bgcolor: "transparent", color: "#f8fafc", p: 0, fontSize: "13px" }
  },
  "& blockquote": {
    borderLeft: "3px solid #e5e7eb",
    pl: 2,
    py: 0.25,
    my: 2,
    color: "#6b7280",
    fontStyle: "italic"
  },
  "& hr": { border: 0, borderTop: "1px solid #f0f0f0", my: 2.5 },
  "& table": {
    width: "100%",
    borderCollapse: "collapse",
    my: 2,
    fontSize: "13px",
    borderRadius: "8px",
    overflow: "hidden",
    border: "1px solid #e5e7eb"
  },
  "& th, & td": { p: 1.25, textAlign: "left", borderBottom: "1px solid #e5e7eb" },
  "& th": { bgcolor: "#f8fafb", fontWeight: 600, color: "#475569", fontSize: "12px" },
  "& a": {
    color: "#111827",
    textDecoration: "underline",
    textDecorationColor: "#d1d5db",
    textUnderlineOffset: "3px",
    "&:hover": { textDecorationColor: "#111827" }
  }
}

export default function MessageBubble({
  role,
  children,
  timestamp,
  isTyping = false
}: MessageBubbleProps) {
  const isUser = role === "user"

  if (isUser) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "flex-end",
          mb: 1.5,
          px: { xs: 0, md: 0 }
        }}
      >
        <Box
          sx={{
            maxWidth: { xs: "88%", md: "72%" },
            bgcolor: "#111827",
            color: "#f9fafb",
            px: 2,
            py: 1.25,
            borderRadius: "18px 18px 4px 18px",
            fontSize: "0.9375rem",
            lineHeight: 1.6,
            wordBreak: "break-word"
          }}
        >
          {children}
        </Box>
      </Box>
    )
  }

  // Assistant
  return (
    <Box
      sx={{
        display: "flex",
        gap: 1.5,
        alignItems: "flex-start",
        mb: 2.5,
        maxWidth: "100%"
      }}
    >
      {/* Avatar */}
      <Box
        sx={{
          width: 28,
          height: 28,
          borderRadius: "50%",
          bgcolor: "#111827",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          mt: 0.25
        }}
      >
        <AutoAwesomeIcon sx={{ fontSize: 14, color: "#fff" }} />
      </Box>

      {/* Content */}
      <Box sx={{ flex: 1, minWidth: 0, pt: 0.25 }}>
        {isTyping && !children ? (
          <TypingIndicator />
        ) : (
          <>
            {typeof children === "string" ? (
              <Box sx={markdownSx}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({ node, ...props }) => (
                      <Box
                        sx={{
                          overflowX: "auto",
                          my: 2,
                          borderRadius: "8px",
                          border: "1px solid #e5e7eb",
                          bgcolor: "#fff"
                        }}
                      >
                        <Box
                          component="table"
                          {...props}
                          sx={{
                            width: "100%",
                            borderCollapse: "collapse",
                            fontSize: "0.8125rem",
                            tableLayout: "auto"
                          }}
                        />
                      </Box>
                    ),
                    th: ({ node, ...props }) => (
                      <Box
                        component="th"
                        {...props}
                        sx={{
                          bgcolor: "#f8fafb",
                          p: 1.25,
                          borderBottom: "1px solid #e5e7eb",
                          textAlign: "left",
                          fontWeight: 600,
                          color: "#475569",
                          whiteSpace: "nowrap",
                          fontSize: "12px"
                        }}
                      />
                    ),
                    td: ({ node, ...props }) => (
                      <Box
                        component="td"
                        {...props}
                        sx={{
                          p: 1.25,
                          borderBottom: "1px solid #e5e7eb",
                          textAlign: "left",
                          minWidth: "80px"
                        }}
                      />
                    ),
                    a: ({ node, ...props }) => (
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

            {timestamp && (
              <Typography
                sx={{ fontSize: 11, color: "#9ca3af", mt: 0.75 }}
              >
                {timestamp}
              </Typography>
            )}
          </>
        )}
      </Box>
    </Box>
  )
}