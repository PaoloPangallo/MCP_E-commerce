import React from "react"
import { Box, Typography } from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

interface MessageBubbleProps {
  role: "user" | "assistant"
  children: React.ReactNode
  timestamp?: string
  isTyping?: boolean
  image?: string
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
  color: "var(--text-primary)",
  "& h1, & h2, & h3": {
    mt: 3,
    mb: 1.25,
    fontWeight: 700,
    color: "var(--text-primary)",
    lineHeight: 1.3
  },
  "& h1": { fontSize: "1.45rem" },
  "& h2": { fontSize: "1.2rem", borderBottom: "1px solid var(--border-color)", pb: 0.75 },
  "& h3": { fontSize: "1.05rem" },
  "& p": { my: 1 },
  "& ul, & ol": { pl: 2.5, my: 1 },
  "& li": { mb: 0.5 },
  "& li + li": { mt: 0.25 },
  "& strong": { fontWeight: 600, color: "var(--text-primary)" },
  "& code": {
    bgcolor: "var(--bg-secondary)",
    px: 0.75,
    py: 0.2,
    borderRadius: 1,
    fontSize: "0.85em",
    fontFamily: "'Fira Code', 'Roboto Mono', monospace"
  },
  "& pre": {
    bgcolor: "#1e293b", // Keep dark for code blocks as standard
    p: 2,
    borderRadius: 2,
    overflowX: "auto",
    my: 2,
    "& code": { bgcolor: "transparent", color: "#f8fafc", p: 0, fontSize: "13px" }
  },
  "& blockquote": {
    borderLeft: "3px solid var(--border-color)",
    pl: 2,
    py: 0.25,
    my: 2,
    color: "var(--text-secondary)",
    fontStyle: "italic"
  },
  "& hr": { border: 0, borderTop: "1px solid var(--border-color)", my: 2.5 },
  "& table": {
    width: "100%",
    borderCollapse: "separate",
    borderSpacing: 0,
    my: 3,
    fontSize: "0.8125rem",
    borderRadius: "12px",
    overflow: "hidden",
    border: "1px solid var(--border-color)",
    boxShadow: "0 1px 3px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.06)",
    bgcolor: "var(--bg-primary)"
  },
  "& th": { 
    bgcolor: "var(--bg-secondary)", 
    fontWeight: 600, 
    color: "var(--text-secondary)", 
    fontSize: "11px",
    textTransform: "uppercase",
    letterSpacing: "0.03em",
    p: 2,
    borderBottom: "1px solid var(--border-color)",
    textAlign: "left"
  },
  "& td": { 
    p: 2, 
    textAlign: "left", 
    borderBottom: "1px solid var(--border-color)",
    color: "var(--text-primary)",
    verticalAlign: "middle"
  },
  "& tr:last-child td": {
    borderBottom: 0
  },
  "& tr:hover td": {
    bgcolor: "var(--bg-secondary)"
  },
  "& a": {
    color: "var(--accent-primary)",
    textDecoration: "none",
    fontWeight: 500,
    "&:hover": { textDecoration: "underline" }
  }
}

export default function MessageBubble({
  role,
  children,
  timestamp,
  isTyping = false,
  image
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
            bgcolor: "var(--accent-primary)",
            color: "var(--bg-primary)",
            px: 2,
            py: 1.25,
            borderRadius: "18px 18px 4px 18px",
            fontSize: "0.9375rem",
            lineHeight: 1.6,
            wordBreak: "break-word",
            boxShadow: "0 2px 10px rgba(0,0,0,0.08)"
          }}
        >
          {image && (
            <Box
              sx={{
                mb: 1.5,
                borderRadius: "12px",
                overflow: "hidden",
                border: "1px solid rgba(255,255,255,0.15)",
                bgcolor: "rgba(255,255,255,0.05)",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                minHeight: 100,
                maxHeight: 320,
                width: "100%"
              }}
            >
              <img
                src={image}
                alt="Uploaded"
                loading="lazy"
                style={{ 
                  maxWidth: "100%", 
                  maxHeight: "320px", 
                  width: "auto",
                  height: "auto",
                  display: "block", 
                  objectFit: "contain" 
                }}
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </Box>
          )}
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
                          my: 3,
                          borderRadius: "12px",
                          width: "100%"
                        }}
                      >
                        <Box
                          component="table"
                          {...props}
                          sx={{
                            width: "100%",
                            borderCollapse: "separate",
                            borderSpacing: 0,
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
                          bgcolor: "var(--bg-secondary)",
                          p: 2,
                          borderBottom: "2px solid var(--border-color)",
                          textAlign: "left",
                          fontWeight: 700,
                          color: "var(--text-secondary)",
                          fontSize: "11px",
                          textTransform: "uppercase",
                          letterSpacing: "0.05em",
                          whiteSpace: "nowrap"
                        }}
                      />
                    ),
                    td: ({ node, children, ...props }) => {
                      const text = String(children || "");
                      const isTrust = text.includes("%") && /\d+/.test(text.trim());
                      const isPrice = text.includes("€") || text.includes("$") || (text.includes(",") && /\d/.test(text));
                      const isCondition = text.toLowerCase().includes("nuovo") || text.toLowerCase().includes("usato");
                      const isLink = React.Children.toArray(children).some(child => 
                        typeof child === 'object' && child !== null && 'type' in child && child.type === 'a'
                      );

                      let renderedContent = children;

                      if (isTrust && !isLink) {
                        const val = parseInt(text.replace(/[^0-9]/g, '')) || 0;
                        const color = val >= 85 ? "var(--success)" : val >= 70 ? "var(--warning)" : "var(--error)";
                        renderedContent = (
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Box sx={{ 
                              px: 1, py: 0.35, borderRadius: '6px', bgcolor: `var(--bg-secondary)`, 
                              color: color, fontWeight: 700, fontSize: '11px', border: `1px solid var(--border-color)`,
                              minWidth: '42px', textAlign: 'center'
                            }}>
                              {text}
                            </Box>
                          </Box>
                        );
                      } else if (isPrice && !isLink) {
                        renderedContent = (
                          <Box sx={{ fontWeight: 700, color: "var(--text-primary)", fontSize: "0.9375rem" }}>
                            {text}
                          </Box>
                        );
                      } else if (isCondition && !isLink) {
                        const isNew = text.toLowerCase().includes("nuovo");
                        const color = isNew ? "var(--success)" : "var(--accent-primary)";
                        renderedContent = (
                          <Box sx={{ 
                            display: 'inline-flex', px: 1, py: 0.25, borderRadius: '4px', 
                            bgcolor: "var(--bg-secondary)", color: color, fontSize: '10px', 
                            fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.02em',
                            border: `1px solid var(--border-color)`
                          }}>
                            {text}
                          </Box>
                        );
                      }

                      return (
                        <Box
                          component="td"
                          {...props}
                          sx={{
                            p: 2,
                            borderBottom: "1px solid var(--border-color)",
                            textAlign: "left",
                            verticalAlign: "middle",
                            color: "var(--text-primary)",
                            "&:first-of-type": { fontWeight: 600, color: "var(--text-primary)", minWidth: "150px" }
                          }}
                        >
                          {renderedContent}
                        </Box>
                      );
                    },
                    a: ({ node, ...props }) => {
                      const isStandaloneLink = props.children === "Vedi" || props.children === "Link";
                      return (
                        <Box 
                          component="a" 
                          {...props} 
                          target="_blank" 
                          rel="noopener noreferrer" 
                          sx={isStandaloneLink ? {
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            bgcolor: 'var(--text-primary)',
                            color: 'var(--bg-primary) !important',
                            px: 2,
                            py: 0.75,
                            borderRadius: '8px',
                            fontSize: '11px',
                            fontWeight: 600,
                            textDecoration: 'none !important',
                            transition: 'all 0.2s',
                            '&:hover': {
                              opacity: 0.9,
                              transform: 'translateY(-1px)',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                            }
                          } : {
                            color: "var(--accent-primary)",
                            textDecoration: "none",
                            "&:hover": { textDecoration: "underline" }
                          }}
                        />
                      );
                    }
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