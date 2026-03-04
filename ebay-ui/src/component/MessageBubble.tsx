import { Box, Avatar, Typography } from "@mui/material";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

interface MessageBubbleProps {
  role: "user" | "assistant";
  children: React.ReactNode;
  timestamp?: string;
  isTyping?: boolean;
}

export default function MessageBubble({
  role,
  children,
  timestamp,
  isTyping = false,
}: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <Box
      sx={{
        width: "100%",
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        gap: 2,
        px: 2,
        mb: 4,
      }}
    >
      {/* Assistant Avatar */}
      {!isUser && (
        <Avatar
          variant="circular"
          sx={{
            width: 30,
            height: 30,
            bgcolor: "#fff",
            border: "1px solid #d1d1d1",
            color: "#000",
            flexShrink: 0,
          }}
        >
          <AutoAwesomeIcon sx={{ fontSize: 18 }} />
        </Avatar>
      )}

      {/* Message Content */}
      <Box
        sx={{
          maxWidth: isUser ? "70%" : "calc(100% - 46px)", // account for avatar width and gap
          display: "flex",
          flexDirection: "column",
          alignItems: isUser ? "flex-end" : "flex-start",
        }}
      >
        {/* Message Bubble */}
        <Box
          sx={{
            p: isUser ? 2 : 0,
            py: isUser ? 1.5 : 0.5,
            bgcolor: isUser ? "#f4f4f4" : "transparent",
            color: "#0d0d0d",
            borderRadius: isUser ? "24px" : 0,
            position: "relative",
          }}
        >
          {/* Typing Indicator */}
          {isTyping ? (
            <Box display="flex" gap={0.75} alignItems="center" height={24}>
              <Box
                sx={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  bgcolor: "#a3a3a3",
                  animation: "bounce 1.4s infinite ease-in-out",
                  animationDelay: "0s",
                  "@keyframes bounce": {
                    "0%, 80%, 100%": { transform: "scale(0)" },
                    "40%": { transform: "scale(1)" },
                  },
                }}
              />
              <Box
                sx={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  bgcolor: "#a3a3a3",
                  animation: "bounce 1.4s infinite ease-in-out",
                  animationDelay: "0.2s",
                }}
              />
              <Box
                sx={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  bgcolor: "#a3a3a3",
                  animation: "bounce 1.4s infinite ease-in-out",
                  animationDelay: "0.4s",
                }}
              />
            </Box>
          ) : (
            <Box
              sx={{
                fontSize: 16,
                lineHeight: 1.6,
                "& p": { m: 0, mb: 1.5, "&:last-child": { mb: 0 } },
                "& pre": {
                  bgcolor: "#f4f4f4",
                  p: 2,
                  borderRadius: 2,
                  overflow: "auto",
                  fontSize: 14,
                  my: 1.5,
                },
                "& code": {
                  bgcolor: isUser ? "rgba(0,0,0,0.05)" : "#f4f4f4",
                  px: 0.75,
                  py: 0.25,
                  borderRadius: 1,
                  fontSize: 14,
                  fontFamily: "monospace",
                },
              }}
            >
              {children}
            </Box>
          )}
        </Box>

        {/* Timestamp */}
        {timestamp && (
          <Typography
            variant="caption"
            sx={{
              mt: 0.5,
              color: "#8e8ea0",
              fontSize: 11,
            }}
          >
            {timestamp}
          </Typography>
        )}
      </Box>
    </Box>
  );
}