import { Box, Paper } from "@mui/material";

export default function MessageBubble({
  role,
  children,
}: {
  role: "user" | "assistant";
  children: React.ReactNode;
}) {
  return (
    <Box
      display="flex"
      justifyContent={role === "user" ? "flex-end" : "flex-start"}
    >
      <Paper
        sx={{
          p: 2,
          maxWidth: "70%",
          bgcolor: role === "user" ? "#000" : "#f5f5f5",
          color: role === "user" ? "#fff" : "#000",
          borderRadius: 3,
        }}
      >
        {children}
      </Paper>
    </Box>
  );
}