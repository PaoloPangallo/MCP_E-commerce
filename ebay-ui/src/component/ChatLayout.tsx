import { Box, Drawer, List, ListItem, ListItemButton, ListItemText, Typography, IconButton, Divider } from "@mui/material";
import AddIcon from '@mui/icons-material/Add';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import MoreVertIcon from '@mui/icons-material/MoreVert';

export default function ChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <Box display="flex" height="100vh" bgcolor="#fff">

      {/* Sidebar */}
      <Drawer
        variant="permanent"
        sx={{
          width: 260,
          "& .MuiDrawer-paper": {
            width: 260,
            boxSizing: "border-box",
            bgcolor: "#f9f9f9",
            borderRight: "1px solid #e5e5e5",
          },
        }}
      >
        {/* New Chat Button */}
        <Box p={2}>
          <ListItemButton
            sx={{
              borderRadius: 2,
              border: "1px solid #d1d1d1",
              bgcolor: "#fff",
              "&:hover": {
                bgcolor: "#f5f5f5",
              },
              py: 1.5,
            }}
          >
            <AddIcon sx={{ mr: 1, fontSize: 20 }} />
            <ListItemText
              primary="Nuova chat"
              primaryTypographyProps={{
                fontSize: 14,
                fontWeight: 500,
              }}
            />
          </ListItemButton>
        </Box>

        {/* Chat History */}
        <Box flex={1} sx={{ overflowY: "auto" }}>
          <List sx={{ px: 1 }}>
            <Typography
              variant="caption"
              sx={{
                px: 2,
                py: 1,
                color: "#6e6e80",
                fontWeight: 600,
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: 0.5,
              }}
            >
              Oggi
            </Typography>

            {[1, 2, 3].map((i) => (
              <ListItem key={i} disablePadding sx={{ mb: 0.5 }}>
                <ListItemButton
                  sx={{
                    borderRadius: 2,
                    py: 1.5,
                    px: 2,
                    "&:hover": {
                      bgcolor: "#ececf1",
                    },
                  }}
                >
                  <ChatBubbleOutlineIcon sx={{ fontSize: 16, mr: 1.5, color: "#6e6e80" }} />
                  <ListItemText
                    primary={`Chat ${i}`}
                    primaryTypographyProps={{
                      fontSize: 14,
                      noWrap: true,
                      sx: { color: "#202123" },
                    }}
                  />
                </ListItemButton>
              </ListItem>
            ))}

            <Divider sx={{ my: 2 }} />

            <Typography
              variant="caption"
              sx={{
                px: 2,
                py: 1,
                color: "#6e6e80",
                fontWeight: 600,
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: 0.5,
              }}
            >
              Ieri
            </Typography>

            {[4, 5].map((i) => (
              <ListItem key={i} disablePadding sx={{ mb: 0.5 }}>
                <ListItemButton
                  sx={{
                    borderRadius: 2,
                    py: 1.5,
                    px: 2,
                    "&:hover": {
                      bgcolor: "#ececf1",
                    },
                  }}
                >
                  <ChatBubbleOutlineIcon sx={{ fontSize: 16, mr: 1.5, color: "#6e6e80" }} />
                  <ListItemText
                    primary={`Chat ${i}`}
                    primaryTypographyProps={{
                      fontSize: 14,
                      noWrap: true,
                      sx: { color: "#202123" },
                    }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>

        {/* User Section */}
        <Box
          p={2}
          borderTop="1px solid #e5e5e5"
          sx={{
            bgcolor: "#f9f9f9",
          }}
        >
          <ListItemButton
            sx={{
              borderRadius: 2,
              py: 1,
              px: 1.5,
              "&:hover": {
                bgcolor: "#ececf1",
              },
            }}
          >
            <Box
              sx={{
                width: 32,
                height: 32,
                borderRadius: "50%",
                bgcolor: "#19c37d",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff",
                fontWeight: 600,
                fontSize: 14,
                mr: 1.5,
              }}
            >
              U
            </Box>
            <ListItemText
              primary="Utente"
              primaryTypographyProps={{
                fontSize: 14,
                fontWeight: 500,
              }}
            />
            <IconButton size="small">
              <MoreVertIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </ListItemButton>
        </Box>
      </Drawer>

      {/* Main Area */}
      <Box flex={1} display="flex" flexDirection="column">
        {children}
      </Box>
    </Box>
  );
}
