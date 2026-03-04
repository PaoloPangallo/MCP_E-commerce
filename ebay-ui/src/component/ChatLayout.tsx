import { Box, Drawer, List, ListItem, ListItemButton, ListItemText, Typography, Divider } from "@mui/material";
import AddIcon from '@mui/icons-material/Add';

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
            borderRight: "none",
          },
        }}
      >
        {/* New Chat Button */}
        <Box p={2}>
          <ListItemButton
            sx={{
              borderRadius: 2,
              bgcolor: "transparent",
              "&:hover": {
                bgcolor: "#ececf1",
              },
              py: 1,
              px: 1.5,
            }}
          >
            <Box
              sx={{
                bgcolor: "#fff",
                border: "1px solid #e5e5e5",
                borderRadius: "50%",
                width: 28,
                height: 28,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                mr: 1.5,
              }}
            >
              <AddIcon sx={{ fontSize: 18 }} />
            </Box>
            <ListItemText
              primary="Nuova chat"
              primaryTypographyProps={{
                fontSize: 14,
                fontWeight: 500,
                color: "#0d0d0d",
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
                    py: 1,
                    px: 2,
                    "&:hover": {
                      bgcolor: "#ececf1",
                    },
                  }}
                >
                  <ListItemText
                    primary={`Chat ${i}`}
                    primaryTypographyProps={{
                      fontSize: 14,
                      noWrap: true,
                      sx: { color: "#0d0d0d" },
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
                    py: 1,
                    px: 2,
                    "&:hover": {
                      bgcolor: "#ececf1",
                    },
                  }}
                >
                  <ListItemText
                    primary={`Chat ${i}`}
                    primaryTypographyProps={{
                      fontSize: 14,
                      noWrap: true,
                      sx: { color: "#0d0d0d" },
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
          sx={{
            bgcolor: "transparent",
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
                bgcolor: "#f4f4f4",
                border: "1px solid #e5e5e5",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#0d0d0d",
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
                color: "#0d0d0d",
              }}
            />
          </ListItemButton>
        </Box>
      </Drawer>

      {/* Main Area */}
      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          ml: "260px",
          width: "calc(100% - 260px)",
          height: "100vh",
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
