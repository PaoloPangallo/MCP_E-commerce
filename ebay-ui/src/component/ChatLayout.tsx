import { useEffect, useState } from "react";
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
  IconButton,
  Divider
} from "@mui/material";

import AddIcon from "@mui/icons-material/Add";
import MoreHorizIcon from "@mui/icons-material/MoreHoriz";
import DeleteIcon from "@mui/icons-material/Delete";
import StarIcon from "@mui/icons-material/Star";

interface HistoryItem {
  query: string;
  results: number;
}

export default function ChatLayout({
  children,
  onSearch
}: {
  children: React.ReactNode;
  onSearch: (q: string) => void;
}) {

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [pinned, setPinned] = useState<string[]>([]);

  // -------------------------
  // LOAD DATA
  // -------------------------

  useEffect(() => {

    const loadHistory = () => {
      const h = JSON.parse(
        localStorage.getItem("search_history") || "[]"
      );

      setHistory(h);
    };

    const loadPinned = () => {
      const p = JSON.parse(
        localStorage.getItem("pinned_searches") || "[]"
      );

      setPinned(p);
    };

    loadHistory();
    loadPinned();

    window.addEventListener(
      "search_history_updated",
      loadHistory
    );

    return () => {
      window.removeEventListener(
        "search_history_updated",
        loadHistory
      );
    };

  }, []);

  // -------------------------
  // CLEAR HISTORY
  // -------------------------

  const clearHistory = () => {

    localStorage.removeItem("search_history");

    setHistory([]);

  };

  // -------------------------
  // PIN SEARCH
  // -------------------------

  const pinSearch = (query: string) => {

    const updated = [query, ...pinned.filter(p => p !== query)];

    const trimmed = updated.slice(0, 10);

    localStorage.setItem(
      "pinned_searches",
      JSON.stringify(trimmed)
    );

    setPinned(trimmed);

  };

  // -------------------------
  // UI
  // -------------------------

  return (
    <Box display="flex" height="100vh" bgcolor="#fff">

      <Drawer
        variant="permanent"
        sx={{
          width: 260,
          "& .MuiDrawer-paper": {
            width: 260,
            bgcolor: "#f9f9f9",
            borderRight: "1px solid #ececec"
          }
        }}
      >

        {/* NEW CHAT */}
        <Box p={2}>
          <ListItemButton
            onClick={() => window.location.reload()}
            sx={{ borderRadius: 2 }}
          >
            <AddIcon sx={{ mr: 1 }} />
            <ListItemText primary="Nuova chat" />
          </ListItemButton>
        </Box>

        <Divider />

        {/* PINNED */}
        {pinned.length > 0 && (

          <>
            <Typography
              variant="caption"
              sx={{
                px: 2,
                pt: 2,
                color: "#8e8ea0",
                fontWeight: 600
              }}
            >
              Salvati
            </Typography>

            <List>

              {pinned.map((q, i) => (

                <ListItem key={i} disablePadding>

                  <ListItemButton
                    onClick={() => onSearch(q)}
                  >

                    <StarIcon
                      sx={{ fontSize: 16, mr: 1 }}
                    />

                    <ListItemText
                      primary={q}
                      primaryTypographyProps={{
                        fontSize: 13
                      }}
                    />

                  </ListItemButton>

                </ListItem>

              ))}

            </List>

            <Divider />

          </>

        )}

        {/* HISTORY */}
        <Box flex={1} overflow="auto">

          <Typography
            variant="caption"
            sx={{
              px: 2,
              pt: 2,
              color: "#8e8ea0",
              fontWeight: 600
            }}
          >
            Oggi
          </Typography>

          <List>

            {history.map((item, i) => (

              <ListItem key={i} disablePadding>

                <ListItemButton
                  onClick={() => onSearch(item.query)}
                >

                  <ListItemText
                    primary={item.query}
                    secondary={`${item.results ?? 0} risultati`}
                    primaryTypographyProps={{
                      fontSize: 13
                    }}
                    secondaryTypographyProps={{
                      fontSize: 11
                    }}
                  />

                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      pinSearch(item.query);
                    }}
                  >
                    <MoreHorizIcon fontSize="small" />
                  </IconButton>

                </ListItemButton>

              </ListItem>

            ))}

          </List>

        </Box>

        {/* CLEAR HISTORY */}
        <Box p={2} borderTop="1px solid #ececec">

          <ListItemButton
            onClick={clearHistory}
            sx={{ borderRadius: 2 }}
          >

            <DeleteIcon sx={{ mr: 1 }} />

            <ListItemText
              primary="Cancella cronologia"
            />

          </ListItemButton>

        </Box>

      </Drawer>

      {/* MAIN */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          height: "100vh"
        }}
      >
        {children}
      </Box>

    </Box>
  );
}