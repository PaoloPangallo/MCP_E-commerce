import { useState, useRef, useEffect } from "react";
import { Box, CircularProgress } from "@mui/material";

import ChatLayout from "./component/ChatLayout";
import ChatInput from "./component/ChatInput";
import MessageBubble from "./component/MessageBubble";
import SearchResultList from "./component/SearchResultList";
import AIAnalysisCard from "./component/AIAnalysisCard";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface SearchBlock {
  query: string;
  results: any[];
  analysis: string | null;
}

export default function App() {

  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Ciao! Dimmi cosa vuoi cercare su eBay.",
    },
  ]);

  const [searches, setSearches] = useState<SearchBlock[]>([]);
  const [loading, setLoading] = useState(false);

  const [cache, setCache] = useState<Record<string, any>>({});

  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, searches]);

  // -----------------------------
  // Save search history
  // -----------------------------

  const saveSearch = (query: string, resultsCount: number = 0) => {

    const history = JSON.parse(
      localStorage.getItem("search_history") || "[]"
    );

    const updated = [
      { query, results: resultsCount },
      ...history.filter((h: any) => h.query !== query)
    ];

    const newHistory = updated.slice(0, 20);

    localStorage.setItem(
      "search_history",
      JSON.stringify(newHistory)
    );

    window.dispatchEvent(new Event("search_history_updated"));
  };

  // -----------------------------
  // Handle search
  // -----------------------------

  const handleSend = async (text: string) => {

    if (!text || !text.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);

    // -----------------------------
    // CACHE HIT
    // -----------------------------

    if (cache[text]) {

      const cached = cache[text];

      saveSearch(text, cached.results.length);

      const assistantMessage: Message = {
        role: "assistant",
        content: `Ho trovato ${cached.results.length} risultati per "${text}" (cache).`,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      setSearches((prev) => [
        ...prev,
        {
          query: text,
          results: cached.results,
          analysis: cached.analysis
        }
      ]);

      return;
    }

    setLoading(true);

    try {

      const res = await fetch("http://127.0.0.1:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: text,
          llm_engine: "ollama",
        }),
      });

      if (!res.ok) {
        throw new Error("Backend error");
      }

      const data = await res.json();

      const newResults = data.results || [];
      const newAnalysis = data.analysis || null;

      // save in cache
      setCache((prev) => ({
        ...prev,
        [text]: {
          results: newResults,
          analysis: newAnalysis
        }
      }));

      // update history
      saveSearch(text, newResults.length);

      const assistantMessage: Message = {
        role: "assistant",
        content: `Ho trovato ${data.results_count || newResults.length} risultati per "${text}".`,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      const newSearch: SearchBlock = {
        query: text,
        results: newResults,
        analysis: newAnalysis
      };

      setSearches((prev) => [...prev, newSearch]);

    } catch (err) {

      console.error("Search error:", err);

      const errorMessage: Message = {
        role: "assistant",
        content: "Errore durante la ricerca.",
      };

      setMessages((prev) => [...prev, errorMessage]);

    } finally {
      setLoading(false);
    }
  };

  return (

    <ChatLayout onSearch={handleSend}>

      <Box
        sx={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
          width: "100%",
          px: 4,
          py: 4,
          position: "relative"
        }}
      >

        <Box sx={{ width: "100%", maxWidth: 1000 }}>

          {/* Chat messages */}
          {messages.map((msg, i) => (
            <Box key={i} mb={2}>
              <MessageBubble role={msg.role}>
                {msg.content}
              </MessageBubble>
            </Box>
          ))}

          {/* Search blocks */}
          {searches.map((search, i) => (

            <Box key={i} mt={3}>

              {search.analysis && (
                <AIAnalysisCard text={search.analysis} />
              )}

              {search.results.length > 0 && (
                <SearchResultList results={search.results} />
              )}

            </Box>

          ))}

          <div ref={bottomRef} />

        </Box>

        {loading && (
          <Box
            sx={{
              position: "fixed",
              top: 20,
              right: 20,
              zIndex: 1000
            }}
          >
            <CircularProgress size={24} />
          </Box>
        )}

      </Box>

      <ChatInput onSend={handleSend} disabled={loading} />

    </ChatLayout>
  );
}