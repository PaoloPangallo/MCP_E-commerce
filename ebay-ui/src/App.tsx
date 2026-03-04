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

export default function App() {

  const [messages, setMessages] = useState<Message[]>(([
    {
      role: "assistant",
      content: "Ciao! Dimmi cosa vuoi cercare su eBay.",
    },
  ]));

  const [results, setResults] = useState<any[]>([]);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, results]);

  const handleSend = async (text: string) => {

    const userMessage: Message = {
      role: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);

    // reset UI
    setResults([]);
    setAnalysis(null);
    setLoading(true);

    try {

      const res = await fetch("http://127.0.0.1:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: text,
          llm_engine: "gemini",
        }),
      });

      if (!res.ok) {
        throw new Error("Backend error");
      }

      const data = await res.json();

      setResults(data.results || []);
      setAnalysis(data.analysis || null);

      const assistantMessage: Message = {
        role: "assistant",
        content: `Ho trovato ${data.results_count || 0} risultati.`,
      };

      setMessages((prev) => [...prev, assistantMessage]);

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
    <ChatLayout>

      {/* Chat area */}
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
        }}
      >

        <Box
          sx={{
            width: "100%",
            maxWidth: 1000,
          }}
        >

          {messages.map((msg, i) => (
            <Box key={i} mb={2}>
              <MessageBubble role={msg.role}>
                {msg.content}
              </MessageBubble>
            </Box>
          ))}

          {loading && (
            <Box display="flex" justifyContent="center" mt={3}>
              <CircularProgress size={28} />
            </Box>
          )}

          {/* AI reasoning */}
          {analysis && (
            <AIAnalysisCard text={analysis} />
          )}

          {/* Search Results */}
          {!loading && results.length > 0 && (
            <Box mt={3}>
              <SearchResultList results={results} />
            </Box>
          )}

          <div ref={bottomRef} />

        </Box>

      </Box>

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={loading} />

    </ChatLayout>
  );
}