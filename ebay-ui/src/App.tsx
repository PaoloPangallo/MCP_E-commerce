import { useState } from "react";
import { Box, CircularProgress } from "@mui/material";

import ChatLayout from "./component/ChatLayout";
import ChatInput from "./component/ChatInput";
import MessageBubble from "./component/MessageBubble";
import SearchResultList from "./component/SearchResultList";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function App() {

const [messages, setMessages] = useState<Message[]>([
  {
    role: "assistant",
    content: "Ciao! Dimmi cosa vuoi cercare su eBay.",
  },
]);

  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (text: string) => {

    const userMessage: Message = {
      role: "user",
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {

      const res = await fetch("http://localhost:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: text,
          llm_engine: "gemini",
        }),
      });

      const data = await res.json();

      setResults(data.results || []);

      const assistantMessage: Message = {
        role: "assistant",
        content: `Ho trovato ${data.results_count} risultati.`,
      };

      setMessages((prev) => [...prev, assistantMessage]);

    } catch (err) {

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

      {/* Chat messages */}
      <Box
        sx={{
          flex: 1,
          overflowY: "auto",
          p: 3,
          maxWidth: 900,
          width: "100%",
          mx: "auto",
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

        {/* Search Results */}
        {!loading && results.length > 0 && (
          <Box mt={3}>
            <SearchResultList results={results} />
          </Box>
        )}

      </Box>

      {/* Input */}
      <ChatInput onSend={handleSend} disabled={loading} />

    </ChatLayout>
  );
}