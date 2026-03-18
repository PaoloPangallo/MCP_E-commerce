import { Box, Typography } from "@mui/material"

const suggestions = [
  {
    label: "iPhone 13 massimo 700€",
    prompt: "iPhone 13 massimo 700 euro"
  },
  {
    label: "Maglia Inter + verifica venditore",
    prompt: "cerca una maglia Inter e controlla il venditore"
  },
  {
    label: "Analizza seller pegaso_italia",
    prompt: "analizza il seller pegaso_italia"
  },
  {
    label: "Confronto Nintendo Switch",
    prompt: "fammi un confronto tra i migliori risultati per Nintendo Switch"
  }
]

export default function WelcomePanel() {
  const dispatch = (prompt: string) => {
    window.dispatchEvent(new CustomEvent("send-chat", { detail: prompt }))
  }

  return (
    <Box
      sx={{
        pt: { xs: 6, md: 10 },
        pb: 5
      }}
    >
      <Typography
        sx={{
          fontSize: { xs: 26, md: 32 },
          fontWeight: 700,
          color: "#111827",
          letterSpacing: "-0.02em",
          lineHeight: 1.2,
          mb: 1.25
        }}
      >
        Cosa vuoi cercare oggi?
      </Typography>

      <Typography
        sx={{
          fontSize: 15,
          color: "#6b7280",
          lineHeight: 1.7,
          maxWidth: 560,
          mb: 3.5
        }}
      >
        Cerca prodotti, analizza venditori e confronta annunci — tutto in
        linguaggio naturale.
      </Typography>

      {/* Suggestion chips */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
        {suggestions.map((s) => (
          <Box
            key={s.prompt}
            component="button"
            onClick={() => dispatch(s.prompt)}
            sx={{
              display: "inline-flex",
              alignItems: "center",
              px: 1.75,
              py: 0.875,
              border: "1px solid #e5e7eb",
              borderRadius: "20px",
              bgcolor: "#ffffff",
              color: "#374151",
              fontSize: 13,
              fontWeight: 400,
              cursor: "pointer",
              transition: "all 0.15s ease",
              fontFamily: "inherit",
              "&:hover": {
                bgcolor: "#f9fafb",
                borderColor: "#d1d5db",
                color: "#111827"
              },
              "&:active": {
                transform: "scale(0.98)"
              }
            }}
          >
            {s.label}
          </Box>
        ))}
      </Box>
    </Box>
  )
}