import { useMemo } from "react"
import { Box, Chip, Typography } from "@mui/material"

export default function WelcomePanel() {
    const suggestions = useMemo(
        () => [
            "iPhone 13 massimo 700 euro",
            "cerca una maglia Inter e controlla il venditore",
            "analizza il seller pegaso_italia",
            "fammi un confronto tra i migliori risultati per Nintendo Switch"
        ],
        []
    )

    return (
        <Box
            sx={{
                maxWidth: 760,
                mx: "auto",
                px: { xs: 2, md: 3 },
                pt: { xs: 8, md: 12 },
                pb: 6
            }}
        >
            <Typography
                sx={{
                    fontSize: { xs: 30, md: 38 },
                    fontWeight: 700,
                    color: "#111827",
                    letterSpacing: "-0.02em",
                    mb: 1
                }}
            >
                Cosa vuoi cercare oggi?
            </Typography>

            <Typography
                sx={{
                    fontSize: 15,
                    color: "#6b7280",
                    lineHeight: 1.75,
                    maxWidth: 700
                }}
            >
                ebayGPT unisce product search, ranking explanation e seller trust analysis
                in un flusso conversazionale unico.
            </Typography>

            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 3 }}>
                {suggestions.map((item) => (
                    <Chip
                        key={item}
                        label={item}
                        sx={{
                            bgcolor: "#ffffff",
                            border: "1px solid #e5e7eb",
                            fontSize: 13
                        }}
                    />
                ))}
            </Box>
        </Box>
    )
}
