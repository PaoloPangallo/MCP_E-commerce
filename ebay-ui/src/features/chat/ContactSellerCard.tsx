import { Box, Button, Link, Paper, Typography } from "@mui/material"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"

interface ContactSellerResult {
  success?: boolean
  status?: string
  contact_status?: string
  contact_url?: string
  seller_name?: string
  message?: string
  message_sent?: string
  detail?: string
  error?: string
}

// The API contact tool generates a link; the Playwright tool may submit a
// message. Treat these as different outcomes to avoid false "sent" claims.
export default function ContactSellerCard({ data }: { data: ContactSellerResult | null }) {
  if (!data) return null

  const submitted = data.contact_status === "message_sent"
  const failed = data.status === "error" || data.success === false
  const title = submitted
    ? "Messaggio inviato"
    : failed
      ? "Contatto venditore non completato"
      : "Contatta il venditore"

  // Tool output must not become an arbitrary clickable URL in the UI.
  const safeLink =
    typeof data.contact_url === "string" &&
    /^https:\/\/(www\.)?ebay\.it\//i.test(data.contact_url)
      ? data.contact_url
      : null

  const description =
    data.detail ||
    data.error ||
    (submitted
      ? "Il browser ha riportato l'invio del messaggio."
      : "Apri la messaggistica di eBay per continuare.")

  return (
    <Paper
      elevation={0}
      sx={{
        border: "1px solid var(--border-color)",
        borderLeft: "4px solid",
        borderLeftColor: failed ? "error.main" : submitted ? "success.main" : "primary.main",
        bgcolor: "var(--bg-primary)",
        borderRadius: 2,
        p: 2,
      }}
    >
      <Typography fontWeight={800} sx={{ color: "var(--text-primary)" }}>
        {title}
      </Typography>
      {data.seller_name && (
        <Typography sx={{ mt: 0.5, fontSize: 12, color: "var(--text-secondary)" }}>
          Venditore: {data.seller_name}
        </Typography>
      )}
      <Typography sx={{ mt: 1, fontSize: 13, color: "var(--text-primary)", whiteSpace: "pre-wrap" }}>
        {description}
      </Typography>
      {submitted && data.message_sent && (
        <Box sx={{ mt: 1.5, p: 1.5, bgcolor: "var(--bg-secondary)", borderRadius: 1.5 }}>
          <Typography sx={{ fontSize: 12, whiteSpace: "pre-wrap" }}>
            {data.message_sent}
          </Typography>
        </Box>
      )}
      {!submitted && safeLink && (
        <Button
          component={Link}
          href={safeLink}
          target="_blank"
          rel="noopener noreferrer"
          variant="outlined"
          size="small"
          endIcon={<OpenInNewIcon />}
          sx={{ mt: 1.5 }}
        >
          Apri messaggistica eBay
        </Button>
      )}
    </Paper>
  )
}
