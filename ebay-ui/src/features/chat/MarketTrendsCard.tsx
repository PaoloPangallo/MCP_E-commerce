import {
  Box, Typography, Card, CardContent, Grid, Stack,
  LinearProgress, Tooltip, Chip, Divider
} from "@mui/material"
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip as ChartTooltip, ResponsiveContainer, ReferenceLine
} from "recharts"
import TrendingUpIcon     from "@mui/icons-material/TrendingUp"
import TrendingDownIcon   from "@mui/icons-material/TrendingDown"
import TrendingFlatIcon   from "@mui/icons-material/TrendingFlat"
import WhatshotIcon       from "@mui/icons-material/Whatshot"
import ArrowDownwardIcon  from "@mui/icons-material/ArrowDownward"
import ShoppingCartIcon   from "@mui/icons-material/ShoppingCart"
import InfoOutlinedIcon   from "@mui/icons-material/InfoOutlined"
import AssessmentOutlinedIcon from "@mui/icons-material/AssessmentOutlined"
import AutoAwesomeIcon    from "@mui/icons-material/AutoAwesome"
import SearchIcon         from "@mui/icons-material/Search"
import LocalOfferIcon     from "@mui/icons-material/LocalOffer"
import StorefrontIcon     from "@mui/icons-material/Storefront"
import OpenInNewIcon      from "@mui/icons-material/OpenInNew"
import LeaderboardIcon    from "@mui/icons-material/Leaderboard"
import type {JSX} from "react"

// ─── Types ────────────────────────────────────────────────────────────────────

interface SellerItem {
  seller: string
  price: number
  title: string
  link?: string
  thumbnail?: string
}

interface ShoppingResult {
  title: string
  source: string
  price: string
  extracted_price?: number
  thumbnail?: string
  link?: string
}

interface MarketTrendsCardProps {
  data: {
    status: string
    query: string
    shopping_data: {
      status: string
      min_price?: number
      max_price?: number
      average_price?: number
      median_price?: number
      std_dev?: number
      price_range?: number
      price_consistency?: string
      samples?: number
      message?: string
      top_results?: ShoppingResult[]
      seller_breakdown?: SellerItem[]
    }
    trends_data: {
      status: string
      current_interest?: number
      trend_direction?: string
      data_points?: number
      message?: string
      interest_graph?: { date: string; value: number }[]
      related_queries?: string[]
    }
    history_data?: {
      status: string
      periods?: {
        "24h":    { avg: number; peak: number }
        "7gg":    { avg: number; peak: number }
        "3mesi":  { avg: number; peak: number }
      }
    }
    verdetto?: string
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const TREND_CONFIG: Record<string, { color: string; bg: string; icon: JSX.Element; label: string }> = {
  "forte crescita": {
    color: "#ef4444",
    bg: "rgba(239,68,68,0.08)",
    icon: <WhatshotIcon sx={{ color: "#ef4444", fontSize: 22 }} />,
    label: "Forte Crescita",
  },
  "in crescita": {
    color: "#10b981",
    bg: "rgba(16,185,129,0.08)",
    icon: <TrendingUpIcon sx={{ color: "#10b981", fontSize: 22 }} />,
    label: "In Crescita",
  },
  "stabile": {
    color: "#6366f1",
    bg: "rgba(99,102,241,0.08)",
    icon: <TrendingFlatIcon sx={{ color: "#6366f1", fontSize: 22 }} />,
    label: "Stabile",
  },
  "in calo": {
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.08)",
    icon: <TrendingDownIcon sx={{ color: "#f59e0b", fontSize: 22 }} />,
    label: "In Calo",
  },
  "forte calo": {
    color: "#94a3b8",
    bg: "rgba(148,163,184,0.08)",
    icon: <ArrowDownwardIcon sx={{ color: "#94a3b8", fontSize: 22 }} />,
    label: "Forte Calo",
  },
}

const getTrendCfg = (dir?: string) => TREND_CONFIG[dir ?? "stabile"] ?? TREND_CONFIG["stabile"]

const fmt = (n?: number) =>
  n !== undefined ? n.toLocaleString("it-IT", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : "—"

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Stat pill usata nella sezione statistiche avanzate */
function StatPill({ label, value, tooltip }: { label: string; value: string; tooltip?: string }) {
  return (
    <Tooltip title={tooltip ?? ""} placement="top">
      <Box sx={{
        flex: 1,
        minWidth: 80,
        p: 1.5,
        bgcolor: "#f8fafc",
        borderRadius: 3,
        border: "1px solid #e2e8f0",
        textAlign: "center",
        cursor: tooltip ? "help" : "default",
      }}>
        <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 700, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.6, display: "block" }}>
          {label}
        </Typography>
        <Typography variant="body2" sx={{ fontWeight: 800, color: "#0f172a", mt: 0.3 }}>
          {value}
        </Typography>
      </Box>
    </Tooltip>
  )
}

/** Badge con colore per ogni fascia di consistenza prezzi */
function ConsistencyBadge({ text }: { text: string }) {
  const isUniform  = text.includes("uniformi")
  const isVariable = text.includes("molto variabili")
  const color  = isUniform ? "#10b981" : isVariable ? "#ef4444" : "#f59e0b"
  const bg     = isUniform ? "rgba(16,185,129,0.08)" : isVariable ? "rgba(239,68,68,0.08)" : "rgba(245,158,11,0.08)"

  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, px: 1.2, py: 0.4, bgcolor: bg, borderRadius: 2 }}>
      <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: color }} />
      <Typography variant="caption" sx={{ color, fontWeight: 700, fontSize: 11 }}>
        {text}
      </Typography>
    </Box>
  )
}

/** Barre colorate per la sezione storico interesse */
function HistoryBar({ label, avg, peak }: { label: string; avg: number; peak: number }) {
  return (
    <Box sx={{ flex: 1 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
        <Typography variant="caption" sx={{ fontWeight: 700, color: "#475569", fontSize: 11 }}>{label}</Typography>
        <Typography variant="caption" sx={{ fontWeight: 800, color: "#6366f1", fontSize: 11 }}>~{avg}/100</Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={avg}
        sx={{
          height: 6, borderRadius: 3, bgcolor: "#f1f5f9",
          "& .MuiLinearProgress-bar": {
            borderRadius: 3,
            background: "linear-gradient(90deg, #6366f1 0%, #818cf8 100%)",
          }
        }}
      />
      <Typography variant="caption" sx={{ color: "#94a3b8", fontSize: 10 }}>
        picco: {peak}
      </Typography>
    </Box>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function MarketTrendsCard({ data }: MarketTrendsCardProps) {
  if (data.status !== "ok" && data.shopping_data?.status !== "ok") return null

  const { shopping_data: sd, trends_data: td, history_data: hd, verdetto } = data

  const hasShopping = sd?.status === "ok" && sd.average_price !== undefined
  const hasTrends   = td?.status === "ok" && td.trend_direction !== undefined
  const hasGraph    = (td?.interest_graph?.length ?? 0) > 0
  const hasHistory  = hd?.status === "ok" && hd.periods !== undefined
  const hasSellers  = (sd?.seller_breakdown?.length ?? 0) > 0

  if (!hasShopping && !hasTrends) return null

  const trendCfg = getTrendCfg(td?.trend_direction)

  const priceProgress =
    hasShopping && sd.max_price && sd.min_price && sd.max_price !== sd.min_price
      ? ((sd.average_price! - sd.min_price!) / (sd.max_price! - sd.min_price!)) * 100
      : 50

  // Calcola il valore medio del grafico per la reference line
  const graphAvg = hasGraph
    ? Math.round(td!.interest_graph!.reduce((s, p) => s + p.value, 0) / td!.interest_graph!.length)
    : undefined

  return (
    <Box sx={{ mt: 3, mb: 3 }}>

      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
        <AssessmentOutlinedIcon sx={{ color: "#6366f1", fontSize: 20 }} />
        <Typography variant="subtitle2" sx={{ fontWeight: 800, color: "#1e293b", letterSpacing: -0.3 }}>
          Analisi Mercato & Trend Online
        </Typography>
      </Stack>

      <Card
        elevation={0}
        sx={{
          background: "linear-gradient(160deg, #ffffff 0%, #f8faff 100%)",
          border: "1px solid #e2e8f0",
          borderRadius: 4,
          boxShadow: "0 4px 24px rgba(99,102,241,0.06)",
          overflow: "hidden",
          position: "relative",
          "&::before": {
            content: '""', position: "absolute",
            top: 0, left: 0, width: "4px", height: "100%",
            background: "linear-gradient(180deg, #6366f1 0%, #818cf8 100%)",
          },
        }}
      >
        <CardContent sx={{ p: 3, "&:last-child": { pb: 3 } }}>

          {/* Query label */}
          <Box sx={{ mb: 2.5 }}>
            <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 700, letterSpacing: 1, fontSize: 10, textTransform: "uppercase", display: "block", mb: 0.4 }}>
              Prodotto Analizzato
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 700, color: "#0f172a" }}>
              {data.query}
            </Typography>
          </Box>

          <Grid container spacing={3}>

            {/* ── SHOPPING SECTION ─────────────────────────────── */}
            {hasShopping && (
              <Grid size={{ xs: 12, md: hasTrends ? 6 : 12 }}>
                <Stack spacing={2.5}>

                  {/* Prezzo medio */}
                  <Stack direction="row" spacing={2} alignItems="flex-start">
                    <Box sx={{ p: 1.5, bgcolor: "rgba(99,102,241,0.08)", borderRadius: 3, display: "flex" }}>
                      <ShoppingCartIcon sx={{ color: "#6366f1", fontSize: 22 }} />
                    </Box>
                    <Box sx={{ flex: 1 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.2 }}>
                        <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 700, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.8 }}>
                          Prezzo Medio di Mercato
                        </Typography>
                        <Tooltip title="Media calcolata con filtro IQR sui risultati Google Shopping in Italia">
                          <InfoOutlinedIcon sx={{ fontSize: 13, color: "#cbd5e1", cursor: "help" }} />
                        </Tooltip>
                      </Box>
                      <Typography variant="h4" sx={{ fontWeight: 900, color: "#0f172a", lineHeight: 1, display: "flex", alignItems: "baseline", gap: 0.5 }}>
                        €{fmt(sd.average_price)}
                        <Typography component="span" sx={{ fontSize: 13, fontWeight: 500, color: "#94a3b8" }}>EUR</Typography>
                      </Typography>
                    </Box>
                  </Stack>

                  {/* Range bar */}
                  <Box>
                    <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.8 }}>
                      <Typography variant="caption" sx={{ fontWeight: 700, color: "#64748b" }}>Min €{fmt(sd.min_price)}</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 700, color: "#64748b" }}>Max €{fmt(sd.max_price)}</Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={priceProgress}
                      sx={{
                        height: 10, borderRadius: 5, bgcolor: "#f1f5f9",
                        "& .MuiLinearProgress-bar": {
                          borderRadius: 5,
                          background: "linear-gradient(90deg, #6366f1 0%, #818cf8 100%)",
                        },
                      }}
                    />
                    <Typography variant="caption" sx={{ color: "#94a3b8", display: "block", mt: 0.8, fontSize: 11 }}>
                      Campione di {sd.samples} offerte attive · mediana €{fmt(sd.median_price)}
                    </Typography>
                  </Box>

                  {/* Stat pills — NUOVO */}
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <StatPill label="Mediana"    value={`€${fmt(sd.median_price)}`} tooltip="Il prezzo centrale, meno sensibile agli outlier rispetto alla media" />
                    <StatPill label="Dev. Std"   value={`€${fmt(sd.std_dev)}`}      tooltip="Deviazione standard: quanto variano i prezzi attorno alla media" />
                    <StatPill label="Spread"     value={`€${fmt(sd.price_range)}`}  tooltip="Differenza tra prezzo massimo e minimo rilevato" />
                  </Stack>

                  {/* Consistenza prezzi — NUOVO */}
                  {sd.price_consistency && (
                    <ConsistencyBadge text={sd.price_consistency} />
                  )}
                </Stack>
              </Grid>
            )}

            {/* ── TRENDS SECTION ───────────────────────────────── */}
            {hasTrends && (
              <Grid size={{ xs: 12, md: hasShopping ? 6 : 12 }}>
                <Stack spacing={2}>

                  {/* Trend header */}
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Box sx={{ p: 1.5, bgcolor: trendCfg.bg, borderRadius: 3, display: "flex" }}>
                      {trendCfg.icon}
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 700, fontSize: 10, textTransform: "uppercase", letterSpacing: 0.8 }}>
                        Interesse Online
                      </Typography>
                      <Stack direction="row" alignItems="center" spacing={1}>
                        <Typography variant="h6" sx={{ fontWeight: 800, color: trendCfg.color, lineHeight: 1 }}>
                          {trendCfg.label}
                        </Typography>
                        <Chip
                          label={`${td!.current_interest}/100`}
                          size="small"
                          sx={{ bgcolor: trendCfg.bg, color: trendCfg.color, fontWeight: 800, fontSize: 11, height: 22, border: "none" }}
                        />
                      </Stack>
                    </Box>
                  </Stack>

                  {/* Area chart con reference line — MIGLIORATO */}
                  {hasGraph && (
                    <Box sx={{ height: 110, width: "100%" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={td!.interest_graph} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                          <defs>
                            <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%"  stopColor={trendCfg.color} stopOpacity={0.25} />
                              <stop offset="95%" stopColor={trendCfg.color} stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                          <XAxis dataKey="date" hide />
                          <YAxis hide domain={[0, 100]} />
                          {graphAvg !== undefined && (
                            <ReferenceLine
                              y={graphAvg}
                              stroke={trendCfg.color}
                              strokeDasharray="4 4"
                              strokeOpacity={0.4}
                            />
                          )}
                          <ChartTooltip
                            contentStyle={{ borderRadius: 8, border: "none", boxShadow: "0 4px 16px rgba(0,0,0,0.08)", fontSize: 12 }}
                            labelStyle={{ fontWeight: 700 }}
                            formatter={(value: any) => [`${value}/100`, "Interesse"]}
                          />
                          <Area
                            type="monotone"
                            dataKey="value"
                            stroke={trendCfg.color}
                            strokeWidth={2}
                            fill="url(#trendGrad)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </Box>
                  )}

                  {/* Storico multi-periodo — NUOVO */}
                  {hasHistory && hd!.periods && (
                    <Box sx={{ p: 1.5, bgcolor: "#f8fafc", borderRadius: 3, border: "1px solid #f1f5f9" }}>
                      <Stack direction="row" alignItems="center" spacing={0.8} sx={{ mb: 1.5 }}>
                        <LeaderboardIcon sx={{ fontSize: 14, color: "#6366f1" }} />
                        <Typography variant="caption" sx={{ fontWeight: 800, color: "#475569", textTransform: "uppercase", letterSpacing: 0.6, fontSize: 10 }}>
                          Storico Interesse
                        </Typography>
                      </Stack>
                      <Stack direction="row" spacing={2}>
                        {(["24h", "7gg", "3mesi"] as const).map((k) => (
                          <HistoryBar key={k} label={k} avg={hd!.periods![k].avg} peak={hd!.periods![k].peak} />
                        ))}
                      </Stack>
                    </Box>
                  )}
                </Stack>
              </Grid>
            )}
          </Grid>

          {/* ── VERDETTO ─────────────────────────────────────────── */}
          {verdetto && (
            <Box sx={{
              mt: 3, p: 2.5,
              background: "linear-gradient(135deg, rgba(99,102,241,0.04) 0%, rgba(129,140,248,0.06) 100%)",
              borderRadius: 3,
              border: "1px dashed rgba(99,102,241,0.22)",
              display: "flex", gap: 2, alignItems: "flex-start",
            }}>
              <AutoAwesomeIcon sx={{ color: "#6366f1", mt: 0.2, flexShrink: 0 }} />
              <Box>
                <Typography variant="caption" sx={{ color: "#6366f1", fontWeight: 800, textTransform: "uppercase", letterSpacing: 1, display: "block", mb: 0.5 }}>
                  Verdetto di Mercato
                </Typography>
                <Typography variant="body2" sx={{ color: "#334155", lineHeight: 1.6, fontWeight: 500 }}>
                  {verdetto}
                </Typography>
              </Box>
            </Box>
          )}

          {/* ── SELLER BREAKDOWN — NUOVO ─────────────────────────── */}
          {hasSellers && (
            <Box sx={{ mt: 3 }}>
              <Divider sx={{ mb: 2, borderColor: "#f1f5f9" }} />
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
                <StorefrontIcon sx={{ fontSize: 16, color: "#6366f1" }} />
                <Typography variant="caption" sx={{ fontWeight: 800, color: "#0f172a", textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Confronto Venditori
                </Typography>
                <Chip label="ordinati per prezzo" size="small" sx={{ fontSize: 10, height: 18, color: "#94a3b8", bgcolor: "#f8fafc", border: "1px solid #e2e8f0" }} />
              </Stack>

              <Stack spacing={1}>
                {sd!.seller_breakdown!.slice(0, 5).map((item, idx) => (
                  <Box
                    key={idx}
                    component={item.link ? "a" : "div"}
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
                      display: "flex", alignItems: "center", gap: 1.5,
                      p: 1.2, borderRadius: 2.5,
                      bgcolor: idx === 0 ? "rgba(99,102,241,0.03)" : "#fafafa",
                      border: `1px solid ${idx === 0 ? "rgba(99,102,241,0.2)" : "#f1f5f9"}`,
                      textDecoration: "none",
                      transition: "all 0.18s ease",
                      "&:hover": { borderColor: "#6366f1", boxShadow: "0 2px 10px rgba(99,102,241,0.1)", transform: "translateY(-1px)" },
                    }}
                  >
                    {/* Rank badge */}
                    <Box sx={{
                      width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                      bgcolor: idx === 0 ? "#6366f1" : "#f1f5f9",
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}>
                      <Typography variant="caption" sx={{ fontWeight: 900, fontSize: 11, color: idx === 0 ? "#fff" : "#94a3b8" }}>
                        {idx + 1}
                      </Typography>
                    </Box>

                    {item.thumbnail && (
                      <Box component="img" src={item.thumbnail} sx={{ width: 36, height: 36, borderRadius: 1.5, objectFit: "contain", bgcolor: "#fff", flexShrink: 0 }} />
                    )}

                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600, color: "#1e293b", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontSize: 13 }}>
                        {item.title}
                      </Typography>
                      <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 500 }}>
                        {item.seller}
                      </Typography>
                    </Box>

                    <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flexShrink: 0 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 900, color: "#6366f1", fontSize: 14 }}>
                        €{item.price.toLocaleString("it-IT", { minimumFractionDigits: 2 })}
                      </Typography>
                      {item.link && <OpenInNewIcon sx={{ fontSize: 12, color: "#cbd5e1" }} />}
                    </Stack>
                  </Box>
                ))}
              </Stack>
            </Box>
          )}

          {/* ── TOP RESULTS (fallback se non c'è seller_breakdown) ── */}
          {!hasSellers && sd?.top_results && sd.top_results.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Divider sx={{ mb: 2, borderColor: "#f1f5f9" }} />
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
                <LocalOfferIcon sx={{ fontSize: 16, color: "#6366f1" }} />
                <Typography variant="caption" sx={{ fontWeight: 800, color: "#0f172a", textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Migliori Offerte Online
                </Typography>
              </Stack>
              <Stack spacing={1}>
                {sd.top_results.map((item, idx) => (
                  <Box
                    key={idx}
                    component={item.link ? "a" : "div"}
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    sx={{
                      display: "flex", alignItems: "center", gap: 2,
                      p: 1.2, borderRadius: 2, bgcolor: "#fff",
                      border: "1px solid #f1f5f9", textDecoration: "none",
                      transition: "all 0.2s",
                      "&:hover": { borderColor: "#6366f1", boxShadow: "0 2px 8px rgba(99,102,241,0.08)", transform: "translateY(-1px)" },
                    }}
                  >
                    {item.thumbnail && (
                      <Box component="img" src={item.thumbnail} sx={{ width: 44, height: 44, borderRadius: 1.5, objectFit: "contain", bgcolor: "#fff" }} />
                    )}
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600, color: "#1e293b", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {item.title}
                      </Typography>
                      <Typography variant="caption" sx={{ color: "#64748b", fontWeight: 500 }}>
                        {item.source}
                      </Typography>
                    </Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 800, color: "#6366f1", whiteSpace: "nowrap" }}>
                      {item.price}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Box>
          )}

          {/* ── RELATED QUERIES ──────────────────────────────────── */}
          {td?.related_queries && td.related_queries.length > 0 && (
            <Box sx={{ mt: 3 }}>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                <SearchIcon sx={{ fontSize: 15, color: "#94a3b8" }} />
                <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5, fontSize: 10 }}>
                  Ricerche Correlate
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {td.related_queries.map((q, idx) => (
                  <Chip
                    key={idx}
                    label={q}
                    size="small"
                    variant="outlined"
                    sx={{
                      borderRadius: 1.5, fontSize: "11px", height: "24px",
                      color: "#475569", borderColor: "#e2e8f0", bgcolor: "#fff",
                      "&:hover": { bgcolor: "#f1f5f9" },
                    }}
                  />
                ))}
              </Stack>
            </Box>
          )}

        </CardContent>
      </Card>
    </Box>
  )
}