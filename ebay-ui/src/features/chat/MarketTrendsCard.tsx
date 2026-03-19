import { Box, Typography, Card, CardContent, Grid, Stack, LinearProgress, Tooltip } from "@mui/material"
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as ChartTooltip, 
  ResponsiveContainer 
} from "recharts"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"
import TrendingDownIcon from "@mui/icons-material/TrendingDown"
import TrendingFlatIcon from "@mui/icons-material/TrendingFlat"
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart"
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined"
import AssessmentOutlinedIcon from "@mui/icons-material/AssessmentOutlined"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

interface MarketTrendsCardProps {
  data: {
    status: string
    query: string
    shopping_data: {
      status: string
      min_price?: number
      max_price?: number
      average_price?: number
      samples?: number
      message?: string
      top_result?: {
        title: string
        source: string
        thumbnail?: string
      }
    }
    trends_data: {
      status: string
      current_interest?: number
      trend_direction?: string
      data_points?: number
      message?: string
      interest_graph?: { date: string, value: number }[]
    }
    verdetto?: string
  }
}

export default function MarketTrendsCard({ data }: MarketTrendsCardProps) {
  if (data.status !== "ok" && data.shopping_data?.status !== "ok") {
    return null
  }

  const { shopping_data, trends_data, verdetto } = data
  const hasShopping = shopping_data?.status === "ok" && shopping_data.average_price
  const hasTrends = trends_data?.status === "ok" && trends_data.trend_direction
  const hasGraph = trends_data?.interest_graph && trends_data.interest_graph.length > 0

  if (!hasShopping && !hasTrends) {
    return null
  }

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case "in crescita":
        return <TrendingUpIcon sx={{ color: "#10b981", fontSize: 24 }} />
      case "in calo":
        return <TrendingDownIcon sx={{ color: "#ef4444", fontSize: 24 }} />
      default:
        return <TrendingFlatIcon sx={{ color: "#6b7280", fontSize: 24 }} />
    }
  }

  const getTrendColor = (direction: string) => {
    switch (direction) {
      case "in crescita": return "#10b981"
      case "in calo": return "#ef4444"
      default: return "#6b7280"
    }
  }

  // Calculate where average price sits in the min-max range
  const priceProgress = hasShopping && shopping_data.max_price && shopping_data.min_price && (shopping_data.max_price !== shopping_data.min_price)
    ? ((shopping_data.average_price! - shopping_data.min_price!) / (shopping_data.max_price! - shopping_data.min_price!)) * 100
    : 50

  return (
    <Box sx={{ mt: 3, mb: 3 }}>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1.5 }}>
        <AssessmentOutlinedIcon sx={{ color: "#4f46e5", fontSize: 20 }} />
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "#1e293b", letterSpacing: -0.2 }}>
          Analisi Mercato & Trend Online
        </Typography>
      </Stack>

      <Card 
        elevation={0}
        sx={{ 
          background: "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
          border: "1px solid #e2e8f0", 
          borderRadius: 4,
          boxShadow: "0 4px 12px rgba(0,0,0,0.03)",
          overflow: "hidden",
          position: "relative",
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '4px',
            height: '100%',
            bgcolor: '#4f46e5'
          }
        }}
      >
        <CardContent sx={{ p: 2.5, "&:last-child": { pb: 2.5 } }}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ color: "#64748b", fontWeight: 500, display: 'block', mb: 0.5 }}>
              PRODOTTO ANALIZZATO
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600, color: "#0f172a", lineHeight: 1.4 }}>
              {data.query}
            </Typography>
          </Box>

          <Grid container spacing={3}>
            {/* SHOPPING DATA SECTION */}
            {hasShopping && (
              <Grid size={{ xs: 12, md: hasTrends ? 6 : 12 }}>
                <Stack spacing={2}>
                  <Stack direction="row" spacing={2} alignItems="flex-start">
                    <Box sx={{ 
                      p: 1.5, 
                      bgcolor: "rgba(79, 70, 229, 0.08)", 
                      borderRadius: 3,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <ShoppingCartIcon sx={{ color: "#4f46e5" }} />
                    </Box>
                    <Box sx={{ flex: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.2 }}>
                        <Typography variant="caption" sx={{ color: "#64748b", textTransform: 'uppercase', letterSpacing: 0.8, fontWeight: 700, fontSize: 10 }}>
                          Prezzo Medio di Mercato
                        </Typography>
                        <Tooltip title="Media calcolata sui risultati Google Shopping in Italia">
                          <InfoOutlinedIcon sx={{ fontSize: 14, color: "#94a3b8", cursor: 'help' }} />
                        </Tooltip>
                      </Box>
                      <Typography variant="h4" sx={{ fontWeight: 800, color: "#0f172a", display: 'flex', alignItems: 'baseline', gap: 0.5 }}>
                        €{shopping_data.average_price?.toLocaleString('it-IT', { minimumFractionDigits: 2 })}
                        <Typography component="span" sx={{ fontSize: 14, fontWeight: 500, color: "#64748b" }}>EUR</Typography>
                      </Typography>
                    </Box>
                  </Stack>
                  
                  {/* Price Range Visualizer */}
                  <Box sx={{ px: 0.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="caption" sx={{ fontWeight: 600, color: "#64748b" }}>Min: €{shopping_data.min_price}</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 600, color: "#64748b" }}>Max: €{shopping_data.max_price}</Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={priceProgress} 
                      sx={{ 
                        height: 8, 
                        borderRadius: 4, 
                        bgcolor: "#f1f5f9",
                        "& .MuiLinearProgress-bar": {
                          borderRadius: 4,
                          background: "linear-gradient(90deg, #4f46e5 0%, #818cf8 100%)"
                        }
                      }} 
                    />
                    <Typography variant="caption" sx={{ color: "#94a3b8", display: 'block', mt: 1, fontStyle: 'italic', fontSize: 11 }}>
                      Basato su un campione di {shopping_data.samples} offerte attive online.
                    </Typography>
                  </Box>
                </Stack>
              </Grid>
            )}

            {/* TRENDS DATA SECTION */}
            {hasTrends && (
              <Grid size={{ xs: 12, md: hasShopping ? 6 : 12 }}>
                <Stack spacing={2}>
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Box sx={{ 
                      p: 1.5, 
                      bgcolor: trends_data.trend_direction === "in crescita" ? "rgba(16, 185, 129, 0.08)" : "rgba(100, 116, 139, 0.08)", 
                      borderRadius: 3,
                      display: 'flex'
                    }}>
                      {getTrendIcon(trends_data.trend_direction || "stabile")}
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ color: "#64748b", textTransform: 'uppercase', letterSpacing: 0.8, fontWeight: 700, fontSize: 10 }}>
                        Tendenza Interesse Online
                      </Typography>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: getTrendColor(trends_data.trend_direction || "stabile"), textTransform: 'capitalize' }}>
                        {trends_data.trend_direction} ({trends_data.current_interest}/100)
                      </Typography>
                    </Box>
                  </Stack>

                  {/* MINI CHART SECTION */}
                  {hasGraph && (
                    <Box sx={{ height: 100, width: '100%', mt: 1 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={trends_data.interest_graph}>
                          <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor={getTrendColor(trends_data.trend_direction || "stabile")} stopOpacity={0.3}/>
                              <stop offset="95%" stopColor={getTrendColor(trends_data.trend_direction || "stabile")} stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                          <XAxis 
                            dataKey="date" 
                            hide={true}
                          />
                          <YAxis hide={true} domain={[0, 'auto']} />
                          <ChartTooltip 
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)', fontSize: '12px' }}
                            labelStyle={{ fontWeight: 'bold', marginBottom: '4px' }}
                          />
                          <Area 
                            type="monotone" 
                            dataKey="value" 
                            stroke={getTrendColor(trends_data.trend_direction || "stabile")} 
                            fillOpacity={1} 
                            fill="url(#colorValue)" 
                            strokeWidth={2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </Box>
                  )}
                </Stack>
              </Grid>
            )}
          </Grid>

          {/* VERDICT SECTION */}
          {verdetto && (
            <Box sx={{ 
              mt: 3, 
              p: 2, 
              bgcolor: "rgba(79, 70, 229, 0.04)", 
              borderRadius: 3, 
              border: "1px dashed rgba(79, 70, 229, 0.2)",
              display: 'flex',
              gap: 2,
              alignItems: 'flex-start'
            }}>
              <AutoAwesomeIcon sx={{ color: "#4f46e5", mt: 0.3 }} />
              <Box>
                <Typography variant="caption" sx={{ color: "#4f46e5", fontWeight: 800, textTransform: 'uppercase', letterSpacing: 1, display: 'block', mb: 0.5 }}>
                  Verdetto di Mercato
                </Typography>
                <Typography variant="body2" sx={{ color: "#334155", lineHeight: 1.5, fontWeight: 500 }}>
                  {verdetto}
                </Typography>
              </Box>
            </Box>
          )}

          {/* TOP RESULT HINT - footer of the card */}
          {shopping_data?.top_result && (
            <Box sx={{ mt: 3, pt: 2, borderTop: "1px solid #f1f5f9", display: 'flex', alignItems: 'center', gap: 1.5 }}>
               {shopping_data.top_result.thumbnail && (
                 <Box 
                   component="img" 
                   src={shopping_data.top_result.thumbnail} 
                   sx={{ width: 40, height: 40, borderRadius: 1.5, objectFit: 'contain', bgcolor: '#fff', border: '1px solid #f1f5f9' }} 
                 />
               )}
               <Box sx={{ flex: 1 }}>
                 <Typography variant="caption" sx={{ color: "#94a3b8", display: 'block', fontSize: 10, fontWeight: 600 }}>
                   MIGLIOR MATCH SHOPPING
                 </Typography>
                 <Typography variant="caption" sx={{ color: "#475569", fontWeight: 500, display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '300px' }}>
                   {shopping_data.top_result.title} <Typography component="span" sx={{ color: "#94a3b8", mx: 0.5 }}>•</Typography> {shopping_data.top_result.source}
                 </Typography>
               </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  )
}
