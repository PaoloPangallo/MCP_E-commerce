import { useState, memo } from "react"
import { Box, Collapse, Paper, Typography } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import type { SearchBlock } from "../search/types.ts"
import { ThinkingPill } from "../agent/components/ThinkingPill.tsx"
import ItemDetailsCard from "./ItemDetailsCard.tsx"
import ShippingCostsCard from "./ShippingCostsCard.tsx"
import MarketTrendsCard from "./MarketTrendsCard.tsx"
import ComparisonDisplay from "../search/components/ComparisonDisplay.tsx"
import SearchResultList from "../search/components/SearchResultList.tsx"
import SellerSummaryCard from "../seller/component/SellerSummaryCard.tsx"
import MetadataCard from "./MetadataCard.tsx"
import DealsDisplay from "../search/components/DealsDisplay.tsx"
import VisionAnalysisCard from "./VisionAnalysisCard.tsx"

import ToolSkeleton from "../../components/ToolSkeleton.tsx"

function CollapsibleSection({
  label,
  count,
  defaultOpen = false,
  children
}: {
  label: string
  count?: number
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <Box
      sx={{
        border: "1px solid var(--border-color)",
        borderRadius: "16px", // Standardized to 16px
        overflow: "hidden",
        bgcolor: "var(--bg-primary)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.02)",
        transition: "all 0.2s ease"
      }}
    >
      <Box
        onClick={() => setOpen((v) => !v)}
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 2.5,
          py: 1.5,
          cursor: "pointer",
          userSelect: "none",
          bgcolor: open ? "var(--bg-secondary)" : "var(--bg-primary)",
          "&:hover": { bgcolor: "var(--bg-secondary)" },
          transition: "background 0.2s"
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
           <Typography sx={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", letterSpacing: '-0.01em' }}>
            {label}
          </Typography>
          {count !== undefined && (
            <Box
              sx={{
                px: 1,
                py: 0.25,
                bgcolor: "var(--bg-secondary)",
                borderRadius: "6px",
                fontSize: 11,
                color: "var(--text-secondary)",
                fontWeight: 700
              }}
            >
              {count}
            </Box>
          )}
        </Box>
        <KeyboardArrowDownIcon
          sx={{
            fontSize: 18,
            color: "var(--text-secondary)",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
          }}
        />
      </Box>

      <Collapse in={open} timeout={300}>
        <Box sx={{ borderTop: "1px solid var(--border-color)" }}>{children}</Box>
      </Collapse>
    </Box>
  )
}

const SearchBlockView = memo(function SearchBlockView({ 
  search,
  hideTrace = false
}: { 
  search: SearchBlock,
  hideTrace?: boolean 
}) {
  const hasSeller = !!search.seller_summary?.seller_name
  const hasResults = Array.isArray(search.results) && search.results.length > 0
  const hasComparison =
    !!search.comparison &&
    Array.isArray(search.comparison.comparison_matrix) &&
    search.comparison.comparison_matrix.length > 0
  const hasMetadata = !!search.metadata && search.metadata.status === "ok"
  const agentTrace = Array.isArray(search.agent_trace) ? search.agent_trace : []
  const hasTrace = agentTrace.length > 0

  // 🔹 TOOL RUNNING STATE DETECTION
  const isSearchRunning = agentTrace.some(s => (s.action?.includes("search") || s.action?.includes("get_ebay_deals")) && s.status === "running") && !hasResults && !search.deals
  const isComparisonRunning = agentTrace.some(s => s.action?.includes("compare") && s.status === "running") && !hasComparison
  const isSellerRunning = agentTrace.some(s => s.action?.includes("analyze_seller") && s.status === "running") && !hasSeller
  const isTrendsRunning = agentTrace.some(s => s.action?.includes("market_trends") && s.status === "running") && !search.market_trends

  const showSellerCard =
    hasSeller && (search.mode === "seller" || search.mode === "hybrid")

  return (
    <Box sx={{ mb: 3, display: "flex", flexDirection: "column", gap: 1.5 }}>

      {hasTrace && !hideTrace && (
        <ThinkingPill steps={agentTrace} loading={false} query={search.query} />
      )}

      {search.vision_analysis && (
        <VisionAnalysisCard
          description={search.vision_analysis.description}
          tags={search.vision_analysis.tags}
        />
      )}

      {/* 🔹 SKELETONS */}
      {isSearchRunning && <ToolSkeleton type="search" />}
      {isComparisonRunning && <ToolSkeleton type="comparison" />}
      {isSellerRunning && <ToolSkeleton type="seller" />}
      {isTrendsRunning && <ToolSkeleton type="trends" />}

      {search.mode !== "seller" && hasResults && !hasComparison && search.results.length >= 2 && (
        <ComparisonDisplay
          data={{
            status: "ok",
            queries_compared: 1,
            candidates_found: search.results.length,
            winner: { 
              ...search.results[0], 
              query: search.query, 
              scores: {
                overall: search.results[0].ranking_score || (search.results[0] as any)._scores?.overall || 0,
                relevance: (search.results[0] as any)._scores?.relevance || 0.8,
                trust: search.results[0].trust_score || 0.9,
                price: (search.results[0] as any)._scores?.price || 0.7,
                condition: 1.0
              }
            } as any,
            winner_reason: "Il sistema ha selezionato questo risultato come migliore compromesso considerando aderenza alla ricerca, prezzo e affidabilità del venditore.",
            comparison_matrix: search.results.map(r => ({ 
              ...r, 
              query: search.query, 
              scores: {
                overall: r.ranking_score || (r as any)._scores?.overall || 0,
                relevance: (r as any)._scores?.relevance || 0.8,
                trust: r.trust_score || 0.9,
                price: (r as any)._scores?.price || 0.7,
                condition: 1.0
              }
            })) as any[]
          }}
        />
      )}

      {hasMetadata && (
        <CollapsibleSection label="Metadata marketplace" defaultOpen={true}>
          <Box sx={{ p: 1.5 }}>
            <MetadataCard data={search.metadata} />
          </Box>
        </CollapsibleSection>
      )}

      {showSellerCard && (
        <SellerSummaryCard
          sellerName={search.seller_summary?.seller_name}
          trustScore={search.seller_summary?.trust_score}
          sentimentScore={search.seller_summary?.sentiment_score}
          count={search.seller_summary?.count}
          feedbacks={search.seller_summary?.feedbacks}
        />
      )}

      {hasComparison && (
        <ComparisonDisplay 
          data={{
            ...search.comparison!,
            comparison_matrix: search.results.map(r => {
              // Ensure we have score structure for all items
              const existing = search.comparison!.comparison_matrix.find(c => c.ebay_id === r.ebay_id);
              return {
                ...r,
                scores: existing?.scores || (r as any).scores || (r as any)._scores || {
                  overall: r.ranking_score || 0,
                  relevance: 0.8,
                  trust: r.trust_score || 0.9,
                  price: 0.7,
                  value: (r as any).value_score || 0
                }
              }
            }) as any[]
          }} 
        />
      )}

      {hasResults && (
        <CollapsibleSection 
          label={hasComparison ? "Tutti i risultati" : "Risultati della ricerca"} 
          count={search.results.length} 
          defaultOpen={!hasComparison}
        >
          <Box sx={{ p: 2 }}>
            <SearchResultList results={search.results} />
          </Box>
        </CollapsibleSection>
      )}

      {search.item_details && <ItemDetailsCard data={search.item_details} />}

      {search.shipping_costs && <ShippingCostsCard data={search.shipping_costs} />}

      {search.market_trends && <MarketTrendsCard data={search.market_trends} />}
      {search.deals && <DealsDisplay data={search.deals} />}


      {search.errors && search.errors.length > 0 && (
        <Paper
          elevation={0}
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: "var(--bg-secondary)",
            border: "1px solid var(--danger)"
          }}
        >
          <Typography sx={{ fontSize: 12, fontWeight: 600, color: "var(--danger)", mb: 0.5 }}>
            Errori backend
          </Typography>
          {search.errors.map((err, idx) => (
            <Typography
              key={`${err}-${idx}`}
              sx={{ fontSize: 12, color: "var(--text-primary)", lineHeight: 1.6 }}
            >
              {err}
            </Typography>
          ))}
        </Paper>
      )}
    </Box>
  )
})

export default SearchBlockView