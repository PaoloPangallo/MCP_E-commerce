import { useState } from "react"
import { Box, Collapse, Paper, Typography } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import type { SearchBlock } from "../search/types.ts"
import { ThinkingPill } from "../agent/components/ThinkingPill.tsx"
import ItemDetailsCard from "./ItemDetailsCard.tsx"
import ShippingCostsCard from "./ShippingCostsCard.tsx"
import MarketTrendsCard from "./MarketTrendsCard.tsx"
import SearchResultList from "../search/components/SearchResultList.tsx"
import ComparisonDisplay from "../search/components/ComparisonDisplay.tsx"
import SellerSummaryCard from "../seller/component/SellerSummaryCard.tsx"
import MetadataCard from "./MetadataCard.tsx"
import DealsDisplay from "../search/components/DealsDisplay.tsx"
import VisionAnalysisCard from "./VisionAnalysisCard.tsx"

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
        border: "1px solid #f0f0f0",
        borderRadius: "12px",
        overflow: "hidden",
        bgcolor: "#fff",
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
          bgcolor: open ? "#fafafa" : "#fff",
          "&:hover": { bgcolor: "#fafafa" },
          transition: "background 0.2s"
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
           <Typography sx={{ fontSize: 13, fontWeight: 600, color: "#374151", letterSpacing: '-0.01em' }}>
            {label}
          </Typography>
          {count !== undefined && (
            <Box
              sx={{
                px: 1,
                py: 0.25,
                bgcolor: "#f3f4f6",
                borderRadius: "6px",
                fontSize: 11,
                color: "#6b7280",
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
            color: "#9ca3af",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
          }}
        />
      </Box>

      <Collapse in={open} timeout={300}>
        <Box sx={{ borderTop: "1px solid #f0f0f0" }}>{children}</Box>
      </Collapse>
    </Box>
  )
}

export default function SearchBlockView({ search }: { search: SearchBlock }) {
  const hasSeller = !!search.seller_summary?.seller_name
  const hasResults = Array.isArray(search.results) && search.results.length > 0
  const hasComparison =
    !!search.comparison &&
    Array.isArray(search.comparison.comparison_matrix) &&
    search.comparison.comparison_matrix.length > 0
  const hasMetadata = !!search.metadata && search.metadata.status === "ok"
  // Guard against possibly-undefined agent_trace
  const agentTrace = Array.isArray(search.agent_trace) ? search.agent_trace : []
  const hasTrace = agentTrace.length > 0

  const showSellerCard =
    hasSeller && (search.mode === "seller" || search.mode === "hybrid")

  return (
    <Box sx={{ mb: 3, display: "flex", flexDirection: "column", gap: 1.5 }}>

      {hasTrace && (
        <ThinkingPill steps={agentTrace} loading={false} query={search.query} />
      )}

      {search.vision_analysis && (
        <VisionAnalysisCard
          description={search.vision_analysis.description}
          tags={search.vision_analysis.tags}
        />
      )}

      {search.mode !== "seller" && hasResults && (
        <CollapsibleSection
          label="Annunci trovati"
          count={search.results.length}
          defaultOpen={false}
        >
          <SearchResultList
            results={search.results}
            aspect_distributions={search.aspect_distributions}
          />
        </CollapsibleSection>
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

      {hasComparison && <ComparisonDisplay data={search.comparison!} />}

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
            bgcolor: "#fff7f7",
            border: "1px solid #fecaca"
          }}
        >
          <Typography sx={{ fontSize: 12, fontWeight: 600, color: "#b91c1c", mb: 0.5 }}>
            Errori backend
          </Typography>
          {search.errors.map((err, idx) => (
            <Typography
              key={`${err}-${idx}`}
              sx={{ fontSize: 12, color: "#7f1d1d", lineHeight: 1.6 }}
            >
              {err}
            </Typography>
          ))}
        </Paper>
      )}
    </Box>
  )
}