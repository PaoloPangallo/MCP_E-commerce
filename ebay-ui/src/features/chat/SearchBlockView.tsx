import { useState } from "react"
import { Box, Collapse, Paper, Typography } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import type { SearchBlock } from "../../types/searchTypes.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx"
import ItemDetailsCard from "./ItemDetailsCard.tsx"
import ShippingCostsCard from "./ShippingCostsCard.tsx"
import SearchResultList from "../search/components/SearchResultList.tsx"
import ComparisonDisplay from "../search/components/ComparisonDisplay.tsx"
import SellerSummaryCard from "../seller/component/SellerSummaryCard.tsx"

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
        borderRadius: 3,
        overflow: "hidden",
        bgcolor: "#fff"
      }}
    >
      <Box
        onClick={() => setOpen((v) => !v)}
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 2,
          py: 1.25,
          cursor: "pointer",
          "&:hover": { bgcolor: "#fafafa" },
          transition: "background 0.15s"
        }}
      >
        <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#6b7280" }}>
          {label}
          {count !== undefined && (
            <Box
              component="span"
              sx={{
                ml: 1,
                px: 0.75,
                py: 0.15,
                bgcolor: "#f3f4f6",
                borderRadius: 1,
                fontSize: 11,
                color: "#9ca3af",
                fontWeight: 500
              }}
            >
              {count}
            </Box>
          )}
        </Typography>
        <KeyboardArrowDownIcon
          sx={{
            fontSize: 16,
            color: "#9ca3af",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.2s"
          }}
        />
      </Box>

      <Collapse in={open} timeout={200}>
        <Box sx={{ borderTop: "1px solid #f5f5f5" }}>{children}</Box>
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
        <CollapsibleSection label="Traccia agente" count={agentTrace.length}>
          <Box sx={{ p: 2 }}>
            <AIThinkingPipeline agentTrace={agentTrace} query={search.query} />
          </Box>
        </CollapsibleSection>
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
        <CollapsibleSection label="Metadata marketplace">
          <Box sx={{ p: 2, maxHeight: 320, overflowY: "auto" }}>
            <Typography
              sx={{
                fontSize: 12,
                color: "#374151",
                whiteSpace: "pre-wrap",
                fontFamily: "monospace"
              }}
            >
              {JSON.stringify(search.metadata.results, null, 2)}
            </Typography>
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