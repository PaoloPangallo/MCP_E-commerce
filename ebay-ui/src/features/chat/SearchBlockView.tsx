import { Box, Paper, Typography } from "@mui/material"
import type { SearchBlock } from "../../types/searchTypes.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx"
import ItemDetailsCard from "./ItemDetailsCard.tsx"
import ShippingCostsCard from "./ShippingCostsCard.tsx"
import SearchResultList from "../search/components/SearchResultList.tsx"
import ComparisonDisplay from "../search/components/ComparisonDisplay.tsx"
import SellerSummaryCard from "../seller/component/SellerSummaryCard.tsx"

export default function SearchBlockView({ search }: { search: SearchBlock }) {
    const hasSeller = !!search.seller_summary?.seller_name
    const hasResults = Array.isArray(search.results) && search.results.length > 0
    const hasComparison = !!search.comparison && Array.isArray(search.comparison.comparison_matrix) && search.comparison.comparison_matrix.length > 0
    const hasMetadata = !!search.metadata && search.metadata.status === "ok"
    const hasTrace = Array.isArray(search.agent_trace) && search.agent_trace.length > 0

    const showSellerCard =
        hasSeller && (search.mode === "seller" || search.mode === "hybrid")

    return (
        <Box sx={{ mt: 1.5, mb: 4 }}>
            {hasTrace ? (
                <Paper
                    elevation={0}
                    sx={{
                        p: 2.5,
                        borderRadius: 4,
                        border: "1px solid #e5e7eb",
                        bgcolor: "#ffffff"
                    }}
                >
                    <AIThinkingPipeline
                        agentTrace={search.agent_trace}
                        query={search.query}
                    />
                </Paper>
            ) : null}

          {/* AI Analysis card removed as it repeats final answer synthesis */}

            {search.mode !== "seller" && hasResults ? (
                <Box mt={2.5}>
                    <Box 
                      sx={{ 
                        borderRadius: 4, 
                        border: "1px solid #e5e7eb", 
                        overflow: "hidden",
                        bgcolor: "#fff"
                      }}
                    >
                      <Box 
                        sx={{ 
                          p: 1.5, 
                          display: "flex", 
                          justifyContent: "space-between", 
                          alignItems: "center",
                          cursor: "pointer",
                          transition: "background 0.2s",
                          "&:hover": { bgcolor: "#f9fafb" }
                        }}
                        onClick={() => {
                          const el = document.getElementById(`results-content-${search.query.replace(/\s+/g, '-')}`);
                          if (el) el.style.display = el.style.display === "none" ? "block" : "none";
                        }}
                      >
                        <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#4b5563", textTransform: "uppercase", letterSpacing: 0.5 }}>
                          📦 Elenco card annunci ({search.results.length})
                        </Typography>
                        <Typography sx={{ fontSize: 11, color: "#9ca3af", fontWeight: 500 }}>
                          Clicca per espandere card complete
                        </Typography>
                      </Box>
                      <Box 
                        id={`results-content-${search.query.replace(/\s+/g, '-')}`}
                        sx={{ display: "none", borderTop: "1px solid #e5e7eb" }}
                      >
                        <SearchResultList
                            results={search.results}
                            aspect_distributions={search.aspect_distributions}
                        />
                      </Box>
                    </Box>
                </Box>
            ) : null}

            {hasMetadata ? (
                <Box mt={2.5}>
                    <Typography
                        variant="h6"
                        sx={{
                            fontSize: 16,
                            fontWeight: 700,
                            mb: 1.5,
                            color: "#111827",
                            display: "flex",
                            alignItems: "center",
                            gap: 1
                        }}
                    >
                        📋 Metadata Marketplace
                    </Typography>
                    <Paper
                        elevation={0}
                        sx={{
                            p: 2,
                            borderRadius: 3,
                            border: "1px solid #e5e7eb",
                            bgcolor: "#f9fafb",
                            maxHeight: 400,
                            overflow: "auto"
                        }}
                    >
                        <Typography
                            sx={{ fontSize: 13, color: "#374151", whiteSpace: "pre-wrap", fontFamily: "monospace" }}
                        >
                            {JSON.stringify(search.metadata.results, null, 2)}
                        </Typography>
                    </Paper>
                </Box>
            ) : null}

            {showSellerCard ? (
                <SellerSummaryCard
                    sellerName={search.seller_summary?.seller_name}
                    trustScore={search.seller_summary?.trust_score}
                    sentimentScore={search.seller_summary?.sentiment_score}
                    count={search.seller_summary?.count}
                    feedbacks={search.seller_summary?.feedbacks}
                />
            ) : null}

            {hasComparison ? (
                <Box mt={2.5}>
                    <ComparisonDisplay data={search.comparison!} />
                </Box>
            ) : null}

            {search.item_details ? (
                <Box mt={2.5}>
                    <ItemDetailsCard data={search.item_details} />
                </Box>
            ) : null}

            {search.shipping_costs ? (
                <Box mt={2.5}>
                    <ShippingCostsCard data={search.shipping_costs} />
                </Box>
            ) : null}

            {search.errors && search.errors.length > 0 ? (
                <Paper
                    elevation={0}
                    sx={{
                        mt: 2.5,
                        p: 2.25,
                        borderRadius: 3,
                        bgcolor: "#fff7f7",
                        border: "1px solid #f2d6d6"
                    }}
                >
                    <Typography
                        sx={{
                            fontSize: 13,
                            fontWeight: 700,
                            color: "#9f2d2d",
                            mb: 0.75
                        }}
                    >
                        Errori o segnali backend
                    </Typography>

                    {search.errors.map((err, idx) => (
                        <Typography
                            key={`${err}-${idx}`}
                            sx={{ fontSize: 13, color: "#7a4b4b", lineHeight: 1.6 }}
                        >
                            {err}
                        </Typography>
                    ))}
                </Paper>
            ) : null}
        </Box>
    )
}
