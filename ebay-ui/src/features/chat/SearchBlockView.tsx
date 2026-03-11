import { Box, Paper, Typography } from "@mui/material"
import type { SearchBlock } from "../../types/searchTypes.ts"
import AIThinkingPipeline from "../agent/components/AIThinkingPipeline.tsx"
import AIAnalysisCard from "../agent/components/AIAnalysisCard.tsx"
import SearchResultList from "../search/components/SearchResultList.tsx"
import ComparisonDisplay from "../search/components/ComparisonDisplay.tsx"
import SellerSummaryCard from "../seller/component/SellerSummaryCard.tsx"

export default function SearchBlockView({ search }: { search: SearchBlock }) {
    const hasSeller = !!search.seller_summary?.seller_name
    const hasResults = Array.isArray(search.results) && search.results.length > 0
    const hasAnalysis = !!search.analysis || !!search.metrics || !!search.rag_context
    const hasComparison = !!search.comparison && Array.isArray(search.comparison.comparison_matrix) && search.comparison.comparison_matrix.length > 0
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

            {hasAnalysis ? (
                <Box mt={2.5}>
                    <AIAnalysisCard
                        text={search.analysis ?? undefined}
                        metrics={search.metrics}
                        rag_context={search.rag_context}
                    />
                </Box>
            ) : null}

            {search.mode !== "seller" && hasResults ? (
                <Box mt={2.5}>
                    <SearchResultList results={search.results} />
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
