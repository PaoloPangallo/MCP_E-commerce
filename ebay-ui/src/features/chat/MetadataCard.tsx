import {
  Box,
  Typography,
  Paper,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
} from "@mui/material"
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined"
import GavelIcon from "@mui/icons-material/Gavel"
import SyncIcon from "@mui/icons-material/Sync"
import LayersIcon from "@mui/icons-material/Layers"

interface MetadataProps {
  data: any
}

export default function MetadataCard({ data }: MetadataProps) {
  if (!data || data.status !== "ok") return null

  const results = data.results || {}
  const policyType = data.policy_type
  const message = data.message // Truncation warning

  return (
    <Paper
      elevation={0}
      sx={{
        width: "100%",
        borderRadius: 4,
        border: "1px solid var(--border-color)",
        bgcolor: "var(--bg-primary)",
        overflow: "hidden",
        mb: 2,
        boxShadow: "0 2px 4px -1px rgb(0 0 0 / 0.05)",
      }}
    >
      <Box p={2.5}>
        <Box display="flex" alignItems="center" gap={1.5} mb={2}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 32,
              height: 32,
              borderRadius: "10px",
              bgcolor: "var(--bg-secondary)",
              color: "var(--text-secondary)",
            }}
          >
            {policyType === "item_conditions" && <GavelIcon sx={{ fontSize: 18 }} />}
            {policyType === "return_policies" && <SyncIcon sx={{ fontSize: 18 }} />}
            {policyType === "listing_structure" && <LayersIcon sx={{ fontSize: 18 }} />}
          </Box>
          <Box>
            <Typography variant="subtitle2" fontWeight={700} color="var(--text-primary)">
              {policyType === "item_conditions" && "Condizioni Ammesse"}
              {policyType === "return_policies" && "Politiche di Reso"}
              {policyType === "listing_structure" && "Struttura Inserzioni"}
            </Typography>
            <Typography variant="caption" color="var(--text-secondary)">
              Marketplace: {data.marketplace_id} {data.category_id ? `| Categoria: ${data.category_id}` : ""}
            </Typography>
          </Box>
        </Box>

        {message && (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1,
              p: 1.5,
              mb: 2.5,
              bgcolor: "var(--bg-secondary)",
              border: "1px solid var(--border-color)",
              borderRadius: 2,
            }}
          >
            <InfoOutlinedIcon sx={{ fontSize: 16, color: "var(--accent-primary)" }} />
            <Typography sx={{ fontSize: 11, color: "var(--text-primary)", fontWeight: 500 }}>
              {message}
            </Typography>
          </Box>
        )}

        {/* RENDERER FOR ITEM CONDITIONS */}
        {policyType === "item_conditions" && results.itemConditionPolicies && (
          <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {results.itemConditionPolicies.map((policy: any, idx: number) => (
              <Box key={idx}>
                <Typography variant="caption" sx={{ fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: 0.5, mb: 1, display: "block" }}>
                  Categoria {policy.categoryId || "Globale"}
                </Typography>
                <Box display="flex" flexWrap="wrap" gap={1}>
                  {policy.itemConditions?.map((cond: any, cidx: number) => (
                    <Chip
                      key={cidx}
                      label={cond.description}
                      size="small"
                      variant="outlined"
                      sx={{
                        fontSize: "11px",
                        fontWeight: 500,
                        color: "var(--text-primary)",
                        borderColor: "var(--border-color)",
                        bgcolor: "var(--bg-secondary)",
                        "&:hover": { bgcolor: "var(--bg-primary)" }
                      }}
                    />
                  ))}
                </Box>
                {idx < results.itemConditionPolicies.length - 1 && <Divider sx={{ mt: 2.5, borderColor: "var(--border-color)" }} />}
              </Box>
            ))}
          </Box>
        )}

        {/* RENDERER FOR RETURN POLICIES */}
        {policyType === "return_policies" && results.returnPolicies && (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase" }}>Categoria</TableCell>
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase" }}>Reso</TableCell>
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase" }}>Periodo</TableCell>
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase" }}>Spese</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.returnPolicies.map((policy: any, idx: number) => (
                  <TableRow key={idx}>
                    <TableCell sx={{ fontSize: 12, fontWeight: 600 }}>{policy.categoryId || "Generale"}</TableCell>
                    <TableCell>
                      <Chip
                        label={policy.returnsAccepted ? "Sì" : "No"}
                        size="small"
                        sx={{
                          height: 20, fontSize: 10, fontWeight: 700,
                          bgcolor: policy.returnsAccepted ? "#dcfce7" : "#fee2e2",
                          color: policy.returnsAccepted ? "#166534" : "#991b1b"
                        }}
                      />
                    </TableCell>
                    <TableCell sx={{ fontSize: 12 }}>
                      {policy.returnPeriod?.value ? `${policy.returnPeriod.value} ${policy.returnPeriod.unit}` : "-"}
                    </TableCell>
                    <TableCell sx={{ fontSize: 11, color: "var(--text-secondary)" }}>
                      {policy.returnShippingCostPayer === "BUYER" ? "Acquirente" : "Venditore"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {/* RENDERER FOR LISTING STRUCTURE */}
        {policyType === "listing_structure" && results.listingStructurePolicies && (
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {results.listingStructurePolicies.map((policy: any, idx: number) => (
              <Box key={idx} p={1.5} borderRadius={2} bgcolor="var(--bg-secondary)" border="1px solid var(--border-color)">
                <Typography sx={{ fontSize: 12, fontWeight: 700, mb: 1, color: "var(--text-primary)" }}>
                  Categoria: {policy.categoryId || "Globale"}
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <Chip
                    label={policy.variationsSupported ? "Variazioni Supportate" : "Singola Inserzione"}
                    size="small"
                    sx={{
                      bgcolor: policy.variationsSupported ? "var(--bg-primary)" : "var(--bg-secondary)",
                      color: policy.variationsSupported ? "var(--accent-primary)" : "var(--text-secondary)",
                      fontWeight: 600, fontSize: 11
                    }}
                  />
                </Box>
              </Box>
            ))}
          </Box>
        )}
      </Box>
    </Paper>
  )
}
