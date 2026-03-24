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
        border: "1px solid #e2e8f0",
        bgcolor: "#ffffff",
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
              bgcolor: "#f1f5f9",
              color: "#475569",
            }}
          >
            {policyType === "item_conditions" && <GavelIcon sx={{ fontSize: 18 }} />}
            {policyType === "return_policies" && <SyncIcon sx={{ fontSize: 18 }} />}
            {policyType === "listing_structure" && <LayersIcon sx={{ fontSize: 18 }} />}
          </Box>
          <Box>
            <Typography variant="subtitle2" fontWeight={700} color="#1e293b">
              {policyType === "item_conditions" && "Condizioni Ammesse"}
              {policyType === "return_policies" && "Politiche di Reso"}
              {policyType === "listing_structure" && "Struttura Inserzioni"}
            </Typography>
            <Typography variant="caption" color="#64748b">
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
              bgcolor: "#fffbeb",
              border: "1px solid #fef3c7",
              borderRadius: 2,
            }}
          >
            <InfoOutlinedIcon sx={{ fontSize: 16, color: "#d97706" }} />
            <Typography sx={{ fontSize: 11, color: "#92400e", fontWeight: 500 }}>
              {message}
            </Typography>
          </Box>
        )}

        {/* RENDERER FOR ITEM CONDITIONS */}
        {policyType === "item_conditions" && results.itemConditionPolicies && (
          <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
            {results.itemConditionPolicies.map((policy: any, idx: number) => (
              <Box key={idx}>
                <Typography variant="caption" sx={{ fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 0.5, mb: 1, display: "block" }}>
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
                        color: "#334155",
                        borderColor: "#e2e8f0",
                        bgcolor: "#f8fafc",
                        "&:hover": { bgcolor: "#f1f5f9" }
                      }}
                    />
                  ))}
                </Box>
                {idx < results.itemConditionPolicies.length - 1 && <Divider sx={{ mt: 2.5 }} />}
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
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase" }}>Categoria</TableCell>
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase" }}>Reso</TableCell>
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase" }}>Periodo</TableCell>
                  <TableCell sx={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase" }}>Spese</TableCell>
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
                    <TableCell sx={{ fontSize: 11, color: "#64748b" }}>
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
              <Box key={idx} p={1.5} borderRadius={2} bgcolor="#f8fafc" border="1px solid #f1f5f9">
                <Typography sx={{ fontSize: 12, fontWeight: 700, mb: 1 }}>
                  Categoria: {policy.categoryId || "Globale"}
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <Chip
                    label={policy.variationsSupported ? "Variazioni Supportate" : "Singola Inserzione"}
                    size="small"
                    sx={{
                      bgcolor: policy.variationsSupported ? "#e0f2fe" : "#f1f5f9",
                      color: policy.variationsSupported ? "#0369a1" : "#64748b",
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
