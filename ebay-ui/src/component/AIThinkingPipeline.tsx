import { Box, Typography, LinearProgress } from "@mui/material"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked"
import AutorenewIcon from "@mui/icons-material/Autorenew"

interface Props {
  loading?: boolean
  timings?: Record<string, number>
}

interface Step {
  key: string
  label: string
  backendKeys: string[]
}

const PIPELINE: Step[] = [

  {
    key: "parse",
    label: "Parsing query",
    backendKeys: ["parse_query"]
  },

  {
    key: "search",
    label: "Searching eBay listings",
    backendKeys: ["ebay_search"]
  },

  {
    key: "feedback",
    label: "Retrieving seller evidence",
    backendKeys: ["feedback_ingest", "rag_retrieve"]
  },

  {
    key: "rerank",
    label: "Ranking products",
    backendKeys: ["rerank"]
  },

  {
    key: "explain",
    label: "Generating explanation",
    backendKeys: ["explain"]
  }

]

export default function AIThinkingPipeline({ loading, timings }: Props) {

  if (!loading && !timings) return null

  const timingKeys = timings ? Object.keys(timings) : []

  const isCompleted = (step: Step) =>
    step.backendKeys.some(key =>
      timingKeys.some(k => k.startsWith(key))
    )

  const completedCount = PIPELINE.filter(isCompleted).length

  return (

    <Box
      sx={{
        p: 2.5,
        mb: 3,
        borderRadius: 3,
        bgcolor: "#f7f7f8",
        border: "1px solid #e5e5e5"
      }}
    >

     <Box display="flex" alignItems="center" gap={1} mb={2}>

  {loading && (
    <AutorenewIcon
  sx={{
    fontSize: 18,
    color: "#10a37f",
    animation: "spin 1s linear infinite",
    "@keyframes spin": {
      from: { transform: "rotate(0deg)" },
      to: { transform: "rotate(360deg)" }
    }
  }}
/>
  )}

  <Typography
    sx={{
      fontWeight: 600,
      fontSize: 14
    }}
  >
    {loading ? "Thinking..." : "AI reasoning"}
  </Typography>

</Box>

      {PIPELINE.map((step, i) => {

        const completed = isCompleted(step)

        const running =
          loading &&
          !completed &&
          i === completedCount

        return (

          <Box
            key={step.key}
            display="flex"
            alignItems="center"
            gap={1}
            mb={1}
          >

            {completed && (
              <CheckCircleIcon
                sx={{ fontSize: 18, color: "#10a37f" }}
              />
            )}

            {running && (
              <AutorenewIcon
                sx={{
                  fontSize: 18,
                  animation: "spin 1s linear infinite",
                  "@keyframes spin": {
                    from: { transform: "rotate(0deg)" },
                    to: { transform: "rotate(360deg)" }
                  }
                }}
              />
            )}

            {!completed && !running && (
              <RadioButtonUncheckedIcon
                sx={{ fontSize: 18, color: "#bbb" }}
              />
            )}

            <Typography
              sx={{
                fontSize: 13,
                color: completed
                ? "#000"
                : running
                  ? "#10a37f"
                  : "#999",
              fontWeight: running ? 600 : 400
              }}
            >
              {step.label}
            </Typography>

          </Box>

        )

      })}

      {loading && (
        <LinearProgress
  sx={{
    mt: 2,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#ececec",
    "& .MuiLinearProgress-bar": {
      backgroundColor: "#10a37f"
    }
  }}
/>
      )}

      {timings && (

        <Box mt={2}>

          <Typography
            sx={{
              fontSize: 12,
              color: "#888",
              mb: 0.5
            }}
          >
            Pipeline timings
          </Typography>

          {Object.entries(timings)
            .filter(([k]) => k !== "total_s")
            .map(([k, v]) => (

              <Typography
                key={k}
                sx={{ fontSize: 12, color: "#666" }}
              >
                {k.replace("_s", "")}: {v}s
              </Typography>

            ))}

        </Box>

      )}

    </Box>

  )

}