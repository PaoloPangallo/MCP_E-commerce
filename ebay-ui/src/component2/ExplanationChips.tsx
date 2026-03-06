import { Box, Chip } from "@mui/material"

interface Props {
  explanations?: string[]
}

export default function ExplanationChips({ explanations = [] }: Props) {
  const cleanExplanations = explanations
    .map(item => item?.trim())
    .filter(Boolean) as string[]

  if (cleanExplanations.length === 0) {
    return null
  }

  return (
    <Box display="flex" gap={1} flexWrap="wrap" mb={2}>
      {cleanExplanations.map(item => (
        <Chip
          key={item}
          label={item}
          size="small"
          sx={{
            backgroundColor: "#fafafa",
            border: "1px solid #e5e5e5",
            fontSize: 11
          }}
        />
      ))}
    </Box>
  )
}
