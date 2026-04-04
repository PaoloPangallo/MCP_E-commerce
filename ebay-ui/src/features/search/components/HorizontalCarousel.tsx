import React, { useRef, useState, useEffect, useCallback } from "react"
import { Box, IconButton, Typography } from "@mui/material"
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft"
import ChevronRightIcon from "@mui/icons-material/ChevronRight"

interface CarouselProps {
  children: React.ReactNode
  title?: string
  count?: number
}

export default function HorizontalCarousel({ children, title}: CarouselProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showLeft, setShowLeft] = useState(false)
  const [showRight, setShowRight] = useState(true)

  const checkScroll = useCallback(() => {
    if (scrollRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current
      setShowLeft(scrollLeft > 5)
      setShowRight(scrollLeft < scrollWidth - clientWidth - 5)
    }
  }, [])

  useEffect(() => {
    checkScroll()
    const current = scrollRef.current
    if (current) {
        current.addEventListener("scroll", checkScroll)
        window.addEventListener("resize", checkScroll)
    }
    return () => {
        if (current) current.removeEventListener("scroll", checkScroll)
        window.removeEventListener("resize", checkScroll)
    }
  }, [checkScroll, children])

  const scroll = (dir: "left" | "right") => {
    if (scrollRef.current) {
      const amount = scrollRef.current.clientWidth * 0.8
      scrollRef.current.scrollBy({
        left: dir === "left" ? -amount : amount,
        behavior: "smooth"
      })
      setTimeout(checkScroll, 350)
    }
  }

  return (
    <Box sx={{ position: "relative", width: "100%", my: 3, px: { xs: 2.5, md: 12 } }}>
      {(title) && (
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2, px: 2 }}>
          <Typography sx={{ fontSize: 13, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            {title}
          </Typography>
        </Box>
      )}

      {showLeft && (
        <IconButton
          onClick={() => scroll("left")}
          sx={{
            position: "absolute",
            left: 20,
            top: "50%",
            transform: "translateY(-50%)",
            zIndex: 10,
            bgcolor: "var(--bg-primary)",
            border: "1px solid var(--border-color)",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            color: "var(--brand-primary)",
            "&:hover": { bgcolor: "var(--bg-secondary)" }
          }}
        >
          <ChevronLeftIcon />
        </IconButton>
      )}

      <Box sx={{ position: "relative" }}>
        <Box
          ref={scrollRef}
          sx={{
            display: "flex",
            gap: 2.5,
            overflowX: "auto",
            scrollbarWidth: "none",
            "&::-webkit-scrollbar": { display: "none" },
            px: 2,
            py: 1,
            scrollSnapType: "x proximity"
          }}
        >
          {children}
        </Box>
      </Box>

      {showRight && (
        <IconButton
          onClick={() => scroll("right")}
          sx={{
            position: "absolute",
            right: 20,
            top: "50%",
            transform: "translateY(-50%)",
            zIndex: 10,
            bgcolor: "var(--bg-primary)",
            border: "1px solid var(--border-color)",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            color: "var(--brand-primary)",
            "&:hover": { bgcolor: "var(--bg-secondary)" }
          }}
        >
          <ChevronRightIcon />
        </IconButton>
      )}
    </Box>
  )
}
