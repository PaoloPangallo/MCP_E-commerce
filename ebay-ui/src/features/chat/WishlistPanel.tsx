import { Box, Typography, IconButton, Tooltip, List, ListItem, ListItemText, Fade, Chip } from "@mui/material"
import FavoriteIcon from "@mui/icons-material/Favorite"
import FavoriteBorderIcon from "@mui/icons-material/FavoriteBorder"
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline"
import CloseIcon from "@mui/icons-material/Close"
import TrendingDownIcon from "@mui/icons-material/TrendingDown"
import { useWishlistStore } from "./store/wishlistStore"
import { useAuth } from "../../auth/useAuth"

/** 
 * WishlistPanel - Premium shopping drawer variant
 */
export default function WishlistPanel({ onClose }: { onClose?: () => void }) {
  const { items, removeItem } = useWishlistStore()
  const { user } = useAuth()

  if (!user) return null

  return (
    <Box
      sx={{
        width: { xs: "100dvw", sm: 380 },
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "var(--bg-primary)",
        backdropFilter: "blur(20px) saturate(180%)",
        borderLeft: "1px solid var(--border-color)",
        boxShadow: "-10px 0 30px rgba(0,0,0,0.1)",
        position: "relative"
      }}
    >
      {/* Header with better spacing and glass effect */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 3,
          py: 3,
          borderBottom: "1px solid var(--border-color)",
          bgcolor: "var(--bg-secondary)",
          opacity: 0.8
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: "50%",
              bgcolor: "rgba(239, 68, 68, 0.1)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}
          >
            <FavoriteIcon sx={{ fontSize: 20, color: "#ef4444" }} />
          </Box>
          <Box>
            <Typography sx={{ fontSize: 16, fontWeight: 800, color: "var(--text-primary)", letterSpacing: -0.5 }}>
              I tuoi preferiti
            </Typography>
            <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", fontWeight: 500 }}>
              {items.length} {items.length === 1 ? 'oggetto salvato' : 'oggetti salvati'}
            </Typography>
          </Box>
        </Box>
        <IconButton 
          size="medium" 
          onClick={onClose} 
          sx={{ 
            color: "var(--text-secondary)",
            bgcolor: "var(--bg-secondary)",
            "&:hover": { bgcolor: "var(--bg-primary)", transform: "rotate(90deg)" },
            transition: "all 0.3s ease"
          }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Item List with Staggered Entrance */}
      <Box sx={{ flex: 1, overflowY: "auto", px: 1, py: 1 }}>
        {items.length === 0 ? (
          <Fade in timeout={800}>
            <Box sx={{ 
              height: "100%", 
              display: "flex", 
              flexDirection: "column", 
              alignItems: "center", 
              justifyContent: "center",
              p: 4, 
              textAlign: "center", 
              color: "var(--text-secondary)" 
            }}>
               <FavoriteBorderIcon sx={{ fontSize: 80, mb: 2, opacity: 0.1, color: "#ef4444" }} />
               <Typography sx={{ fontWeight: 600, color: "var(--text-primary)" }}>Nessun preferito salvato</Typography>
               <Typography variant="caption" sx={{ mt: 1, display: "block" }}>
                 Salva gli oggetti che ti piacciono per ritrovarli facilmente qui.
               </Typography>
            </Box>
          </Fade>
        ) : (
          <List disablePadding>
            {items.map((item, idx) => (
              <Fade key={item.id} in timeout={400 + idx * 100}>
                <ListItem
                  disablePadding
                  secondaryAction={
                    <Tooltip title="Rimuovi">
                      <IconButton
                        edge="end"
                        size="small"
                        onClick={() => removeItem(item.ebay_id)}
                        sx={{ 
                          color: "var(--text-secondary)", 
                          mr: 1,
                          "&:hover": { color: "#ef4444", bgcolor: "rgba(239, 68, 68, 0.1)" } 
                        }}
                      >
                        <DeleteOutlineIcon sx={{ fontSize: 20 }} />
                      </IconButton>
                    </Tooltip>
                  }
                  sx={{
                    px: 2,
                    py: 2,
                    my: 0.5,
                    borderRadius: 3,
                    border: "1px solid transparent",
                    "&:hover": { 
                      bgcolor: "var(--bg-secondary)", 
                      borderColor: "var(--border-color)",
                      boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                      transform: "translateY(-2px)"
                    },
                    transition: "all 0.2s ease",
                    alignItems: "flex-start"
                  }}
                >
                  <Box
                    component="img"
                    src={item.image_url ?? ""}
                    sx={{
                      width: 64,
                      height: 64,
                      borderRadius: 2,
                      objectFit: "cover",
                      mr: 2,
                      bgcolor: "var(--bg-secondary)",
                      display: item.image_url ? "block" : "none",
                      boxShadow: "0 2px 5px rgba(0,0,0,0.05)"
                    }}
                  />
                  <ListItemText
                    primaryTypographyProps={{ component: 'div' }}
                    secondaryTypographyProps={{ component: 'div' }}
                    primary={
                      <Typography
                        component="a"
                        href={item.url ?? "#"}
                        target="_blank"
                        rel="noopener noreferrer"
                        sx={{
                          fontSize: 13,
                          lineHeight: 1.4,
                          fontWeight: 700,
                          color: "var(--text-primary)",
                          textDecoration: "none",
                          display: "-webkit-box",
                          overflow: "hidden",
                          WebkitBoxOrient: "vertical",
                          WebkitLineClamp: 2,
                          mb: 0.5,
                          "&:hover": { color: "#3b82f6" },
                          transition: "color 0.2s"
                        }}
                      >
                        {item.title ?? item.ebay_id}
                      </Typography>
                    }
                    secondary={
                      <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
                        {item.price != null && (
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                            <Typography sx={{ fontSize: 14, color: "#059669", fontWeight: 800 }}>
                              {item.price.toLocaleString("it-IT", { style: "currency", currency: item.currency ?? "EUR" })}
                            </Typography>
                            {item.previous_price != null && item.previous_price > item.price && (
                              <Chip
                                icon={<TrendingDownIcon sx={{ fontSize: "13px !important", color: "#059669 !important" }} />}
                                label={`-${(item.previous_price - item.price).toLocaleString("it-IT", { style: "currency", currency: item.currency ?? "EUR" })}`}
                                size="small"
                                sx={{
                                  height: 20,
                                  fontSize: 10,
                                  fontWeight: 800,
                                  bgcolor: "rgba(5, 150, 105, 0.1)",
                                  color: "#059669",
                                  border: "1px solid rgba(5, 150, 105, 0.3)",
                                  ".MuiChip-icon": { ml: "4px" },
                                  animation: "pricePop 0.4s ease-out",
                                  "@keyframes pricePop": {
                                    "0%": { transform: "scale(0.8)", opacity: 0 },
                                    "70%": { transform: "scale(1.05)" },
                                    "100%": { transform: "scale(1)", opacity: 1 },
                                  }
                                }}
                              />
                            )}
                          </Box>
                        )}
                        {item.seller_name && (
                          <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", fontWeight: 500 }}>
                             {item.seller_name}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
              </Fade>
            ))}
          </List>
        )}
      </Box>

      {/* Footer Info */}
      {items.length > 0 && (
        <Box sx={{ p: 2.5, borderTop: "1px solid var(--border-color)", bgcolor: "var(--bg-secondary)", opacity: 0.8 }}>
           <Typography variant="caption" sx={{ color: "var(--text-secondary)", lineHeight: 1.4, display: "block" }}>
             * Prezzi e disponibilità visualizzati potrebbero subire variazioni su eBay.
           </Typography>
        </Box>
      )}
    </Box>
  )
}

/** Reusable heart icon button to toggle wishlist on product cards */
export function WishlistToggleButton({
  ebayId,
  title,
  price,
  currency,
  condition,
  imageUrl,
  url,
  sellerName,
  size = "small"
}: {
  ebayId: string
  title?: string
  price?: number
  currency?: string
  condition?: string
  imageUrl?: string
  url?: string
  sellerName?: string
  size?: "small" | "medium"
}) {
  const { isInWishlist, addItem, removeItem } = useWishlistStore()
  const { user } = useAuth()
  const saved = isInWishlist(ebayId)

  if (!user) return null

  const toggle = async (e: React.MouseEvent) => {
    e.stopPropagation()
    e.preventDefault()
    if (saved) {
      await removeItem(ebayId)
    } else {
      await addItem({
        ebay_id: ebayId,
        title: title ?? null,
        price: price ?? null,
        currency: currency ?? "EUR",
        condition: condition ?? null,
        image_url: imageUrl ?? null,
        url: url ?? null,
        seller_name: sellerName ?? null,
      })
    }
  }

  return (
    <Tooltip title={saved ? "Rimuovi dalla Wishlist" : "Salva in Wishlist"}>
      <IconButton
        size={size}
        onClick={toggle}
        sx={{
          color: saved ? "#ef4444" : "var(--text-secondary)",
          transition: "all 0.2s ease",
          "&:hover": { color: "#ef4444", transform: "scale(1.1)" },
          p: 0.5
        }}
      >
        {saved ? <FavoriteIcon sx={{ fontSize: size === "small" ? 16 : 20 }} /> : <FavoriteBorderIcon sx={{ fontSize: size === "small" ? 16 : 20 }} />}
      </IconButton>
    </Tooltip>
  )
}
