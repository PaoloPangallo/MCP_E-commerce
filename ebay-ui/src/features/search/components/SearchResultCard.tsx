import { useState } from "react"
import { Box, Button, Collapse, Link, Typography, Chip } from "@mui/material"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import SettingsSuggestIcon from '@mui/icons-material/SettingsSuggest';
import LocalShippingIcon from "@mui/icons-material/LocalShipping"
import MoreVertIcon from "@mui/icons-material/MoreVert"
import Inventory2Icon from "@mui/icons-material/Inventory2"
import PersonSearchIcon from "@mui/icons-material/PersonSearch"
import AnalyticsIcon from "@mui/icons-material/Analytics"
import { Menu, MenuItem, ListItemIcon, ListItemText, IconButton } from "@mui/material"
import ContactMailIcon from "@mui/icons-material/ContactMail"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked"
import CompareArrowsIcon from "@mui/icons-material/CompareArrows"

import SellerTrustGauge from "../../seller/component/SellerTrustGauge.tsx"
import SellerFeedbackPanel from "../../seller/component/SellerFeedbackPanel.tsx"
import SellerInfo from "../../seller/SellerInfo.tsx"
import ExplanationChips from "./ExplanationChips.tsx"
import { WishlistToggleButton } from "../../chat/WishlistPanel"
import type { SearchItem } from "../types"

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "—"
  const formatted = price.toLocaleString('it-IT', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${formatted} ${currency ?? ""}`.trim()
}


// ─── Main card ────────────────────────────────────────────────────────────────

export default function SearchResultCard({ 
  item, 
  variant = 'list',
  index = 0,
  selectable = false,
  isSelected = false,
  onSelect
}: { 
  item: SearchItem, 
  variant?: 'list' | 'compact',
  index?: number,
  selectable?: boolean,
  isSelected?: boolean,
  onSelect?: (id: string) => void
}) {
  const [imageError, setImageError] = useState(false)
  const [sellerOpen, setSellerOpen] = useState(false)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const menuOpen = Boolean(anchorEl)

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    event.stopPropagation()
    setAnchorEl(event.currentTarget)
  }

  const handleMenuClose = (event?: any) => {
    if (event && event.stopPropagation) event.stopPropagation()
    setAnchorEl(null)
  }

  const triggerChat = (prompt: string, event?: React.MouseEvent<HTMLElement>) => {
    if (event) event.stopPropagation()
    handleMenuClose()
    window.dispatchEvent(new CustomEvent("send-chat", { detail: prompt }))
  }

  const handleContactSeller = (event: React.MouseEvent<HTMLElement>) => {
    event.stopPropagation();
    handleMenuClose();
    
    // eBay contact link format: 
    // https://contact.ebay.it/ws/eBayISAPI.dll?ContactUserNextGen&recipient=USERNAME&itemID=ITEMID
    const seller = item.seller_name || "";
    // ItemID extraction logic (use middle part for v1|...|... format if available)
    let finalId = item.ebay_id;
    if (item.ebay_id?.includes("|")) {
      const parts = item.ebay_id.split("|");
      if (parts.length >= 2) finalId = parts[1];
    }
    
    const contactUrl = `https://www.ebay.it/cnt/IntermediatedFAQ?seller_name=${seller}&item_id=${finalId}`;
    window.open(contactUrl, "_blank");
  };

  const rankingPct = typeof item.ranking_score === "number" ? item.ranking_score : null
  const valuePct = (item as any).value_score ?? 0

  const isCompact = variant === 'compact'


  if (isCompact) {
    return (
      <Box
        sx={{
          width: 280,
          flexShrink: 0,
          borderRadius: "20px",
          bgcolor: "var(--bg-primary)",
          border: "1px solid",
          borderColor: isSelected ? "var(--brand-primary)" : "var(--border-color)",
          boxShadow: isSelected ? "0 0 0 2px var(--brand-primary), 0 12px 24px rgba(0,0,0,0.12)" : "0 4px 12px rgba(0,0,0,0.03)",
          p: 2,
          display: "flex",
          flexDirection: "column",
          gap: 2,
          "&:hover": {
            transform: "translateY(-4px)",
            boxShadow: isSelected ? "0 0 0 2px var(--brand-primary), 0 16px 32px rgba(0,0,0,0.18)" : "0 12px 24px rgba(0,0,0,0.08)",
            borderColor: "var(--brand-primary)"
          },
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          cursor: "pointer",
          scrollSnapAlign: "start",
          position: "relative"
        }}
        onClick={() => {
          if (selectable && onSelect) {
            onSelect(item.ebay_id || "");
          } else if (item.url) {
            window.open(item.url, '_blank');
          }
        }}
      >
        <Box sx={{ position: "relative", width: "100%", aspectRatio: "1/1", borderRadius: "12px", overflow: "hidden", bgcolor: "var(--bg-secondary)", display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-color)' }}>
          {!imageError && item.image_url ? (
            <Box
              component="img"
              src={item.image_url}
              alt={item.title || ""}
              onError={() => setImageError(true)}
              sx={{ width: "100%", height: "100%", objectFit: "contain", p: 2, mixBlendMode: 'multiply' }}
            />
          ) : (
              <Typography variant="caption" color="text.disabled">No Image</Typography>
          )}

          {/* Wishlist Toggle Button */}
          <Box sx={{ position: "absolute", top: 8, right: 44, zIndex: 10 }}>
            <WishlistToggleButton
              ebayId={item.ebay_id || `search-${index}`}
              title={item.title}
              price={item.price}
              currency={item.currency || "EUR"}
              imageUrl={item.image_url}
              url={item.url}
              sellerName={item.seller_name}
            />
          </Box>

          {/* Selection Overlay */}
          {selectable && (
             <Box 
               sx={{ 
                 position: "absolute", 
                 top: 8, 
                 left: 8, 
                 zIndex: 20,
                 bgcolor: isSelected ? "var(--brand-primary)" : "rgba(255,255,255,0.9)",
                 color: isSelected ? "#fff" : "var(--text-secondary)",
                 borderRadius: "50%",
                 width: 28,
                 height: 28,
                 display: "flex",
                 alignItems: "center",
                 justifyContent: "center",
                 boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
                 border: `2px solid ${isSelected ? "#fff" : "transparent"}`,
                 transition: "all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
               }}
               onClick={(e) => {
                 e.stopPropagation();
                 onSelect?.(item.ebay_id || "");
               }}
             >
               {isSelected ? <CheckCircleIcon sx={{ fontSize: 18 }} /> : <RadioButtonUncheckedIcon sx={{ fontSize: 18 }} />}
             </Box>
          )}

          <IconButton
            size="small"
            onClick={handleMenuClick}
            sx={{
              position: "absolute",
              top: 8,
              right: 8,
              bgcolor: "rgba(255, 255, 255, 0.9)",
              color: "var(--text-primary)",
              boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
              "&:hover": { bgcolor: "#fff", transform: "scale(1.05)" }
            }}
          >
            <MoreVertIcon sx={{ fontSize: 18 }} />
          </IconButton>
          
          <Menu
            anchorEl={anchorEl}
            open={menuOpen}
            onClose={handleMenuClose}
            onClick={(e) => e.stopPropagation()}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            PaperProps={{
              sx: {
                mt: 1,
                boxShadow: "0 10px 40px -10px rgba(0,0,0,0.15)",
                border: "1px solid var(--border-color)",
                borderRadius: "12px",
                minWidth: 200
              }
            }}
          >
            <MenuItem onClick={(e) => triggerChat(`Dettagli prodotto per ${item.title} (ID: ${item.ebay_id})`, e)} sx={{ py: 1.5, px: 2 }}>
              <ListItemIcon sx={{ minWidth: 32 }}><Inventory2Icon sx={{ fontSize: 18, color: "var(--brand-primary)" }} /></ListItemIcon>
              <ListItemText primary="Dettagli Prodotto" primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }} />
            </MenuItem>
            
            {item.seller_name && (
              <MenuItem onClick={(e) => triggerChat(`Verifica affidabilità del venditore ${item.seller_name}`, e)} sx={{ py: 1.5, px: 2 }}>
                <ListItemIcon sx={{ minWidth: 32 }}><PersonSearchIcon sx={{ fontSize: 18, color: "var(--brand-primary)" }} /></ListItemIcon>
                <ListItemText primary="Analisi Venditore" primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }} />
              </MenuItem>
            )}

            <MenuItem onClick={handleContactSeller} sx={{ py: 1.5, px: 2 }}>
              <ListItemIcon sx={{ minWidth: 32 }}><ContactMailIcon sx={{ fontSize: 18, color: "var(--brand-primary)" }} /></ListItemIcon>
              <ListItemText primary="Contatta Venditore" primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }} />
            </MenuItem>

            <MenuItem onClick={(e) => triggerChat(`Analisi di mercato e storico prezzi per ${item.title}`, e)} sx={{ py: 1.5, px: 2, borderTop: "1px solid var(--border-color)" }}>
              <ListItemIcon sx={{ minWidth: 32 }}><AnalyticsIcon sx={{ fontSize: 18, color: "var(--brand-primary)" }} /></ListItemIcon>
              <ListItemText primary="Analisi Mercato" primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }} />
            </MenuItem>
          </Menu>

          {!selectable && index < 2 && rankingPct && rankingPct > 0.6 && (
            <Chip
              label="BEST CHOICE"
              size="small"
              sx={{
                position: "absolute",
                bottom: 8,
                left: 8,
                bgcolor: "var(--brand-primary)",
                color: "#ffffff",
                fontWeight: 900,
                fontSize: 8,
                height: 18,
                letterSpacing: '0.05em'
              }}
            />
          )}
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
          <Typography
            sx={{
              fontSize: 14,
              fontWeight: 700,
              color: "var(--text-primary)",
              lineHeight: 1.3,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
              minHeight: "2.6em",
              letterSpacing: '-0.01em'
            }}
          >
            {item.title}
          </Typography>

          <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
            <Typography sx={{ fontSize: 18, fontWeight: 800, color: "var(--text-primary)" }}>
              {formatPrice(item.price, item.currency)}
            </Typography>
            {item.condition && (
               <Typography sx={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", textTransform: 'uppercase', letterSpacing: '0.02em' }}>
                 · {item.condition}
               </Typography>
            )}
          </Box>
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
          <Box sx={{ width: "100%" }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
              <Typography sx={{ fontSize: 9, fontWeight: 800, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: '0.05em' }}>Affinità</Typography>
              <Typography sx={{ fontSize: 9, fontWeight: 900, color: "var(--brand-primary)" }}>{Math.round((rankingPct || 0) * 100)}%</Typography>
            </Box>
            <Box sx={{ height: 2, bgcolor: "var(--bg-secondary)", borderRadius: 1, overflow: "hidden" }}>
              <Box sx={{ width: `${Math.round((rankingPct || 0) * 100)}%`, height: "100%", bgcolor: "var(--brand-primary)" }} />
            </Box>
          </Box>
          
          <Box sx={{ width: "100%" }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
              <Typography sx={{ fontSize: 9, fontWeight: 800, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: '0.05em' }}>Valore</Typography>
              <Typography sx={{ fontSize: 9, fontWeight: 900, color: "var(--success)" }}>{Math.round(valuePct * 100)}%</Typography>
            </Box>
            <Box sx={{ height: 2, bgcolor: "var(--bg-secondary)", borderRadius: 1, overflow: "hidden" }}>
              <Box sx={{ width: `${Math.round(valuePct * 100)}%`, height: "100%", bgcolor: "var(--success)" }} />
            </Box>
          </Box>
        </Box>

        {item.shipping && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
            <LocalShippingIcon sx={{ fontSize: 14, color: "var(--text-secondary)", opacity: 0.6 }} />
            <Typography sx={{ fontSize: 10, fontWeight: 700, color: item.shipping.free ? "var(--success)" : "var(--text-primary)", letterSpacing: '0.02em' }}>
              {item.shipping.free ? "CONSEGNA GRATUITA" : `SPEDIZIONE: ${formatPrice(item.shipping.cost, item.shipping.currency || '€')}`}
            </Typography>
          </Box>
        )}

        <Box sx={{ display: "flex", gap: 1, mt: "auto", pt: 1 }}>
          <Button
            fullWidth
            component="a"
            variant="contained"
            disableElevation
            href={item.url || "#"}
            target="_blank"
            rel="noreferrer"
            sx={{
              textTransform: "none",
              borderRadius: "10px",
              fontSize: 12,
              fontWeight: 800,
              bgcolor: "var(--brand-primary)",
              color: "#fff",
              py: 1,
              "& .MuiButton-root": { color: "#fff" },
              "&:hover": { bgcolor: "var(--brand-primary)", opacity: 0.9 }
            }}
          >
            Acquista
          </Button>
          <IconButton 
            size="small"
            sx={{ 
                borderRadius: "10px", 
                border: "1px solid var(--border-color)",
                color: "var(--text-secondary)",
                px: 1.5
            }}
            onClick={() => item.url && window.open(item.url, '_blank')}
          >
            <OpenInNewIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Box>
      </Box>
    )
  }

  // Original List View
  const ragPreviews = Array.isArray(item.rag_feedback)
    ? item.rag_feedback.map((fb) => fb?.comment || "").filter(Boolean).slice(0, 2)
    : []

  const specs = item.ner_attributes?.specs || {};
  const hasNer = !!(item.ner_attributes?.brand || item.ner_attributes?.model || Object.keys(specs).length > 0);

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "flex-start",
        gap: 2,
        py: 2.5,
        borderBottom: "1px solid var(--border-color)",
        "&:last-child": { borderBottom: "none" },
        "&:first-of-type": { pt: 0.5 },
        position: "relative",
        transition: "all 0.2s ease",
        px: selectable ? 2 : 0,
        mx: selectable ? -2 : 0,
        borderRadius: "12px",
        bgcolor: isSelected ? "rgba(0, 100, 210, 0.04)" : "transparent",
        "&:hover": {
          bgcolor: isSelected ? "rgba(0, 100, 210, 0.08)" : "rgba(0, 0, 0, 0.02)"
        }
      }}
      onClick={() => {
        if (selectable && onSelect) {
          onSelect(item.ebay_id || "");
        }
      }}
    >
      {/* Selection Indicator for List View */}
      {selectable && (
        <Box 
          sx={{ 
            position: "absolute", 
            top: 12, 
            left: 12, 
            zIndex: 20,
            bgcolor: isSelected ? "var(--brand-primary)" : "rgba(255,255,255,0.9)",
            color: isSelected ? "#fff" : "var(--text-secondary)",
            borderRadius: "50%",
            width: 24,
            height: 24,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
            border: `2px solid ${isSelected ? "#fff" : "transparent"}`,
          }}
        >
          {isSelected ? <CheckCircleIcon sx={{ fontSize: 16 }} /> : <RadioButtonUncheckedIcon sx={{ fontSize: 16 }} />}
        </Box>
      )}

      <Box
        component={item.url ? "a" : "div"}
        href={item.url}
        target="_blank"
        rel="noreferrer"
        sx={{
          width: 88,
          height: 88,
          borderRadius: "12px",
          overflow: "hidden",
          bgcolor: "var(--bg-primary)",
          border: "1px solid var(--border-color)",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          textDecoration: "none",
          transition: "all 0.2s ease-in-out",
          boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
          "&:hover": { 
            transform: "scale(1.02)",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            borderColor: "var(--brand-primary)"
          },
        }}
      >
        {!imageError && item.image_url ? (
          <Box
            component="img"
            src={item.image_url}
            alt={item.title || ""}
            loading="lazy"
            onError={() => setImageError(true)}
            sx={{ width: "100%", height: "100%", objectFit: "contain", p: 0.5 }}
          />
        ) : (
          <Box sx={{ width: 32, height: 32, bgcolor: "var(--bg-secondary)", borderRadius: "8px", display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
             <Typography variant="caption" color="text.disabled">No img</Typography>
          </Box>
        )}
        
        {/* Wishlist Toggle Button (Overlay) */}
        <Box sx={{ position: "absolute", top: 4, right: 4, zIndex: 5 }}>
          <WishlistToggleButton
            ebayId={item.ebay_id || `list-${index}`}
            title={item.title}
            price={item.price}
            currency={item.currency || "EUR"}
            imageUrl={item.image_url}
            url={item.url}
            sellerName={item.seller_name}
            size="small"
          />
        </Box>
      </Box>

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", mb: 0.5 }}>
          <Box sx={{ display: "flex", alignItems: "flex-start", gap: 0.5 }}>
            <Link
              href={item.url}
              target="_blank"
              rel="noreferrer"
              underline="none"
              sx={{
                fontSize: 15,
                fontWeight: 600,
                color: "var(--text-primary)",
                lineHeight: 1.3,
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                "&:hover": { color: "var(--brand-primary)" },
              }}
            >
              {item.title || "Titolo non disponibile"}
            </Link>
            {item.url && <OpenInNewIcon sx={{ fontSize: 13, color: "#9ca3af", flexShrink: 0, mt: 0.4 }} />}
          </Box>
        </Box>

        {hasNer && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
            {(item.ner_attributes?.brand || item.ner_attributes?.model) && (
              <Box 
                sx={{ 
                  px: 1, py: 0.25, borderRadius: '4px', bgcolor: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                  display: 'flex', alignItems: 'center', gap: 0.5
                }}
              >
                <SettingsSuggestIcon sx={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                <Typography sx={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                  {item.ner_attributes?.brand} {item.ner_attributes?.model}
                </Typography>
              </Box>
            )}
          </Box>
        )}

        <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 1.5, mb: 1.5 }}>
          <Typography sx={{ fontSize: 18, fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
            {formatPrice(item.price, item.currency)}
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', alignItems: 'center' }}>
            {item.condition && (
              <Box
                sx={{
                  px: 1, py: 0.25, borderRadius: "4px",
                  bgcolor: (item.condition || "").toLowerCase().includes("nuovo") ? "var(--brand-soft)" : "var(--bg-secondary)",
                  border: "1px solid",
                  borderColor: (item.condition || "").toLowerCase().includes("nuovo") ? "var(--brand-primary)" : "var(--border-color)",
                  opacity: (item.condition || "").toLowerCase().includes("nuovo") ? 1 : 0.8
                }}
              >
                <Typography sx={{ fontSize: 10, fontWeight: 700, color: (item.condition || "").toLowerCase().includes("nuovo") ? "var(--brand-primary)" : "var(--text-secondary)", textTransform: 'uppercase' }}>
                  {item.condition}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>

        <Box sx={{ mb: 0.5 }}>
          <SellerInfo seller_name={item.seller_name} seller_rating={item.seller_rating} />
        </Box>

        {typeof item.trust_score === "number" && (
          <Box sx={{ mb: 0.6 }}>
            <SellerTrustGauge score={item.trust_score} />
          </Box>
        )}

        {item.explanations?.length ? (
          <Box sx={{ mb: 0.6 }}>
            <ExplanationChips explanations={item.explanations} />
          </Box>
        ) : null}

        {ragPreviews.length > 0 && (
          <Box sx={{ mb: 1, p: 0.5 }}>
            {ragPreviews.map((text, i) => (
              <Typography key={i} sx={{ fontSize: 11.5, color: "var(--text-secondary)", fontStyle: "italic", lineHeight: 1.55 }}>
                "{text}"
              </Typography>
            ))}
          </Box>
        )}

        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1 }}>
          {item.url && (
            <Button
              size="small"
              variant="contained"
              disableElevation
              href={item.url}
              target="_blank"
              rel="noreferrer"
              startIcon={<ShoppingCartIcon sx={{ fontSize: 14 }} />}
              sx={{
                textTransform: "none",
                borderRadius: "8px",
                fontSize: 12,
                fontWeight: 700,
                bgcolor: "#0064d2",
                color: "#fff",
                px: 2,
                "&:hover": { bgcolor: "#0053b3" }
              }}
            >
              Acquista
            </Button>
          )}
          <Button
            size="small"
            variant="contained"
            disableElevation
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("send-chat", {
                  detail: `Dettagli per ${item.title} (ID: ${item.ebay_id})`,
                })
              )
            }
            sx={{
              textTransform: "none",
              borderRadius: "8px",
              fontSize: 12,
              fontWeight: 600,
              bgcolor: "var(--brand-primary)",
              color: "#fff",
              px: 2,
              "&:hover": { bgcolor: "var(--brand-primary)", opacity: 0.9 }
            }}
          >
            Dettagli
          </Button>

          {item.seller_name && (
            <Button
              size="small"
              variant="text"
              onClick={() => setSellerOpen((v) => !v)}
              endIcon={
                <KeyboardArrowDownIcon
                  sx={{
                    fontSize: 14,
                    transform: sellerOpen ? "rotate(180deg)" : "none",
                    transition: "transform 0.2s",
                  }}
                />
              }
              sx={{
                textTransform: "none",
                fontSize: 12,
                fontWeight: 600,
                color: "var(--text-secondary)",
                "&:hover": { color: "var(--text-primary)", bgcolor: "transparent" }
              }}
            >
              Seller
            </Button>
          )}
        </Box>

        {item.seller_name && (
          <Collapse in={sellerOpen} timeout={200} unmountOnExit>
            <Box sx={{ mt: 1 }}>
              <SellerFeedbackPanel seller={item.seller_name} />
            </Box>
          </Collapse>
        )}
      </Box>
    </Box>
  )
}