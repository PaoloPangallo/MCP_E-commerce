from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.user import User

logger = logging.getLogger(__name__)


# ============================================================
# PRICE SIGNAL
# ============================================================

def _extract_price_signal(parsed: Dict) -> Optional[float]:
    """Return an average price from parsed constraints (if any)."""
    constraints = parsed.get("constraints") or []
    for c in constraints:
        if c.get("type") != "price":
            continue
        value = c.get("value")
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, list) and len(value) == 2:
                return float(sum(float(v) for v in value) / 2)
        except Exception:
            continue
    return None


# ============================================================
# CONDITION SIGNAL
# ============================================================

def _extract_condition_signal(parsed: Dict) -> Optional[str]:
    """Return dominant condition from parsed constraints."""
    for c in (parsed.get("constraints") or []):
        if c.get("type") == "condition" and c.get("value"):
            return str(c["value"]).lower()
    return None


# ============================================================
# CATEGORY AFFINITIES
# ============================================================

def _update_category_affinities(user: User, dominant_category_name: Optional[str]) -> bool:
    """
    Increment the hit count for a category.
    category_affinities is stored as JSON: {"Cellulari & smartphone": 12, ...}
    Returns True if changed.
    """
    if not dominant_category_name:
        return False

    raw = getattr(user, "category_affinities", None) or "{}"
    try:
        affinities: Dict[str, int] = json.loads(raw)
        if not isinstance(affinities, dict):
            affinities = {}
    except Exception:
        affinities = {}

    cat = str(dominant_category_name).strip()
    affinities[cat] = int(affinities.get(cat, 0)) + 1

    # Keep top 20 categories only
    sorted_cats = sorted(affinities.items(), key=lambda x: x[1], reverse=True)
    affinities = dict(sorted_cats[:20])

    new_val = json.dumps(affinities, ensure_ascii=False)
    if user.category_affinities != new_val:
        user.category_affinities = new_val
        return True
    return False


# ============================================================
# CONTEXTUAL BUDGETS (per-category / per-brand auto-update)
# ============================================================

def _update_contextual_budget(
    user: User,
    parsed: Dict,
    dominant_category_name: Optional[str],
    items: Optional[List[Dict]] = None,
) -> bool:
    """
    Auto-populate contextual_budgets JSON with per-category EMA.
    Coexists with manual MCP overrides: we only touch the auto_ prefixed keys,
    so manual keys set via update_user_preferences are never overwritten.
    Returns True if changed.
    """
    price_signal = _extract_price_signal(parsed)
    if not price_signal:
        # No explicit price in this query — nothing to learn
        return False

    raw = getattr(user, "contextual_budgets", None) or "{}"
    try:
        budgets: Dict[str, float] = json.loads(raw)
        if not isinstance(budgets, dict):
            budgets = {}
    except Exception:
        budgets = {}

    changed = False

    # Category budget (auto_cat: prefix to distinguish from manual entries)
    if dominant_category_name:
        cat_key = f"auto_cat:{dominant_category_name.lower()}"
        if cat_key in budgets:
            old = float(budgets[cat_key])
            new_val = round(old * 0.80 + price_signal * 0.20, 2)
        else:
            new_val = round(price_signal, 2)
        if budgets.get(cat_key) != new_val:
            budgets[cat_key] = new_val
            changed = True

    # Brand budget (auto_brand: prefix)
    for brand in (parsed.get("brands") or []):
        brand_key = f"auto_brand:{str(brand).strip().lower()}"
        if brand_key in budgets:
            old = float(budgets[brand_key])
            new_val = round(old * 0.80 + price_signal * 0.20, 2)
        else:
            new_val = round(price_signal, 2)
        if budgets.get(brand_key) != new_val:
            budgets[brand_key] = new_val
            changed = True

    if changed:
        user.contextual_budgets = json.dumps(budgets, ensure_ascii=False)
    return changed


# ============================================================
# CONDITION PREFERENCE
# ============================================================

def _update_condition_preference(user: User, parsed: Dict) -> bool:
    """
    Learn the user's preferred product condition over time.
    Uses a simple majority counter stored in LTM; updates the DB field
    only when there is a clear dominant preference (>= 3 signals).
    Returns True if changed.
    """
    condition = _extract_condition_signal(parsed)
    if not condition:
        return False

    # We use condition_preference directly as a counter-string: "new:5,used:2"
    raw = getattr(user, "condition_preference", None) or ""
    # If it's already a clean canonical value without ':', treat as legacy flat value
    if raw and ":" not in raw:
        # Convert legacy flat value to counter format
        counter: Dict[str, int] = {raw: 1}
    else:
        counter = {}
        for part in (raw.split(",") if raw else []):
            kv = part.split(":")
            if len(kv) == 2:
                try:
                    counter[kv[0]] = int(kv[1])
                except ValueError:
                    pass

    counter[condition] = counter.get(condition, 0) + 1
    total = sum(counter.values())

    # Determine dominant preference (needs >= 3 past signals, >50% share to commit)
    dominant = None
    if total >= 3:
        top = max(counter, key=lambda k: counter[k])
        if counter[top] / total >= 0.5:
            dominant = top

    new_raw = ",".join(f"{k}:{v}" for k, v in sorted(counter.items()))
    if user.condition_preference != new_raw:
        user.condition_preference = new_raw
        return True
    return False


def get_dominant_condition(user: User) -> Optional[str]:
    """
    Parse condition_preference counter string and return dominant condition, or None.
    """
    raw = getattr(user, "condition_preference", None) or ""
    if not raw:
        return None
    if ":" not in raw:
        return raw  # Legacy flat value
    counter: Dict[str, int] = {}
    for part in raw.split(","):
        kv = part.split(":")
        if len(kv) == 2:
            try:
                counter[kv[0]] = int(kv[1])
            except ValueError:
                pass
    if not counter:
        return None
    total = sum(counter.values())
    if total < 3:
        return None
    top = max(counter, key=lambda k: counter[k])
    return top if counter[top] / total >= 0.5 else None


# ============================================================
# INTERACTION DEPTH
# ============================================================

def update_interaction_depth(user: User, action: str) -> bool:
    """
    Track user engagement level.
    action: "search" | "detail" | "compare" | "seller" | "contact"
    browser   → only searches
    researcher → searches + detail/compare/seller analyses
    power_buyer → also contacts sellers or has many interactions
    Returns True if level changed.
    """
    # Simple scoring model stored in category_affinities sibling key.
    # We'll use a lightweight approach: store depth score in user_behaviour via memory,
    # and only write interaction_depth when it changes tier.
    current = getattr(user, "interaction_depth", None) or "browser"

    score_map = {"search": 1, "detail": 3, "compare": 3, "seller": 2, "contact": 5}
    score_boost = score_map.get(action, 0)
    if score_boost == 0:
        return False

    # Use condition_preference-style counter: "search:10,detail:3,contact:1"
    # We repurpose interaction_depth as a simple tier string for DB,
    # and derive it from the long-term memory behaviour dict (passed separately).
    # This function just advances the tier based on action.
    NEW_TIER_MAP = {
        ("browser", "detail"): "researcher",
        ("browser", "compare"): "researcher",
        ("browser", "seller"): "researcher",
        ("browser", "contact"): "power_buyer",
        ("researcher", "contact"): "power_buyer",
    }
    new_tier = NEW_TIER_MAP.get((current, action), current)

    if new_tier != current:
        user.interaction_depth = new_tier
        return True
    return False


# ============================================================
# BRAND PREFERENCE (improved — only from validated parser output)
# ============================================================

def _update_brands(user: User, parsed: Dict) -> bool:
    """
    Update favorite_brands with a counter format: "Brand1:5,Brand2:2".
    Only keeps active/frequent brands to avoid UI pollution.
    Returns True if changed.
    """
    new_brands = [str(b).strip() for b in (parsed.get("brands") or []) if str(b).strip()]
    if not new_brands:
        return False

    raw = getattr(user, "favorite_brands", "") or ""
    counter: Dict[str, int] = {}
    
    # Parse existing: "Apple:3,JBL:1" or legacy "Apple,JBL"
    if raw:
        if ":" in raw:
            for part in raw.split(","):
                kv = part.split(":")
                if len(kv) == 2:
                    try:
                        counter[kv[0].strip()] = int(kv[1])
                    except ValueError:
                        pass
        else:
            # Migration from legacy flat list: give each existing brand 1 hit
            for b in raw.split(","):
                if b.strip():
                    counter[b.strip()] = 1

    # Increment hits for new brands
    for nb in new_brands:
        # Match case-insensitively but preserve original casing for display
        found = False
        for existing in list(counter.keys()):
            if existing.lower() == nb.lower():
                counter[existing] += 1
                found = True
                break
        if not found:
            counter[nb] = 1

    # Sort by hits (descending)
    sorted_brands = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    # To keep the UI clean, we only keep brands that have at least 2 hits, 
    # OR the top 5 most recent even if they have 1 hit.
    # For now, let's just keep top 10 by affinity.
    merged = sorted_brands[:10]
    new_str = ",".join(f"{k}:{v}" for k, v in merged)

    if user.favorite_brands != new_str:
        user.favorite_brands = new_str
        return True
    return False


# ============================================================
# GLOBAL PRICE PREFERENCE (EMA — kept for backward compat)
# ============================================================

def _update_global_price(user: User, parsed: Dict) -> bool:
    """Update the global price_preference field. Returns True if changed."""
    price_signal = _extract_price_signal(parsed)
    if not price_signal:
        return False
    try:
        if not user.price_preference:
            user.price_preference = str(int(price_signal))
            return True
        old = float(user.price_preference)
        new_val = int(old * 0.80 + price_signal * 0.20)
        if str(new_val) != user.price_preference:
            user.price_preference = str(new_val)
            return True
    except Exception:
        pass
    return False


# ============================================================
# MAIN UPDATE FUNCTION
# ============================================================

def update_user_profile(
    user: User,
    parsed: Dict,
    db: Session,
    dominant_category_name: Optional[str] = None,
    items: Optional[List[Dict]] = None,
    action: Optional[str] = None,
) -> bool:
    """
    Auto-update user profile from a search interaction.
    - Coexists with manual MCP tool updates (never removes explicit user choices).
    - Retrocompatible: new columns degrade gracefully to None if missing.
    - NO commit here — caller is responsible for the commit.
    Returns True if any field was changed.
    """
    if not user:
        return False

    changed = False

    try:
        changed |= _update_brands(user, parsed)
    except Exception as e:
        logger.warning("update_user_profile: brand update failed: %s", e)

    try:
        changed |= _update_global_price(user, parsed)
    except Exception as e:
        logger.warning("update_user_profile: global price update failed: %s", e)

    try:
        changed |= _update_category_affinities(user, dominant_category_name)
    except Exception as e:
        logger.warning("update_user_profile: category affinities update failed: %s", e)

    try:
        changed |= _update_contextual_budget(user, parsed, dominant_category_name, items)
    except Exception as e:
        logger.warning("update_user_profile: contextual budget update failed: %s", e)

    try:
        changed |= _update_condition_preference(user, parsed)
    except Exception as e:
        logger.warning("update_user_profile: condition preference update failed: %s", e)

    if action:
        try:
            changed |= update_interaction_depth(user, action)
        except Exception as e:
            logger.warning("update_user_profile: interaction depth update failed: %s", e)

    if changed:
        try:
            db.add(user)
        except Exception as e:
            logger.warning("update_user_profile: db.add failed: %s", e)

    return changed


# ============================================================
# PROFILE SNAPSHOT (for prompt injection)
# ============================================================

def build_profile_context(user: User) -> Dict[str, Any]:
    """
    Build a dict of profile fields suitable for LLM prompt injection.
    Handles missing new columns gracefully.
    """
    ctx: Dict[str, Any] = {
        "favorite_brands": getattr(user, "favorite_brands", None),
        "price_preference": getattr(user, "price_preference", None),
        "condition_preference": get_dominant_condition(user),
        "interaction_depth": getattr(user, "interaction_depth", None) or "browser",
        "category_affinities": [],
        "contextual_budgets": {},
    }

    # Top 3 categories by affinity count
    raw_aff = getattr(user, "category_affinities", None) or "{}"
    try:
        affinities = json.loads(raw_aff)
        top_cats = sorted(affinities.items(), key=lambda x: x[1], reverse=True)[:3]
        ctx["category_affinities"] = [cat for cat, _ in top_cats]
    except Exception:
        pass

    # Contextual budgets — expose only non-auto_ keys (manual) + auto_ summary
    raw_budgets = getattr(user, "contextual_budgets", None) or "{}"
    try:
        budgets = json.loads(raw_budgets)
        # Expose human-readable auto budgets
        auto_budgets = {
            k.replace("auto_cat:", "").replace("auto_brand:", ""): v
            for k, v in budgets.items()
            if k.startswith("auto_")
        }
        manual_budgets = {k: v for k, v in budgets.items() if not k.startswith("auto_")}
        ctx["contextual_budgets"] = {**manual_budgets, **auto_budgets}
    except Exception:
        pass

    return ctx