from typing import List, Dict, Optional


def _fmt(x) -> str:
    if x is None:
        return "-"
    return str(x)


def build_context(
    query: str,
    products: List[Dict],
    docs: List[Dict],
    max_products: int = 5,
    max_docs: int = 6
) -> str:
    lines: List[str] = []

    query = (query or "").strip()
    lines.append(f"User query: {query}")
    lines.append("")

    # -------------------------
    # PRODUCTS (post-rerank)
    # -------------------------
    if products:
        lines.append("Top products (ranked):")

        for p in products[:max_products]:
            title = p.get("title")
            price = p.get("price")
            currency = p.get("currency")
            seller = p.get("seller_name")
            trust = p.get("trust_score")

            score = p.get("_rerank_score")
            if score is None:
                score = p.get("ranking_score")

            reasons = p.get("explanations") or []
            why = "; ".join(reasons[:3]) if reasons else None

            row = (
                f"- {title} | {price} {currency} | seller: {seller} | trust: {_fmt(trust)}"
                f" | score: {_fmt(score)}"
            )
            if why:
                row += f" | why: {why}"

            lines.append(row)

    # -------------------------
    # RETRIEVED DOCS (hybrid)
    # -------------------------
    if docs:
        lines.append("")
        lines.append("Retrieved context (hybrid):")

        for d in docs[:max_docs]:
            seller = d.get("seller")
            text = d.get("text")

            doc_type = d.get("type")
            sources = d.get("_sources") or ([d.get("_source")] if d.get("_source") else [])
            rrf = d.get("_rrf_score")
            sim = d.get("_similarity")
            bm = d.get("_bm25_score")

            row = f"- [{_fmt(doc_type)}] ({_fmt(seller)}) {text}"

            if sources:
                row += f" | sources: {','.join([s for s in sources if s])}"
            if rrf is not None:
                row += f" | rrf: {round(float(rrf), 5)}"
            if sim is not None:
                row += f" | sim: {round(float(sim), 4)}"
            if bm is not None:
                row += f" | bm25: {round(float(bm), 4)}"

            lines.append(row)

    return "\n".join(lines)