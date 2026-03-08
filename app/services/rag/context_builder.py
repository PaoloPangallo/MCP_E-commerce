from typing import List, Dict


def _fmt(x) -> str:
    if x is None:
        return "-"
    return str(x)


def _round(x, n=4):
    try:
        return round(float(x), n)
    except Exception:
        return x


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

    # =====================================================
    # PRODUCTS
    # =====================================================

    if products:

        lines.append("Top ranked products:")
        lines.append("")

        for rank, p in enumerate(products[:max_products], start=1):

            title = p.get("title")
            price = p.get("price")
            currency = p.get("currency")
            seller = p.get("seller_name") or p.get("seller_username")

            trust = p.get("trust_score")
            rating = p.get("seller_rating")

            final_score = p.get("_final_score")
            rerank_score = p.get("_rerank_score")
            cross_score = p.get("_cross_score")

            rag_prod = p.get("_rag_product_boost")
            rag_seller = p.get("_rag_seller_boost")
            rag_sent = p.get("_rag_sentiment_signal")

            row = (
                f"{rank}. {title} | {price} {currency}"
                f" | seller: {seller}"
                f" | trust: {_fmt(trust)}"
            )

            if rating:
                row += f" | rating: {_fmt(rating)}"

            if final_score is not None:
                row += f" | final_score: {_round(final_score)}"

            if rerank_score is not None:
                row += f" | base_score: {_round(rerank_score)}"

            if cross_score is not None:
                row += f" | cross: {_round(cross_score)}"

            lines.append(row)

            # -------------------------
            # RAG SIGNALS
            # -------------------------

            rag_signals = []

            if rag_prod and rag_prod > 0:
                rag_signals.append(f"product_rag={_round(rag_prod)}")

            if rag_seller and rag_seller > 0:
                rag_signals.append(f"seller_rag={_round(rag_seller)}")

            if rag_sent:
                rag_signals.append(f"seller_sentiment={_round(rag_sent)}")

            if rag_signals:
                lines.append(
                    f"   RAG signals: {', '.join(rag_signals)}"
                )

            # -------------------------
            # EXPLANATIONS
            # -------------------------

            reasons = p.get("explanations") or []

            if reasons:
                why = "; ".join(reasons[:3])
                lines.append(f"   Why: {why}")

            # -------------------------
            # SELLER FEEDBACK EVIDENCE
            # -------------------------

            feedback = p.get("rag_feedback") or []

            if feedback:

                lines.append("   Seller feedback evidence:")

                for f in feedback[:2]:

                    text = f.get("text")
                    rrf = f.get("rrf_score")

                    row = f"     - {text}"

                    if rrf:
                        row += f" | rrf={_round(rrf)}"

                    lines.append(row)

            # -------------------------
            # PRODUCT CONTEXT
            # -------------------------

            prod_ctx = p.get("rag_product_context") or []

            if prod_ctx:

                lines.append("   Product context evidence:")

                for c in prod_ctx[:2]:

                    text = c.get("text")
                    rrf = c.get("rrf_score")

                    row = f"     - {text}"

                    if rrf:
                        row += f" | rrf={_round(rrf)}"

                    lines.append(row)

            lines.append("")

    # =====================================================
    # RETRIEVED DOCUMENTS
    # =====================================================

    if docs:

        lines.append("")
        lines.append("Retrieved knowledge (hybrid RAG):")
        lines.append("")

        for d in docs[:max_docs]:

            seller = d.get("seller")
            text = d.get("text")

            doc_type = d.get("type")

            sources = d.get("_sources") or (
                [d.get("_source")] if d.get("_source") else []
            )

            rrf = d.get("_rrf_score")
            sim = d.get("_similarity")
            bm = d.get("_bm25_score")

            row = f"- [{_fmt(doc_type)}] ({_fmt(seller)}) {text}"

            if sources:
                row += f" | sources={','.join([s for s in sources if s])}"

            if rrf is not None:
                row += f" | rrf={_round(rrf,5)}"

            if sim is not None:
                row += f" | sim={_round(sim)}"

            if bm is not None:
                row += f" | bm25={_round(bm)}"

            lines.append(row)

    return "\n".join(lines)