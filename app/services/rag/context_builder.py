from typing import List, Dict


def build_context(query: str, products: List[Dict], docs: List[Dict]) -> str:
    lines: List[str] = []

    query = (query or "").strip()
    lines.append(f"User query: {query}")
    lines.append("")

    if products:
        lines.append("Top products:")

        for p in products[:5]:
            title = p.get("title")
            price = p.get("price")
            currency = p.get("currency")
            seller = p.get("seller_name")
            trust = p.get("trust_score")

            # score: prefer rerank, fallback ranking_score
            score = p.get("_rerank_score")
            if score is None:
                score = p.get("ranking_score")

            exp = p.get("explanations") or []
            exp_txt = "; ".join(exp[:3]) if exp else None

            row = f"- {title} | {price} {currency} | seller: {seller} | trust: {trust}"
            if score is not None:
                row += f" | score: {score}"
            if exp_txt:
                row += f" | why: {exp_txt}"

            lines.append(row)

    if docs:
        lines.append("")
        lines.append("Retrieved context (seller feedback / product docs):")

        for d in docs[:5]:
            seller = d.get("seller")
            text = d.get("text")

            # hybrid retrieval info
            sources = d.get("_sources") or ([d.get("_source")] if d.get("_source") else [])
            rrf = d.get("_rrf_score")

            # per doc score
            sim = d.get("_similarity")
            bm25 = d.get("_bm25_score")

            row = f"- ({seller}) {text}"

            if sources:
                row += f" | sources: {','.join([s for s in sources if s])}"
            if rrf is not None:
                row += f" | rrf: {round(float(rrf), 5)}"
            if sim is not None:
                row += f" | sim: {round(float(sim), 4)}"
            if bm25 is not None:
                row += f" | bm25: {round(float(bm25), 4)}"

            lines.append(row)

    return "\n".join(lines)