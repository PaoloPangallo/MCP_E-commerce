from typing import List, Dict


def explain_results(query: str, items: List[Dict]) -> str:

    if not items:
        return "Non ho trovato risultati per la tua ricerca."

    best = items[0]

    best_title = best.get("title")
    best_price = best.get("price")
    best_seller = best.get("seller_name") or best.get("seller_username")

    trust = best.get("trust_score")
    score = best.get("_final_score") or best.get("ranking_score") or best.get("_rerank_score")

    explanation = []

    explanation.append(
        f"Ho trovato {len(items)} risultati per '{query}'."
    )

    if best_title:
        explanation.append(
            f"Il prodotto più rilevante è '{best_title}'."
        )

    if trust is not None and best_seller:
        explanation.append(
            f"Il venditore {best_seller} ha un trust score di {round(float(trust) * 100)}%."
        )

    if best_price is not None:
        explanation.append(
            f"Il prezzo del prodotto è {best_price}€."
        )

    reasons = best.get("explanations") or []
    if reasons:
        readable = ", ".join(reasons[:4])
        explanation.append(
            f"Questo risultato è stato classificato in alto perché: {readable}."
        )
    elif score:
        explanation.append(
            "Questo prodotto è stato classificato in cima grazie alla combinazione di rilevanza rispetto alla query, affidabilità del venditore, segnali RAG e prezzo competitivo."
        )

    product_boost = best.get("_rag_product_boost")
    seller_boost = best.get("_rag_seller_boost")
    sentiment_signal = best.get("_rag_sentiment_signal")

    rag_bits = []

    if isinstance(product_boost, (int, float)) and product_boost > 0.04:
        rag_bits.append("il contesto prodotto recuperato è coerente con questo annuncio")

    if isinstance(seller_boost, (int, float)) and seller_boost > 0.04:
        rag_bits.append("i feedback recuperati confermano il profilo del venditore")

    if isinstance(sentiment_signal, (int, float)):
        if sentiment_signal > 0.01:
            rag_bits.append("i feedback recenti mostrano segnali positivi")
        elif sentiment_signal < -0.01:
            rag_bits.append("nei feedback recenti compaiono alcuni segnali negativi")

    if rag_bits:
        explanation.append(
            "Inoltre, il ranking è stato rafforzato dal RAG perché " + ", ".join(rag_bits) + "."
        )

    feedbacks = best.get("rag_feedback") or []
    if feedbacks:
        examples = []

        for f in feedbacks[:2]:
            text = f.get("text")
            if text:
                examples.append(text)

        if examples:
            joined = " | ".join(examples)
            explanation.append(
                f"Alcuni feedback recenti sul venditore indicano: \"{joined}\"."
            )

    return " ".join(explanation)