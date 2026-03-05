from typing import List, Dict


def explain_results(query: str, items: List[Dict]) -> str:

    if not items:
        return "Non ho trovato risultati per la tua ricerca."

    best = items[0]

    best_title = best.get("title")
    best_price = best.get("price")
    best_seller = best.get("seller_name")

    trust = best.get("trust_score")
    score = best.get("ranking_score") or best.get("_rerank_score")

    explanation = []

    # ------------------------------------------------
    # overview
    # ------------------------------------------------

    explanation.append(
        f"Ho trovato {len(items)} risultati per '{query}'."
    )

    if best_title:
        explanation.append(
            f"Il prodotto più rilevante è '{best_title}'."
        )

    # ------------------------------------------------
    # trust
    # ------------------------------------------------

    if trust is not None and best_seller:
        explanation.append(
            f"Il venditore {best_seller} ha un trust score di {round(trust * 100)}%."
        )

    # ------------------------------------------------
    # prezzo
    # ------------------------------------------------

    if best_price:
        explanation.append(
            f"Il prezzo del prodotto è {best_price}€."
        )

    # ------------------------------------------------
    # explainable ranking
    # ------------------------------------------------

    reasons = best.get("explanations") or []

    if reasons:

        readable = ", ".join(reasons)

        explanation.append(
            f"Questo risultato è stato classificato in alto perché: {readable}."
        )

    elif score:
        explanation.append(
            "Questo prodotto è stato classificato in cima grazie alla combinazione di rilevanza rispetto alla query, affidabilità del venditore e prezzo competitivo."
        )

    # ------------------------------------------------
    # RAG feedback
    # ------------------------------------------------

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