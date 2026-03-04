from typing import List, Dict


def explain_results(query: str, items: List[Dict]) -> str:

    if not items:
        return "Non ho trovato risultati per la tua ricerca."

    best = items[0]

    best_title = best.get("title")
    best_price = best.get("price")
    best_seller = best.get("seller_name")
    trust = best.get("trust_score")
    score = best.get("_rerank_score")

    explanation = []

    explanation.append(
        f"Ho trovato {len(items)} risultati per '{query}'."
    )

    if best_title:
        explanation.append(
            f"Il risultato più rilevante è '{best_title}'."
        )

    if trust:
        explanation.append(
            f"Il venditore {best_seller} ha un trust score di {round(trust*100)}%."
        )

    if best_price:
        explanation.append(
            f"Il prezzo è {best_price}€."
        )

    if score:
        explanation.append(
            "Questo prodotto è stato classificato in cima grazie alla combinazione di affidabilità del venditore, prezzo e rilevanza rispetto alla tua ricerca."
        )

    # feedback RAG
    feedbacks = best.get("rag_feedback") or []

    if feedbacks:
        example = feedbacks[0].get("text")

        if example:
            explanation.append(
                f"Ad esempio, un feedback recente dice: \"{example}\"."
            )

    return " ".join(explanation)