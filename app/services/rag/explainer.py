from typing import List, Dict


def explain_results(query: str, items: List[Dict]) -> str:

    if not items:
        return "Non ho trovato risultati per la tua ricerca."

    best = items[0]

    best_title = best.get("title")
    best_price = best.get("price")
    best_seller = best.get("seller_name")
    trust = best.get("trust_score")

    explanation = []

    explanation.append(f"Ho trovato {len(items)} risultati per '{query}'.")

    if best_title:
        explanation.append(
            f"Il risultato migliore è '{best_title}'."
        )

    if trust:
        explanation.append(
            f"Il venditore {best_seller} ha trust score {round(trust,2)}."
        )

    if best_price:
        explanation.append(
            f"Il prezzo è {best_price}."
        )

    return "\n".join(explanation)