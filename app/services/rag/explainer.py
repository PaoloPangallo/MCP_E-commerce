from typing import List, Dict


def explain_results(query: str, items: List[Dict], missing_info: List[str] = None) -> str:

    if not items:
        # Se non ci sono risultati, cerchiamo di essere utili
        msg = f"🔍 **Ricerca per: '{query}'**\n\nPurtroppo non ho trovato risultati che corrispondano esattamente ai tuoi criteri su eBay."
        if missing_info:
            readable = ", ".join([f"**{m}**" for m in missing_info])
            msg += f"\n\n💡 **Suggerimento:** Prova a specificare meglio questi dettagli per aiutarmi a trovare quello che cerchi: {readable}."
        return msg

    best = items[0]
    best_title = best.get("title", "Prodotto senza titolo")
    best_price = best.get("price", "N.D.")
    best_currency = best.get("currency", "€")
    best_seller = best.get("seller_name", "Venditore Privato")
    trust = best.get("trust_score", 0)
    
    explanation = []
    explanation.append(f"### 🎯 Risultato consigliato per: *{query}*")
    explanation.append(f"Ho analizzato **{len(items)}** prodotti e questo è il migliore per te:\n")
    
    # Card del prodotto
    explanation.append(f"**[{best_title}]({best.get('url', '#')})**")
    explanation.append(f"💰 **Prezzo:** {best_price} {best_currency}")
    explanation.append(f"👤 **Venditore:** {best_seller} (Affidabilità: **{round(trust * 100)}%**)")
    explanation.append("")

    # Motivazioni del ranking
    reasons = best.get("explanations") or []
    if reasons:
        explanation.append("✨ **Perché lo abbiamo scelto:**")
        for r in reasons:
            explanation.append(f"- {r}")
    else:
        explanation.append("✨ **Perché lo abbiamo scelto:** Questo prodotto offre il miglior bilanciamento tra pertinenza, prezzo e reputazione del venditore.")
    
    explanation.append("")

    # RAG Social Proof
    feedbacks = best.get("rag_feedback") or []
    if feedbacks:
        explanation.append("💬 **Cosa dicono gli acquirenti di questo venditore:**")
        for f in feedbacks[:2]:
            text = f.get("text", "").strip()
            if text:
                explanation.append(f"> \"{text}\"")
        explanation.append("")

    # Suggerimenti per affinare
    if missing_info:
        readable = ", ".join([f"**{m}**" for m in missing_info])
        explanation.append(f"💡 **Vuoi essere più preciso?** Se mi dici il tuo **{readable}** posso filtrare i risultati ancora meglio.")

    return "\n".join(explanation)
