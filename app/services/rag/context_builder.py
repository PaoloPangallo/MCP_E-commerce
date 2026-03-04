from typing import List, Dict


def build_context(query: str, products: List[Dict], docs: List[Dict]) -> str:
    """
    Costruisce il contesto da passare all'LLM.
    """

    context = f"User query:\n{query}\n\n"

    if products:
        context += "Products found:\n"

        for p in products[:5]:
            context += f"- {p.get('title')} ({p.get('price')} {p.get('currency')}) seller: {p.get('seller_name')}\n"

    if docs:
        context += "\nRelevant seller feedback:\n"

        for d in docs[:5]:
            context += f"- {d.get('text')}\n"

    return context