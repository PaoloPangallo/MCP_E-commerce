from typing import List, Dict


def build_context(query: str, products: List[Dict], docs: List[Dict]) -> str:
    lines = []

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

            lines.append(
                f"- {title} | {price} {currency} | seller: {seller} | trust: {trust}"
            )

    if docs:
        lines.append("")
        lines.append("Relevant seller feedback:")

        for d in docs[:5]:
            seller = d.get("seller")
            text = d.get("text")

            lines.append(f"- ({seller}) {text}")

    return "\n".join(lines)
