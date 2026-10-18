"""Two explicit validators for the sales analytics chatbot."""
import re

SALES_TERMS = {
    "sales", "profit", "profitability", "revenue", "order", "orders",
    "product", "products", "category", "sub-category", "customer",
    "customers", "region", "country", "market", "discount", "quantity",
    "shipping", "ship mode", "segment", "monthly", "trend", "top",
    "highest", "lowest", "average", "total", "margin", "loss", "date",
    "compare", "comparison", "dashboard", "data", "dataset",
}

SUPPORTED_FIELDS = {
    "sales", "profit", "quantity", "discount", "shipping_cost",
    "order_date", "ship_date", "ship_mode", "customer_id", "customer_name",
    "segment", "city", "state", "country", "market", "region", "category",
    "sub-category", "product_id", "product_name", "order_priority", "order_id",
}

def validator_1_dataset_related(question: str) -> tuple[bool, str]:
    """Reject greetings and clearly unrelated prompts, while allowing sales questions."""
    q = (question or "").strip().lower()
    if not q:
        return False, "Please enter a question."
    if q in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}:
        return False, "Please ask a question about the connected sales dataset."
    tokens = set(re.findall(r"[a-z][a-z-]*", q))
    if not (tokens & SALES_TERMS):
        return False, "This chatbot answers questions about the connected Superstore sales dataset."
    return True, "Question appears related to the sales dataset."

def validator_2_answerable(question: str) -> tuple[bool, str]:
    """Check for supported sales-analysis intent before asking the model for a plan."""
    q = (question or "").lower()
    intent_words = (
        "total", "sum", "average", "mean", "highest", "lowest", "top",
        "bottom", "compare", "trend", "monthly", "by ", "profit", "sales",
        "count", "margin", "discount", "quantity", "loss", "which", "what",
        "how many", "how much", "orders", "products", "customers",
    )
    if any(term in q for term in intent_words):
        return True, "Question matches a supported analytics intent."
    return False, (
        "I can answer totals, averages, counts, rankings, comparisons, "
        "monthly trends, and profitability questions using the available columns."
    )
