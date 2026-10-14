"""Small helpers shared by the task builders."""

# Legacy ~100-word pool, used only when no --words-file is given (pool-size sensitivity check).
CHAR_PROBE_WORDS_BY_LEN = {
    3: ["cat", "dog", "run", "big", "the", "and", "for", "not", "you", "all"],
    4: ["that", "with", "have", "this", "will", "your", "from", "they", "been", "call"],
    5: ["about", "would", "there", "their", "which", "could", "other", "after", "first", "great"],
    6: ["people", "before", "should", "become", "school", "simple", "church", "strong", "inside", "market"],
    7: ["because", "between", "another", "problem", "thought", "company", "nothing", "against", "history", "program"],
    8: ["children", "national", "together", "business", "possible", "research", "question", "actually", "standard", "building"],
    9: ["important", "community", "education", "different", "political", "something", "necessary", "condition", "following", "beginning"],
    10: ["government", "understand", "experience", "management", "themselves", "generation", "absolutely", "commercial", "definitely", "incredible"],
    11: ["development", "environment", "information", "performance", "opportunity", "technology", "competition", "personality", "independent", "comfortable"],
    12: ["organization", "relationship", "professional", "construction", "metropolitan", "circumstance", "appreciation", "intelligence", "constitution", "conversation"],
}


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th'."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
