"""
nlp_helper.py
Answers a user's question about ONE plant, using only the text stored in
plant_info.json. Nothing is invented: if no stored sentence matches the
question, the function says so.

Steps:
  1. Safety check   - emergency words -> tell the user to get medical help
  2. Intent check   - keywords decide if the user wants precautions,
                      how-to-use, or the scientific name
  3. Retrieval      - otherwise TF-IDF + cosine similarity finds the stored
                      sentence that best matches the question
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

EMERGENCY_WORDS = [
    "poison", "overdose", "swallowed", "ate too much", "emergency",
    "allergic reaction", "can't breathe", "cannot breathe", "unconscious",
    "seizure", "bleeding heavily",
]

INTENT_KEYWORDS = {
    "precautions": [
        "precaution", "side effect", "safe", "safety", "pregnan", "child",
        "kid", "baby", "avoid", "toxic", "harm", "danger", "risk",
        "allerg", "warning", "careful",
    ],
    "how_to_use": [
        "how to", "how do i", "how can i", "how should i", "prepare",
        "make", "take", "dose", "dosage", "consume", "apply", "tea",
        "juice", "recipe",
    ],
    "scientific_name": ["scientific", "latin", "botanical", "species"],
}

MIN_SIMILARITY = 0.10  # below this score we say "I don't know"

# Words that appear in almost every sentence about plants. They carry no
# meaning for matching, so we ignore them together with normal English
# stop words (the, is, for, ...).
GENERIC_WORDS = [
    "help", "helps", "helping", "good", "used", "use", "uses", "useful",
    "traditionally", "benefit", "benefits", "cure", "cures", "treat",
    "treats", "plant", "does", "work", "works",
]
STOP_WORDS = set(ENGLISH_STOP_WORDS) | set(GENERIC_WORDS)


def _tokenize(text):
    """Lower-case, split into words, drop stop words, and make simple
    plurals match (colds -> cold, coughs -> cough)."""
    tokens = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP_WORDS:
            continue
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        tokens.append(w)
    return tokens

DISCLAIMER = "For educational purposes only. Not medical advice."


def _detect_intent(question):
    q = question.lower()
    for intent, words in INTENT_KEYWORDS.items():
        if any(w in q for w in words):
            return intent
    return None


def answer_question(question, plant_name, info):
    """Return (answer_text, kind) where kind is one of:
    'emergency', 'precautions', 'how_to_use', 'scientific_name',
    'uses', 'unknown'."""
    q = question.strip()
    if not q:
        return "Please type a question.", "unknown"

    nice_name = plant_name.replace("_", " ")
    q_lower = q.lower()

    # 1. safety first
    if any(w in q_lower for w in EMERGENCY_WORDS):
        return (
            "This sounds like it could be urgent. Please contact a doctor, "
            "your local emergency number, or a poison control centre right "
            "away. Do not rely on this app for emergencies.",
            "emergency",
        )

    # 2. intent-based answers
    intent = _detect_intent(q)
    if intent == "precautions":
        return f"Precautions for {nice_name}: {info['precautions']}", "precautions"
    if intent == "how_to_use":
        return f"How {nice_name} is used: {info['how_to_use']}", "how_to_use"
    if intent == "scientific_name":
        return f"Scientific name of {nice_name}: {info['scientific_name']}", "scientific_name"

    # 3. retrieval over the 'uses' sentences
    uses = [u for u in info.get("uses", []) if u.strip()]
    if uses:
        try:
            vec = TfidfVectorizer(tokenizer=_tokenize, lowercase=False, token_pattern=None)
            matrix = vec.fit_transform(uses + [q])
            scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
            best = int(scores.argmax())
            if scores[best] >= MIN_SIMILARITY:
                matched = [uses[i] for i, s in enumerate(scores) if s >= MIN_SIMILARITY]
                return (
                    f"About {nice_name}: " + "; ".join(matched) + ".",
                    "uses",
                )
        except ValueError:
            # happens if the question has only stop words (e.g. "is it?")
            pass

    return (
        f"I don't have verified information about that for {nice_name}. "
        "Please ask a doctor or a qualified Ayurvedic practitioner.",
        "unknown",
    )
