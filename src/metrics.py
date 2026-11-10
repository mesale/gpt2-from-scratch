"""Text-quality metrics for the generation experiments (pure Python, no torch)."""


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def repetition_rate(tokens, n=3):
    """Fraction of n-grams that already appeared earlier in the same text (0 = no repeats)."""
    grams = ngrams(tokens, n)
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def distinct_n(samples, n=2):
    """Unique n-grams / total n-grams, pooled over several samples (1 = all different)."""
    grams = [g for s in samples for g in ngrams(s, n)]
    if not grams:
        return float("nan")
    return len(set(grams)) / len(grams)