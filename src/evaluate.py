"""Evaluation: exact-match and per-attribute accuracy.

Includes the parser that splits a (possibly misspelled) generated word
back into size / color / shape [/ relation / size / color / shape].
"""


from difflib import get_close_matches
from typing import Dict, List, Optional

SIZES = ["small", "large"]
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle", "cross"]
RELATIONS = ["leftof", "rightof", "above", "below"]


def _match_best_prefix(word: str, candidates: List[str]) -> Optional[str]:
    """Trouve le candidat le plus probable au debut de `word`, meme mal orthographie."""
    for cand in sorted(candidates, key=len, reverse=True):
        if word.startswith(cand):
            return cand

    best_match = None
    best_ratio = 0.0
    for cand in candidates:
        prefix = word[: len(cand)]
        matches = get_close_matches(prefix, [cand], n=1, cutoff=0.4)
        if matches:
            import difflib
            ratio = difflib.SequenceMatcher(None, prefix, cand).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = cand
    return best_match


def parse_word(word: str) -> Dict[str, Optional[str]]:
    """Parse 'largeredcircle' -> {size:'large', color:'red', shape:'circle', ...}"""
    result = {"size": None, "color": None, "shape": None,
              "relation": None, "size2": None, "color2": None, "shape2": None}
    remaining = word

    def consume(candidates, key):
        nonlocal remaining
        match = _match_best_prefix(remaining, candidates)
        if match is None:
            return False
        result[key] = match
        remaining = remaining[len(match):]
        return True

    if not consume(SIZES, "size"):
        return result
    if not consume(COLORS, "color"):
        return result
    if not consume(SHAPES, "shape"):
        return result
    if remaining == "":
        return result
    if not consume(RELATIONS, "relation"):
        return result
    if not consume(SIZES, "size2"):
        return result
    if not consume(COLORS, "color2"):
        return result
    consume(SHAPES, "shape2")
    return result


def evaluate_predictions(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    n = len(targets)
    exact_matches = 0
    attr_correct = {k: 0 for k in ["size", "color", "shape", "relation", "size2", "color2", "shape2"]}
    attr_total = {k: 0 for k in attr_correct}

    for pred, target in zip(predictions, targets):
        if pred == target:
            exact_matches += 1

        pred_attrs = parse_word(pred)
        target_attrs = parse_word(target)

        for key in attr_correct:
            if target_attrs[key] is not None:
                attr_total[key] += 1
                if pred_attrs[key] == target_attrs[key]:
                    attr_correct[key] += 1

    metrics = {"exact_match": exact_matches / n}
    for key in attr_correct:
        if attr_total[key] > 0:
            metrics[f"attr_acc_{key}"] = attr_correct[key] / attr_total[key]
    return metrics


if __name__ == "__main__":
    preds = ["largeredcircle", "smallbluesquareleftoflargeyellowtriangle", "largeredcirle"]
    targets = ["largeredcircle", "smallbluesquareleftoflargeyellowtriangle", "largeredcircle"]
    print(evaluate_predictions(preds, targets))
