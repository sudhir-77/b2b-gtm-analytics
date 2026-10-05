"""Regex fact-check: every number in a draft must appear in the computed tables."""

from __future__ import annotations

import re
from typing import Any

# A number, with optional thousands separators and a trailing percent sign.
# The leading lookbehind skips digits inside identifiers such as SmolLM2.
# The trailing check allows sentence punctuation ("999.") while still
# consuming a percent sign that sits in front of that punctuation.
NUMBER_RE = re.compile(
    r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%)?(?!\d)"
)


def collect_numbers(obj: Any, found: list[float] | None = None) -> list[float]:
    if found is None:
        found = []
    if isinstance(obj, bool) or obj is None:
        return found
    if isinstance(obj, (int, float)):
        value = float(obj)
        if value == value and value not in (float("inf"), float("-inf")):
            found.append(value)
        return found
    if isinstance(obj, dict):
        for value in obj.values():
            collect_numbers(value, found)
        return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            collect_numbers(value, found)
        return found
    return found


def extract_numbers(text: str) -> list[dict[str, float | int | bool | str]]:
    tokens: list[dict[str, float | int | bool | str]] = []
    for match in NUMBER_RE.finditer(text):
        whole, frac, percent = match.group(1), match.group(2), match.group(3)
        digits = whole.replace(",", "") + (frac or "")
        decimals = 0 if frac is None else len(frac) - 1
        tokens.append(
            {
                "text": match.group(0),
                "value": float(digits),
                "decimals": decimals,
                "percent": bool(percent),
            }
        )
    return tokens


def _is_allowed(token: dict, allowed: list[float]) -> bool:
    decimals = int(token["decimals"])
    tolerance = 1e-6 if decimals == 0 else 0.51 * (10 ** (-decimals))
    candidates = [float(token["value"])]
    if token["percent"]:
        candidates.append(float(token["value"]) / 100.0)
    for candidate in candidates:
        for item in allowed:
            if abs(candidate - item) <= tolerance:
                return True
    return False


def fact_check(text: str, tables: Any) -> dict:
    """Return the share of numeric tokens that match ``tables``.

    A draft with no numbers fails. The published summary has to carry
    figures, and those figures have to come from the computed tables.
    """
    allowed = collect_numbers(tables)
    tokens = extract_numbers(text)
    if not tokens:
        return {
            "match_rate": 0.0,
            "n_numbers": 0,
            "n_matched": 0,
            "passed": False,
            "unmatched": [],
        }
    unmatched = [str(token["text"]) for token in tokens if not _is_allowed(token, allowed)]
    matched = len(tokens) - len(unmatched)
    rate = matched / len(tokens)
    return {
        "match_rate": float(rate),
        "n_numbers": len(tokens),
        "n_matched": matched,
        "passed": len(unmatched) == 0,
        "unmatched": unmatched,
    }


def drop_unverified_sentences(text: str, tables: Any) -> str:
    """Keep sentences whose numbers all match. Number-free sentences stay."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for part in parts:
        if not part:
            continue
        checked = fact_check(part, tables)
        if checked["n_numbers"] == 0 or checked["passed"]:
            kept.append(part)
    return " ".join(kept).strip()
