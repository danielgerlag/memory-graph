"""Helpers for independently grading structured agent answers."""

import json


def parse_answer_with_prose(text, parse_json):
    """Extract one JSON object and report any surrounding, unscored prose."""
    try:
        return parse_json(text), False
    except json.JSONDecodeError:
        if not isinstance(text, str):
            raise
        start = text.find("{")
        if start < 0:
            raise
        _, end = json.JSONDecoder().raw_decode(text[start:])
        outside = text[:start] + text[start + end :]
        if "{" in outside or "}" in outside:
            raise ValueError("answer contains more than one JSON object")
        return parse_json(text[start : start + end]), bool(outside.strip())


def normalize_sources(value):
    """Treat empty placeholders as absence, while retaining every real source."""
    if not isinstance(value, list):
        return None
    sources = []
    normalized = False
    for item in value:
        if isinstance(item, str):
            if not item.strip():
                normalized = True
            else:
                sources.append(item)
        elif isinstance(item, list) and not item:
            normalized = True
        else:
            return None
    return sources, normalized


def normalize_notices(value, expected, expected_sources, expected_facts):
    """Keep uncertainty about an unsourced retrieved claim visible, not a false failure.

    An extra "unknown" next to a correctly retrieved claim is reported as ambiguous; any other
    notice, and a missing expected one, still decides the result.
    """
    if (
        isinstance(value, list)
        and "unknown" in value
        and "unknown" not in expected
        and not expected_sources
        and expected_facts
    ):
        return [item for item in value if item != "unknown"], True
    return value, False
