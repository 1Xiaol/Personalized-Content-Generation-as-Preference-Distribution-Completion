"""Offline validation and aggregation for the frozen LLM-judge protocol."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

INCREMENTAL_STATES = frozenset({"weak_observed", "weak_partial", "not_observed"})


def classify_aspect_states(
    aspect_ids: Sequence[str], p2_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, int | str]]:
    """Combine every history chunk before assigning the five frozen states."""
    counts = {aspect_id: Counter() for aspect_id in aspect_ids}
    for row in p2_rows:
        groups = {
            "direct": row.get("direct_aspect_ids", []),
            "related": row.get("related_aspect_ids", []),
            "uncertain": row.get("uncertain_aspect_ids", []),
        }
        flattened = [aspect for values in groups.values() for aspect in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("P2 relation groups must be mutually exclusive per history row")
        unknown = set(flattened) - set(aspect_ids)
        if unknown:
            raise ValueError(f"P2 returned unknown aspects: {sorted(unknown)}")
        for relation, values in groups.items():
            for aspect_id in values:
                counts[aspect_id][relation] += 1

    output = {}
    for aspect_id in aspect_ids:
        count = counts[aspect_id]
        if count["direct"] >= 2:
            state = "observed"
        elif count["uncertain"] >= 1:
            state = "uncertain"
        elif count["direct"] == 1:
            state = "weak_observed"
        elif count["related"] >= 1:
            state = "weak_partial"
        else:
            state = "not_observed"
        output[aspect_id] = {
            "state": state,
            "direct_count": count["direct"],
            "related_count": count["related"],
            "uncertain_count": count["uncertain"],
        }
    return output


def incremental_target_ids(
    aspects: Sequence[Mapping[str, Any]], states: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    """Return scoreable targets in weak-history states, not strict semantic novelty."""
    result = set()
    for aspect in aspects:
        aspect_id = str(aspect["aspect_id"])
        if states[aspect_id]["state"] in INCREMENTAL_STATES:
            result.update(map(str, aspect["target_ids"]))
    return result


def quality_map(p3_result: Mapping[str, Any]) -> dict[str, int | None]:
    result = {}
    for row in p3_result["rows"]:
        result[str(row["generation_id"])] = (
            int(row["quality"]) if row["status"] == "ok" else None
        )
    return result


def target_validity(
    p5_result: Mapping[str, Any], generation_quality: Mapping[str, int | None]
) -> dict[str, int | None]:
    """A target is covered when semantic score >=2 and generation quality ==2."""
    order = list(map(str, p5_result["candidate_order"]))
    if set(order) - set(generation_quality):
        raise ValueError("P5 candidates and P3 generation IDs differ")
    output = {}
    for row in p5_result["rows"]:
        target_id = str(row["target_id"])
        if row["status"] != "ok":
            output[target_id] = None
            continue
        scores = row["scores"]
        if len(scores) != len(order) or any(score not in (0, 1, 2, 3) for score in scores):
            raise ValueError("P5 score axis is invalid")
        output[target_id] = int(
            any(score >= 2 and generation_quality[generation_id] == 2 for generation_id, score in zip(order, scores))
        )
    return output


def aggregate_ivc_users(
    users: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    orders: Sequence[str] = ("0", "1"),
    bootstrap_resamples: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Target-within-user, presentation-order averaged, then user-macro IVC."""
    user_ids = [user["user_id"] for user in users]
    if len(user_ids) != len(set(user_ids)):
        raise ValueError("judge cohort contains duplicate user IDs")
    per_user = []
    for user in users:
        incremental = set(map(str, user["incremental_target_ids"]))
        common = sorted(
            target_id
            for target_id in incremental
            if all(
                user["validity"][method][str(order)].get(target_id) is not None
                for method in methods
                for order in orders
            )
        )
        values = None
        if common:
            values = {
                method: float(
                    np.mean(
                        [
                            user["validity"][method][str(order)][target_id]
                            for target_id in common
                            for order in orders
                        ]
                    )
                )
                for method in methods
            }
        per_user.append({"user_id": user["user_id"], "common_targets": common, "ivc": values})

    summary = {}
    rng = np.random.default_rng(seed)
    for method in methods:
        values = np.asarray(
            [row["ivc"][method] for row in per_user if row["ivc"] is not None], dtype=np.float64
        )
        if not len(values):
            summary[method] = {"estimate": None, "ci_low": None, "ci_high": None, "n_users": 0}
            continue
        draws = values[rng.integers(0, len(values), size=(bootstrap_resamples, len(values)))].mean(1)
        summary[method] = {
            "estimate": float(values.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "n_users": int(len(values)),
        }
    return {"cohort_users": len(users), "summary": summary, "per_user": per_user}
