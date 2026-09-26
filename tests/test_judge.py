from recdiffusion.judge import (
    aggregate_ivc_users,
    classify_aspect_states,
    incremental_target_ids,
    target_validity,
)


def test_aspect_state_precedence_and_incremental_targets() -> None:
    rows = [
        {
            "direct_aspect_ids": ["A", "B"],
            "related_aspect_ids": ["C"],
            "uncertain_aspect_ids": ["D"],
        },
        {
            "direct_aspect_ids": ["A"],
            "related_aspect_ids": [],
            "uncertain_aspect_ids": [],
        },
    ]
    states = classify_aspect_states(["A", "B", "C", "D", "E"], rows)
    assert [states[key]["state"] for key in "ABCDE"] == [
        "observed",
        "weak_observed",
        "weak_partial",
        "uncertain",
        "not_observed",
    ]
    aspects = [{"aspect_id": key, "target_ids": [f"T{key}"]} for key in "ABCDE"]
    assert incremental_target_ids(aspects, states) == {"TB", "TC", "TE"}


def test_validity_requires_semantic_and_quality() -> None:
    p5 = {
        "candidate_order": ["G1", "G2"],
        "rows": [
            {"target_id": "T1", "status": "ok", "scores": [3, 0]},
            {"target_id": "T2", "status": "ok", "scores": [1, 2]},
        ],
    }
    assert target_validity(p5, {"G1": 1, "G2": 2}) == {"T1": 0, "T2": 1}


def test_users_without_incremental_targets_are_missing() -> None:
    result = aggregate_ivc_users(
        [
            {
                "user_id": 1,
                "incremental_target_ids": [],
                "validity": {"ours": {"0": {}, "1": {}}},
            }
        ],
        methods=["ours"],
        bootstrap_resamples=10,
    )
    assert result["cohort_users"] == 1
    assert result["summary"]["ours"]["estimate"] is None
    assert result["summary"]["ours"]["n_users"] == 0


def test_judge_cohort_size_is_derived_from_input() -> None:
    users = [
        {
            "user_id": user_id,
            "incremental_target_ids": ["T1"],
            "validity": {"method": {"0": {"T1": 1}, "1": {"T1": 1}}},
        }
        for user_id in (11, 23, 47)
    ]
    result = aggregate_ivc_users(users, methods=["method"], bootstrap_resamples=10)
    assert result["cohort_users"] == 3
    assert result["summary"]["method"]["n_users"] == 3


def test_duplicate_judge_users_are_rejected() -> None:
    users = [
        {
            "user_id": 5,
            "incremental_target_ids": [],
            "validity": {"method": {"0": {}, "1": {}}},
        },
        {
            "user_id": 5,
            "incremental_target_ids": [],
            "validity": {"method": {"0": {}, "1": {}}},
        },
    ]
    try:
        aggregate_ivc_users(users, methods=["method"], bootstrap_resamples=10)
    except ValueError as error:
        assert "duplicate user IDs" in str(error)
    else:
        raise AssertionError("duplicate users must fail")
