# Data Contracts

## Overview

All NumPy archives are loaded with `allow_pickle=False`. Embeddings must be finite and L2-normalized before transport training or evaluation. Boolean padding masks use `true` for padded positions.

## Transport Training NPZ

`scripts/train_transport.py` reads arrays sharing the same user axis:

| Field | Shape | Description |
|---|---|---|
| `source` | `[N,K,D]` | Source samples on the unit sphere |
| `target` | `[N,K,D]` | Paired training endpoints |
| `history` | `[N,H,D]` | Observed-history semantic embeddings |
| `history_mask` | `[N,H]` bool | History padding mask |
| `evidence` | `[N,E,D]` | Collaborative-evidence semantic embeddings |
| `evidence_mask` | `[N,E]` bool | Evidence padding mask |

The history-conditioned transport ignores the evidence arrays. They remain part of the shared contract so the same training file can be used for both transport variants.

## Source Calibration NPZ

`scripts/train_source_calibration.py` reads the transport fields above plus `target_valid`:

| Field | Shape | Description |
|---|---|---|
| `source` | `[N,K,D]` | Source samples used by the calibration stage |
| `target` | `[N,M,D]` | User target embeddings, padded along `M` |
| `target_valid` | `[N,M]` bool | `true` for a real target and `false` for padding |
| `history`, `history_mask` | As above | Observed-history condition |
| `evidence`, `evidence_mask` | As above | Collaborative-evidence condition |

Every user must have at least one valid target. The evaluator ignores padded targets in Energy and Coverage.

## Generation NPZ

`scripts/generate_embeddings.py` reads `user_id`, `source`, `history`, `history_mask`, `evidence`, and `evidence_mask`, and writes `user_id` plus `generated` arrays. The generated file can be combined with the ragged history and target arrays to form the evaluation contract.

## Embedding Evaluation NPZ

`recdiffusion evaluate` reads:

| Field | Shape | Description |
|---|---|---|
| `user_id` | `[U]` | Stable integer user identifier |
| `generated` | `[U,K,D]` | Re-encoded generated-caption embeddings |
| `history_values` | `[sum H_u,D]` | Concatenated history embeddings |
| `history_offsets` | `[U+1]` | Ragged history offsets |
| `target_values` | `[sum M_u,D]` | Concatenated held-out target embeddings |
| `target_offsets` | `[U+1]` | Ragged target offsets |

Offsets must start at zero, be monotonically nondecreasing, and end at the number of rows in the corresponding values array. All methods in a comparison must use the same user axis, targets, and number of generated samples per user.

## Judge Aggregation JSON

`scripts/aggregate_judge.py` expects a top-level `users` array. Each user record contains:

```json
{
  "user_id": 0,
  "incremental_target_ids": ["T001"],
  "validity": {
    "method_a": {
      "0": {"T001": 1},
      "1": {"T001": 1}
    }
  }
}
```

Each validity value is `0`, `1`, or `null`. A `null` value denotes an incomplete annotation. The cohort size is inferred from `users`; the number of candidates in a semantic-matching task is inferred from `candidate_order`.

## Array Preparation

`scripts/prepare_arrays.py` accepts one `--array NAME=PATH` argument per required field, writes a compressed NPZ, and validates it with the corresponding loader before returning successfully.
