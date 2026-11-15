from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


SECURE_AGG_SCHEME = "ring-pair-mask-v1"


def average_client_updates(client_updates: list[dict[str, Any]]) -> tuple[list[np.ndarray], int]:
    if not client_updates:
        raise ValueError("Secure aggregation requires at least one client update")

    total_examples = sum(int(row["num_examples"]) for row in client_updates)
    if total_examples <= 0:
        raise ValueError("Secure aggregation requires a positive number of training examples")

    averaged_update = [np.zeros_like(array, dtype=np.float64) for array in client_updates[0]["update"]]
    for row in client_updates:
        for index, array in enumerate(row["update"]):
            averaged_update[index] += array.astype(np.float64, copy=False) * float(row["num_examples"])
    scale = 1.0 / float(total_examples)
    return [array * scale for array in averaged_update], total_examples


def _build_mask_pairs(client_ids: list[int], degree: int) -> list[tuple[int, int]]:
    if len(client_ids) < 2 or degree <= 0:
        return []

    ordered_ids = sorted(client_ids)
    max_degree = min(degree, len(ordered_ids) - 1)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for offset in range(1, max_degree + 1):
        for index, left_id in enumerate(ordered_ids):
            right_id = ordered_ids[(index + offset) % len(ordered_ids)]
            pair = (left_id, right_id) if left_id < right_id else (right_id, left_id)
            if pair[0] == pair[1] or pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def _mask_scale_factor(parameter_count: int, target_norm: float) -> float:
    if parameter_count <= 0 or target_norm <= 0:
        return 0.0
    # Uniform[-1, 1] masks have variance 1/3, so this targets a stable overall mask norm.
    return target_norm / math.sqrt(parameter_count / 3.0)


def _pair_mask(
    *,
    round_number: int,
    run_nonce: bytes,
    left_id: int,
    right_id: int,
    tensor_index: int,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    scale_factor: float,
) -> np.ndarray:
    value_count = int(np.prod(shape, dtype=np.int64))
    if value_count == 0 or scale_factor == 0.0:
        return np.zeros(shape, dtype=dtype)

    seed_material = b"|".join(
        [
            SECURE_AGG_SCHEME.encode("ascii"),
            run_nonce.hex().encode("ascii"),
            str(round_number).encode("ascii"),
            str(left_id).encode("ascii"),
            str(right_id).encode("ascii"),
            str(tensor_index).encode("ascii"),
        ]
    )
    random_bytes = hashlib.shake_256(seed_material).digest(value_count * 4)
    values = np.frombuffer(random_bytes, dtype=np.uint32, count=value_count).astype(np.float64, copy=False)
    values = ((values / np.float64(np.iinfo(np.uint32).max)) * 2.0) - 1.0
    mask = (values * scale_factor).reshape(shape)
    return mask.astype(dtype, copy=False)


def secure_aggregate_weighted_updates(
    *,
    client_updates: list[dict[str, Any]],
    round_number: int,
    mask_degree: int,
    mask_scale: float,
    clipping_norm: float,
    run_nonce: bytes,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    plain_average, total_examples = average_client_updates(client_updates)
    participant_ids = [int(row["client_id"]) for row in client_updates]
    pair_count = 0

    if len(participant_ids) < 2 or mask_degree <= 0:
        return plain_average, {
            "enabled": len(participant_ids) > 1,
            "scheme": "single-client-bypass" if len(participant_ids) < 2 else "masking-disabled",
            "participant_ids": participant_ids,
            "protected_participant_count": max(len(participant_ids) - 1, 0),
            "pair_count": pair_count,
            "mask_degree": max(mask_degree, 0),
            "mask_target_norm": 0.0,
            "cancellation_error_max_abs": 0.0,
            "total_examples": total_examples,
        }

    client_arrays = [
        [(array * row["num_examples"]).astype(array.dtype, copy=False) for array in row["update"]]
        for row in client_updates
    ]
    index_by_client_id = {int(row["client_id"]): index for index, row in enumerate(client_updates)}
    mask_pairs = _build_mask_pairs(participant_ids, mask_degree)
    pair_count = len(mask_pairs)

    parameter_count = int(sum(array.size for array in client_updates[0]["update"]))
    reference_norm = max(float(clipping_norm), 1.0)
    target_mask_norm = reference_norm * float(mask_scale)
    scale_factor = _mask_scale_factor(parameter_count, target_mask_norm)

    for left_id, right_id in mask_pairs:
        left_buffers = client_arrays[index_by_client_id[left_id]]
        right_buffers = client_arrays[index_by_client_id[right_id]]
        for tensor_index, (left_array, right_array) in enumerate(zip(left_buffers, right_buffers, strict=True)):
            mask = _pair_mask(
                round_number=round_number,
                run_nonce=run_nonce,
                left_id=left_id,
                right_id=right_id,
                tensor_index=tensor_index,
                shape=left_array.shape,
                dtype=left_array.dtype,
                scale_factor=scale_factor,
            )
            left_array += mask
            right_array -= mask

    masked_average = [np.zeros_like(array, dtype=np.float64) for array in client_updates[0]["update"]]
    for weighted_update in client_arrays:
        for index, array in enumerate(weighted_update):
            masked_average[index] += array.astype(np.float64, copy=False)

    scale = 1.0 / float(total_examples)
    masked_average = [array * scale for array in masked_average]
    cancellation_error = max(
        float(np.max(np.abs(masked - plain)))
        for masked, plain in zip(masked_average, plain_average, strict=True)
    )

    return masked_average, {
        "enabled": True,
        "scheme": SECURE_AGG_SCHEME,
        "participant_ids": participant_ids,
        "protected_participant_count": len(participant_ids),
        "pair_count": pair_count,
        "mask_degree": min(mask_degree, len(participant_ids) - 1),
        "mask_target_norm": float(target_mask_norm),
        "cancellation_error_max_abs": cancellation_error,
        "total_examples": total_examples,
    }
