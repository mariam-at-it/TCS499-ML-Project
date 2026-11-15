from __future__ import annotations

from opacus.accountants import RDPAccountant


def compute_privacy_budget(*, noise_multiplier: float, sample_rate: float, rounds: int, delta: float = 1e-5) -> dict[str, float]:
    if noise_multiplier <= 0 or sample_rate <= 0 or rounds <= 0:
        return {"epsilon": float("inf"), "delta": delta}
    accountant = RDPAccountant()
    for _ in range(rounds):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    epsilon = accountant.get_epsilon(delta=delta)
    return {"epsilon": float(epsilon), "delta": float(delta)}

