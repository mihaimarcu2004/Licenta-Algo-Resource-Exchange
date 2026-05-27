from __future__ import annotations

from typing import Iterable, Optional, Set

EV_HEURISTIC_FREEZE_OTHER_ALLOCATIONS = "freeze_other_allocations"
EV_HEURISTIC_ONE_STEP_CONSTANT_UTILITY = "one_step_constant_utility"
EV_HEURISTIC_CAP_HORIZON_20 = "cap_horizon_20"
EV_HEURISTIC_CAP_HORIZON_50 = "cap_horizon_50"
EV_HEURISTIC_SKIP_INACTIVE_PRUNING = "skip_inactive_pruning"
EV_HEURISTIC_NO_FORECAST_RENEWALS = "no_forecast_renewals"
EV_HEURISTIC_PROPORTIONAL_SCREENING = "proportional_screening"
EV_HEURISTIC_TOP_K_CANDIDATES = "top_k_candidates"
EV_HEURISTIC_LOCAL_ROLLOUT = "local_rollout"
EV_HEURISTIC_ADAPTIVE_HORIZON = "adaptive_horizon"
EV_HEURISTIC_COOLDOWN_REJECTIONS = "cooldown_rejections"
EV_HEURISTIC_HYSTERESIS_THRESHOLD = "hysteresis_threshold"

EV_HEURISTICS = {
    EV_HEURISTIC_FREEZE_OTHER_ALLOCATIONS,
    EV_HEURISTIC_ONE_STEP_CONSTANT_UTILITY,
    EV_HEURISTIC_CAP_HORIZON_20,
    EV_HEURISTIC_CAP_HORIZON_50,
    EV_HEURISTIC_SKIP_INACTIVE_PRUNING,
    EV_HEURISTIC_NO_FORECAST_RENEWALS,
    EV_HEURISTIC_PROPORTIONAL_SCREENING,
    EV_HEURISTIC_TOP_K_CANDIDATES,
    EV_HEURISTIC_LOCAL_ROLLOUT,
    EV_HEURISTIC_ADAPTIVE_HORIZON,
    EV_HEURISTIC_COOLDOWN_REJECTIONS,
    EV_HEURISTIC_HYSTERESIS_THRESHOLD,
}

EV_HEURISTIC_DESCRIPTIONS = {
    EV_HEURISTIC_FREEZE_OTHER_ALLOCATIONS: (
        "Keep all non-candidate-link allocations fixed at their current amounts during EV rollouts. "
        "Only the candidate or renewal link is reallocated by the strategy."
    ),
    EV_HEURISTIC_ONE_STEP_CONSTANT_UTILITY: (
        "Simulate only the first EV step and assume that same utility repeats for the whole horizon."
    ),
    EV_HEURISTIC_CAP_HORIZON_20: "Cap EV rollout horizons at 20 periods.",
    EV_HEURISTIC_CAP_HORIZON_50: "Cap EV rollout horizons at 50 periods.",
    EV_HEURISTIC_SKIP_INACTIVE_PRUNING: "Do not remove low-flow links inside EV rollouts.",
    EV_HEURISTIC_NO_FORECAST_RENEWALS: "Do not renew other expiring links inside EV rollouts.",
    EV_HEURISTIC_PROPORTIONAL_SCREENING: (
        "For proportional strategies, screen new-link candidates with an upper bound and fast score. "
        "Only borderline candidates need the expensive EV rollout."
    ),
    EV_HEURISTIC_TOP_K_CANDIDATES: "Run expensive new-link EV rollouts only for each agent's top-K fast-score candidates.",
    EV_HEURISTIC_LOCAL_ROLLOUT: "During EV rollout, simulate only {i,j} and their current neighbors.",
    EV_HEURISTIC_ADAPTIVE_HORIZON: "Shorten EV horizons when the remaining discounted upper bound is negligible.",
    EV_HEURISTIC_COOLDOWN_REJECTIONS: "Temporarily skip recently rejected directed candidate links.",
    EV_HEURISTIC_HYSTERESIS_THRESHOLD: "Require a positive margin above cost before fast acceptance.",
}


def normalize_ev_heuristics(heuristics: Optional[Iterable[str]]) -> Set[str]:
    if heuristics is None:
        return set()

    normalized = {str(name).strip() for name in heuristics if str(name).strip()}
    unknown = normalized - EV_HEURISTICS
    if unknown:
        valid = ", ".join(sorted(EV_HEURISTICS))
        raise ValueError(f"Unknown EV heuristics: {sorted(unknown)}. Valid heuristics: {valid}")
    return normalized


__all__ = [
    "EV_HEURISTICS",
    "EV_HEURISTIC_DESCRIPTIONS",
    "EV_HEURISTIC_FREEZE_OTHER_ALLOCATIONS",
    "EV_HEURISTIC_ONE_STEP_CONSTANT_UTILITY",
    "EV_HEURISTIC_CAP_HORIZON_20",
    "EV_HEURISTIC_CAP_HORIZON_50",
    "EV_HEURISTIC_SKIP_INACTIVE_PRUNING",
    "EV_HEURISTIC_NO_FORECAST_RENEWALS",
    "EV_HEURISTIC_PROPORTIONAL_SCREENING",
    "EV_HEURISTIC_TOP_K_CANDIDATES",
    "EV_HEURISTIC_LOCAL_ROLLOUT",
    "EV_HEURISTIC_ADAPTIVE_HORIZON",
    "EV_HEURISTIC_COOLDOWN_REJECTIONS",
    "EV_HEURISTIC_HYSTERESIS_THRESHOLD",
    "normalize_ev_heuristics",
]
