"""Server-issued random challenge scripts for active liveness (M4).

Randomness comes from `secrets` so prompts cannot be predicted and pre-recorded.
"""
from __future__ import annotations

import secrets
from dataclasses import asdict

from ml.common.interface import ChallengeStep

_KINDS = {
    "turn_left": "Slowly turn your head to the LEFT",
    "turn_right": "Slowly turn your head to the RIGHT",
    "blink_twice": "Blink TWICE",
    "look_up": "Look UP briefly",
    "smile": "SMILE",
}


def generate_challenge(duration_ms: int = 7000, steps: int = 3, digits_len: int = 3) -> list[ChallengeStep]:
    rng = secrets.SystemRandom()
    pool = list(_KINDS)
    rng.shuffle(pool)
    kinds = pool[: max(steps - 1, 1)]
    # Always include read_digits, at a random position
    kinds.insert(rng.randrange(len(kinds) + 1), "read_digits")
    lead_in = 500
    slot = (duration_ms - lead_in) // len(kinds)
    out: list[ChallengeStep] = []
    for i, k in enumerate(kinds):
        start = lead_in + i * slot
        if k == "read_digits":
            digits = [rng.randrange(10) for _ in range(digits_len)]
            prompt = "Read aloud: " + "-".join(map(str, digits))
            payload = {"digits": digits}
        else:
            prompt, payload = _KINDS[k], {}
        out.append(ChallengeStep(kind=k, prompt=prompt, start_ms=start, end_ms=start + slot, payload=payload))
    return out


def challenge_to_json(steps: list[ChallengeStep]) -> list[dict]:
    return [asdict(s) for s in steps]


def challenge_from_json(data: list[dict]) -> list[ChallengeStep]:
    return [ChallengeStep(**d) for d in data]
