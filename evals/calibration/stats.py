"""Agreement maths for judge calibration: Wilson intervals and Cohen's kappa.

Pure functions on ``(label, verdict)`` pairs, with "pass" (``True``) as the positive class.
The labelled set is small, so every rate comes with a 95% interval.
"""

import math
import random
from collections.abc import Sequence

Pair = tuple[bool, bool]
"""``(label, verdict)``: the expected verdict and the judge's."""


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval of a proportion.

    It stays inside [0, 1] and behaves well for small ``n`` and rates near 0 or 1, unlike
    the normal approximation.

    Args:
        successes: Number of successes.
        n: Number of trials; must be positive.
        z: Standard-normal quantile; 1.96 gives a 95% interval.

    Returns:
        ``(low, high)``.

    Raises:
        ValueError: If ``n`` is not positive or ``successes`` is outside ``[0, n]``.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"Need 0 <= successes <= n and n > 0, got {successes}/{n}")
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(pairs: Sequence[Pair]) -> float | None:
    """Return Cohen's kappa: agreement between labels and verdicts beyond chance.

    Args:
        pairs: ``(label, verdict)`` pairs.

    Returns:
        Kappa in [-1, 1], or ``None`` when it is undefined: no pairs, or chance agreement of
        1 (labels and verdicts all the same single value).
    """
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(label == verdict for label, verdict in pairs) / n
    labels_pass = sum(label for label, _ in pairs) / n
    verdicts_pass = sum(verdict for _, verdict in pairs) / n
    chance = labels_pass * verdicts_pass + (1 - labels_pass) * (1 - verdicts_pass)
    if chance == 1:
        return None
    return (observed - chance) / (1 - chance)


def kappa_interval(
    pairs: Sequence[Pair], resamples: int = 1000, seed: int = 0
) -> tuple[float, float] | None:
    """Return a 95% percentile-bootstrap interval of Cohen's kappa.

    Resamples where kappa is undefined are skipped.

    Args:
        pairs: ``(label, verdict)`` pairs.
        resamples: Number of bootstrap resamples.
        seed: Seed of the resampling, so a report is reproducible.

    Returns:
        ``(low, high)``, or ``None`` if kappa is undefined on ``pairs`` or on every
        resample.
    """
    if cohen_kappa(pairs) is None:
        return None
    rng = random.Random(seed)
    values = sorted(
        kappa
        for _ in range(resamples)
        if (kappa := cohen_kappa(rng.choices(pairs, k=len(pairs)))) is not None
    )
    if not values:
        return None
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


__all__ = ["Pair", "cohen_kappa", "kappa_interval", "wilson"]
