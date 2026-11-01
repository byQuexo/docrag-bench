"""Uncertainty for benchmark numbers: bootstrap CIs over documents, paired McNemar.

* ``doc_bootstrap_ci``: questions about the same PDF are correlated (one bad parse
  fails all of them), so we resample DOCUMENTS, not questions ("cluster
  bootstrap"). Resampling questions would give intervals that are too narrow.
* ``mcnemar``: two configs answer the SAME questions, so only the questions where
  they disagree carry information. Exact binomial test on those.
"""

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def doc_bootstrap_ci(
    values: Sequence[float],
    docs: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean, lower, upper) of the per-question mean, resampling documents.

    NaN values (metric not applicable to that question) are dropped first.
    For a paired comparison pass per-question differences (b - a).
    """
    by_doc: defaultdict[str, list[float]] = defaultdict(list)
    for v, d in zip(values, docs, strict=True):
        if not math.isnan(v):
            by_doc[d].append(v)
    if not by_doc:
        return math.nan, math.nan, math.nan
    sums = np.array([sum(vs) for vs in by_doc.values()])
    counts = np.array([len(vs) for vs in by_doc.values()])
    mean = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    # Each row = one bootstrap replicate: which documents were drawn.
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """(wins, losses, exact two-sided p) of b vs a on paired outcomes.

    wins = questions b got right and a got wrong; losses = the reverse.
    """
    wins = sum(bb and not aa for aa, bb in zip(a, b, strict=True))
    losses = sum(aa and not bb for aa, bb in zip(a, b, strict=True))
    n, k = wins + losses, min(wins, losses)
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return wins, losses, min(1.0, 2 * tail)
