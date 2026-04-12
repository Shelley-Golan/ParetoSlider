"""Scalarization methods for multi-objective Pareto-set learning.

Each function has signature:
    (pref: Tensor[B, R], losses: Tensor[B, R], active_mask: Tensor[B, R] | None) -> Tensor[B]

The factory ``make_scalarizer(config)`` reads config attributes and returns a
``functools.partial``-bound callable with the unified signature above.
"""

import functools
import torch

# Near-zero threshold for weight masking.  Consistent with the 1e-8 clamp in
# _apply_mask and the log-clamp in ew().  Anything below this is treated as
# "objective disabled" to prevent near-zero weights from leaking into
# logsumexp or log(w) computations.
_ZERO_TOL = 1e-7


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _apply_mask(pref, active_mask):
    """Zero masked objectives, renormalize surviving weights to preserve pref sum."""
    if active_mask is None:
        return pref
    w = pref * active_mask.float()
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return w / w_sum * pref.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Scalarization functions
# ---------------------------------------------------------------------------

def linear(pref, losses, active_mask=None):
    """Weighted linear scalarization."""
    w = _apply_mask(pref, active_mask)
    return (w * losses).sum(dim=1)

