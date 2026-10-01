# SPDX-License-Identifier: Apache-2.0
"""Explicit-Gaussian sampling/log-prob for the rollout inference path.

rsl_rl's ``GaussianDistribution`` goes through ``torch.distributions.Normal``
for the two hot calls of every rollout step -- ``sample()`` and
``log_prob()``.  The distribution classes are Python-heavy and measured
~5.5 ms and ~3.8 ms per act call at 4096 envs on the XPU (the raw tensor
math underneath is ~0.1 ms).  This module swaps both methods for the
closed-form equivalent tensor expressions:

    a     = mean + std * randn_like(mean)          (Normal.rsample's formula)
    logp  = sum_d [ -0.5*((a-m)/s)^2 - log(s) - 0.5*log(2*pi) ]

Values are distributionally identical to ``Normal.sample``/``log_prob``
(same formula, same per-element rounding up to op order).  Both sides of
the PPO ratio use the same patched ``log_prob`` (the update recomputes
new-log-probs through the same method), so the surrogate stays internally
consistent.  The distribution object itself (and its Normal) is left in
place for the update path's KL/entropy machinery.

Kill switch: ``MJLAB_SYCL_ACT_FUSE=0``.
"""

from __future__ import annotations

import math
import os

_installed = False
_orig_sample = None
_orig_log_prob = None
_LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)


def _enabled() -> bool:
    return os.environ.get("MJLAB_SYCL_ACT_FUSE", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def install() -> None:
    global _installed
    if _installed:
        return
    if not _enabled():
        return
    try:
        import torch  # noqa: F401  (device-stack import must happen here,
        # not at module import: loading torch before warpsycl pins pip's
        # sycl8.dll by name and warpsycl then fails with WinError 127)
        from rsl_rl.modules.distribution import GaussianDistribution
    except Exception:
        return
    _installed = True
    global _orig_sample, _orig_log_prob
    _orig_sample = GaussianDistribution.sample
    _orig_log_prob = GaussianDistribution.log_prob

    def sample(self):
        # identical to Normal.rsample() for a diagonal Gaussian: the base
        # sample is a standard normal of the broadcast shape
        mean = self.mean
        std = self.std
        with torch.no_grad():
            return mean + std * torch.randn_like(mean)

    def log_prob(self, outputs):
        # Normal.log_prob(...).sum(-1) in closed form
        mean = self.mean
        std = self.std
        z = (outputs - mean) / std
        return (-0.5 * z * z - torch.log(std) - _LOG_2PI_HALF).sum(dim=-1)

    GaussianDistribution.sample = sample
    GaussianDistribution.log_prob = log_prob
    print(
        "[sycl-act-fuse] explicit Gaussian sample/log_prob installed "
        "(MJLAB_SYCL_ACT_FUSE=0 to disable)"
    )


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    from rsl_rl.modules.distribution import GaussianDistribution

    GaussianDistribution.sample = _orig_sample
    GaussianDistribution.log_prob = _orig_log_prob
    _installed = False
