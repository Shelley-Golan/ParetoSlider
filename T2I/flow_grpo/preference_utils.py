"""
Preference sampling utilities for multi-objective conditioning.

Provides deterministic, rank-consistent preference assignment for GRPO training.
Ported from ParetoNFT's train_cond_nft_sd3.py.
"""

import hashlib
import numpy as np
import torch


def stable_prompt_hash_u32(prompt: str) -> int:
    """Stable hash across processes/ranks (unlike Python's built-in hash())."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _sample_structured_preference(num_rewards, rng, pref_min_weight=0.0):
    """Sample a preference vector using structured vertex/edge/interior sampling.

    For 3+ objectives, uniform simplex sampling (Dirichlet) puts non-zero weight
    on all objectives, allowing whichever has strongest signal to dominate.

    Instead, we sample preferences with exact zeros:
      - 50% VERTEX (one-hot): optimize exactly one objective
      - 35% EDGE (pair): pairwise trade-off
      - 15% INTERIOR (full simplex): smooth interpolation at inference

    Args:
        pref_min_weight: Minimum weight for any objective (clamped then renormalized).
            Prevents identity collapse by ensuring no objective is ever fully ignored.
    """
    if num_rewards == 1:
        return np.ones(1, dtype=np.float32)
    if num_rewards == 2:
        # Pure Dirichlet — no corner (one-hot) sampling to avoid identity collapse
        raw = rng.dirichlet([1.0, 1.0]).astype(np.float32)
        if pref_min_weight > 0:
            raw = np.clip(raw, pref_min_weight, 1.0 - pref_min_weight)
            raw = raw / raw.sum()
        return raw

    roll = rng.random()

    if roll < 0.50:
        # VERTEX: one-hot
        pref = np.zeros(num_rewards, dtype=np.float32)
        pref[rng.integers(0, num_rewards)] = 1.0
    elif roll < 0.85:
        # EDGE: pair of objectives
        pref = np.zeros(num_rewards, dtype=np.float32)
        pair = rng.choice(num_rewards, size=2, replace=False)
        w = rng.dirichlet([1.0, 1.0]).astype(np.float32)
        pref[pair[0]] = w[0]
        pref[pair[1]] = w[1]
    else:
        # INTERIOR: all objectives
        pref = rng.dirichlet([1.0] * num_rewards).astype(np.float32)

    return pref


def build_consistent_preferences_for_global_batch(
    prompts_local, num_rewards, device, epoch, batch_idx, base_seed,
    num_pref_per_prompt=1, pref_min_weight=0.0,
):
    """
    Assigns preference vectors to samples.

    When num_pref_per_prompt=1: one preference per unique prompt per batch,
    broadcast to all repeats (original GDPO design).

    When num_pref_per_prompt>1: K distinct preferences per prompt, cycled
    across repeats. Each (prompt, pref_slot) sub-group shares the same
    preference, giving the projector contrastive signal on the same content.
    GDPO should normalize per (prompt, pref_slot) — use the returned
    pref_slots to construct composite group keys.

    Returns:
        preferences: Tensor (B, num_rewards)
        pref_slots: Tensor (B,) int — which preference slot each sample belongs to.
            For num_pref_per_prompt=1, all zeros.
    """
    B = len(prompts_local)

    if num_rewards == 1:
        return (
            torch.ones((B, 1), dtype=torch.float32, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )

    if num_pref_per_prompt <= 1:
        # Original behavior: one preference per prompt per batch
        unique_prompts = sorted(set(prompts_local))
        prompt_to_pref = {}

        for prompt in unique_prompts:
            prompt_hash = stable_prompt_hash_u32(prompt)
            step_offset = epoch * 100000 + batch_idx
            seed = (base_seed + prompt_hash + step_offset) % (2**32)

            rng = np.random.default_rng(seed)
            raw_weights = _sample_structured_preference(num_rewards, rng, pref_min_weight)

            prompt_to_pref[prompt] = torch.tensor(
                raw_weights, dtype=torch.float32, device=device
            )

        prefs = [prompt_to_pref[p] for p in prompts_local]
        return (
            torch.stack(prefs, dim=0),
            torch.zeros(B, dtype=torch.long, device=device),
        )

    # Multi-preference: K distinct preferences per prompt.
    # Seeds use (prompt, epoch, slot) — NOT batch_idx — so the same K
    # preferences are assigned consistently regardless of which batch a
    # repeat lands in after shuffling.
    unique_prompts = sorted(set(prompts_local))
    prompt_to_prefs = {}

    for prompt in unique_prompts:
        prompt_hash = stable_prompt_hash_u32(prompt)
        slot_prefs = []
        for k in range(num_pref_per_prompt):
            seed = (base_seed + prompt_hash + epoch * 100000 + k * 7919) % (2**32)
            rng = np.random.default_rng(seed)
            raw_weights = _sample_structured_preference(num_rewards, rng, pref_min_weight)
            slot_prefs.append(
                torch.tensor(raw_weights, dtype=torch.float32, device=device)
            )
        prompt_to_prefs[prompt] = slot_prefs

    # Cycle through K preference slots by occurrence index within this batch.
    prompt_occurrence = {}
    prefs = []
    slots = []
    for p in prompts_local:
        occ = prompt_occurrence.get(p, 0)
        slot = occ % num_pref_per_prompt
        prefs.append(prompt_to_prefs[p][slot])
        slots.append(slot)
        prompt_occurrence[p] = occ + 1

    return (
        torch.stack(prefs, dim=0),
        torch.tensor(slots, dtype=torch.long, device=device),
    )


def compute_per_objective_advantages_gdpo(prompts, reward_vec):
    """
    Compute per-objective GDPO advantages: per-group, per-channel normalization.

    Args:
        prompts: list of prompt strings (N,)
        reward_vec: numpy array (N, R) of raw reward values

    Returns:
        advantages_per_obj: numpy array (N, R) of normalized per-objective advantages
    """
    prompts = np.array(prompts, dtype=object)
    reward_vec = np.asarray(reward_vec, dtype=np.float64)
    N, R = reward_vec.shape
    unique_prompts = np.unique(prompts)
    advantages_per_channel = np.zeros_like(reward_vec)

    # Per-group, per-channel normalization (standard GDPO)
    for prompt in unique_prompts:
        mask = prompts == prompt
        group_rewards = reward_vec[mask]  # (K, R)

        for r in range(R):
            channel_rewards = group_rewards[:, r]  # (K,)
            mean = np.mean(channel_rewards)
            std = np.std(channel_rewards) + 1e-4
            advantages_per_channel[mask, r] = (reward_vec[mask, r] - mean) / std

    return advantages_per_channel.astype(np.float32)
