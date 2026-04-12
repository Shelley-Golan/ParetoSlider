# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import sys
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np

# Ensure we import the local `flow_grpo` package from this repo (and not a different checkout
# that might also be on PYTHONPATH).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptAndPreferenceStatTracker
from flow_grpo.scalarization import make_scalarizer
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.rewards import expand_reward_fn_to_objectives
from flow_grpo.diffusers_patch.transformer_sd3 import SD3Transformer2DModelWithConditioning, get_modules_to_save
from flow_grpo.diffusers_patch.transformer_ablations import (
    ABLATION_MODES as _ABLATION_MODES,
    SD3AblationTransformer,
    get_ablation_modules_to_save,
)
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.amp import GradScaler, autocast as torch_autocast
import hashlib

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["objective"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def stable_prompt_hash_u32(prompt: str) -> int:
    # Stable across processes/ranks (unlike Python's built-in hash()).
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def build_consistent_preferences_for_global_batch(prompts_local, num_rewards, device, epoch, batch_idx, base_seed):
    """
    Assigns ONE random preference vector per unique prompt, broadcast to all repeats.
    Ensures GDPO groups (repeats of same prompt) share the same goal.
    
    This fixes the singleton group bug where stratified sampling assigned different
    preferences to each repeat, making GDPO normalization invalid (group size = 1).
    
    Args:
        prompts_local: List of prompts (may have repeats)
        num_rewards: Number of reward objectives
        device: torch device
        epoch: Current epoch (for deterministic seeding)
        batch_idx: Current batch index (for deterministic seeding)
        base_seed: Base seed from config
    
    Returns:
        Tensor of shape (B, num_rewards) where repeats of same prompt have identical values
    """
    B = len(prompts_local)
    
    if num_rewards == 1:
        return torch.ones((B, 1), dtype=torch.float32, device=device)
    
    # Build a mapping from unique prompts to their preference vectors
    # Use deterministic seeding based on prompt + epoch + batch for reproducibility across ranks
    unique_prompts = sorted(set(prompts_local))
    prompt_to_pref = {}
    
    for prompt in unique_prompts:
        # Create a deterministic seed from prompt content + training step
        # This ensures all ranks agree on the preference for each prompt
        prompt_hash = stable_prompt_hash_u32(prompt)
        step_offset = epoch * 100000 + batch_idx
        seed = (base_seed + prompt_hash + step_offset) % (2**32)
        
        # Use numpy RNG to avoid affecting global torch RNG state
        rng = np.random.default_rng(seed)
        
        # Sample from Dirichlet(1, 1, ...) which gives uniform distribution on simplex
        # alpha=1.0 gives uniform; could use higher alpha to concentrate around center
        dirichlet_alpha = 1.0
        raw_weights = rng.dirichlet([dirichlet_alpha] * num_rewards).astype(np.float32)
        
        prompt_to_pref[prompt] = torch.tensor(raw_weights, dtype=torch.float32, device=device)
    
    # Broadcast the preference to all repeats of each prompt
    prefs = [prompt_to_pref[p] for p in prompts_local]
    return torch.stack(prefs, dim=0)


def _sample_structured_preference(num_rewards, rng):
    """Sample a preference vector using structured vertex/edge/interior sampling.

    For 3+ objectives, uniform simplex sampling (Dirichlet) puts non-zero weight
    on all objectives. Instead, sample with exact zeros for better coverage:
      - 50% VERTEX (one-hot): optimize exactly one objective
      - 35% EDGE (pair): pairwise trade-off
      - 15% INTERIOR (full simplex): smooth interpolation
    """
    if num_rewards == 1:
        return np.ones(1, dtype=np.float32)
    if num_rewards == 2:
        # 10% corners (one-hot) + 90% Dirichlet (smooth blend)
        if rng.random() < 0.1:
            pref = np.zeros(2, dtype=np.float32)
            pref[rng.integers(0, 2)] = 1.0
            return pref
        return rng.dirichlet([1.0, 1.0]).astype(np.float32)

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


def build_preferences_with_subgroups(
    prompts_local, num_rewards, device, epoch, batch_idx, base_seed,
    num_pref_per_prompt,
):
    """
    Assigns K distinct preference vectors per prompt, cycled across repeats.

    Each (prompt, pref_slot) sub-group shares the same preference, so GDPO
    i.i.d. assumption is preserved within each sub-group. Different sub-groups
    of the same prompt get different preferences, training multiple Pareto
    front points per prompt per batch.

    Use the returned pref_slots to construct composite GDPO group keys:
        gdpo_key = f"{prompt}__pref{slot}"

    Args:
        prompts_local: List of prompts (may have repeats)
        num_rewards: Number of reward objectives
        device: torch device
        epoch: Current epoch (for deterministic seeding)
        batch_idx: Current batch index (for deterministic seeding)
        base_seed: Base seed from config
        num_pref_per_prompt: K distinct preferences per prompt

    Returns:
        preferences: Tensor (B, num_rewards)
        pref_slots: Tensor (B,) int — which preference slot each sample belongs to
    """
    B = len(prompts_local)

    if num_rewards == 1:
        return (
            torch.ones((B, 1), dtype=torch.float32, device=device),
            torch.zeros(B, dtype=torch.long, device=device),
        )

    # Sample K distinct preferences per unique prompt.
    # Seeds use (prompt, epoch, slot) — NOT batch_idx — so the same K
    # preferences are assigned consistently regardless of batch ordering.
    unique_prompts = sorted(set(prompts_local))
    prompt_to_prefs = {}

    for prompt in unique_prompts:
        prompt_hash = stable_prompt_hash_u32(prompt)
        slot_prefs = []
        for k in range(num_pref_per_prompt):
            seed = (base_seed + prompt_hash + epoch * 100000 + k * 7919) % (2**32)
            rng = np.random.default_rng(seed)
            raw_weights = _sample_structured_preference(num_rewards, rng)
            slot_prefs.append(torch.tensor(raw_weights, dtype=torch.float32, device=device))
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


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    # Use expanded objective count for multi-output scorers
    expanded_objectives = expand_reward_fn_to_objectives(config.reward_fn)
    num_rewards = len(expanded_objectives)

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        eval_preferences = torch.full((current_batch_size, num_rewards), 1/num_rewards, device=device)

        with torch_autocast("cuda", enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                    preference=eval_preferences,
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            # Skip nested dict entries (e.g. weighted_scores) - only gather numeric arrays
            if isinstance(value, dict):
                continue
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        # Save a small metadata file to make resuming/debugging less error-prone.
        try:
            meta = {
                "pretrained_model": getattr(config.pretrained, "model", None),
                "use_lora": bool(getattr(config, "use_lora", False)),
                "reward_fn_keys": list(getattr(config, "reward_fn", {}).keys()),
                "conditioning_mode": getattr(config, "conditioning_mode", "hybrid"),
                "pref_dim": len(expand_reward_fn_to_objectives(config.reward_fn)),
                "objective_names": [name for name, _ in expand_reward_fn_to_objectives(config.reward_fn)],
            }
            # Save ablation-specific params for eval loading
            _cm = meta["conditioning_mode"]
            if _cm in _ABLATION_MODES:
                meta["block_mod_form"] = getattr(config, "block_mod_form", "affine")
                meta["use_pooled_text"] = bool(getattr(config, "use_pooled_text", False))
                meta["num_freqs"] = int(getattr(config, "num_freqs", 64))
                meta["mod_block_fraction"] = float(getattr(config, "mod_block_fraction", 0.667))
            # Attempt to infer a representative hidden size (best-effort).
            for k, v in model_to_save.state_dict().items():
                if k.endswith("attn.to_q.weight") and hasattr(v, "shape") and len(v.shape) == 2:
                    meta["transformer_hidden_size"] = int(v.shape[0])
                    break
            with open(os.path.join(save_root, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.warning(f"Failed to write checkpoint metadata.json: {e}")

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="pareto-slider", name=config.run_name, config=config.to_dict(), dir=log_dir)
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler("cuda", enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    # --- build conditioned transformer and copy weights ---
    base_transformer = pipeline.transformer

    # Extract config as dict and instantiate directly (not via from_config)
    # to ensure all architecture parameters (especially hidden size) are preserved.
    base_config_dict = dict(base_transformer.config)
    # Remove keys that are not valid __init__ arguments for the model class
    base_config_dict.pop("_class_name", None)
    base_config_dict.pop("_diffusers_version", None)
    base_config_dict.pop("_name_or_path", None)

    # Determine pref_dim: use override if specified (for multi-output scorers), otherwise use expanded objective count
    expanded_objectives_for_pref = expand_reward_fn_to_objectives(config.reward_fn)
    pref_dim = getattr(config, 'pref_dim_override', None) or len(expanded_objectives_for_pref)
    
    conditioning_mode = getattr(config, "conditioning_mode", "hybrid")
    if conditioning_mode in _ABLATION_MODES:
        cond_transformer = SD3AblationTransformer(
            **base_config_dict,
            pref_dim=pref_dim,
            pref_gate_init=getattr(config, 'pref_gate_init', 1e-3),
            conditioning_mode=conditioning_mode,
            block_mod_form=getattr(config, 'block_mod_form', 'affine'),
            use_pooled_text=getattr(config, 'use_pooled_text', False),
            num_freqs=getattr(config, 'num_freqs', 64),
            mod_block_fraction=getattr(config, 'mod_block_fraction', 0.667),
        )
    else:
        cond_transformer = SD3Transformer2DModelWithConditioning(
            **base_config_dict,
            pref_dim=pref_dim,  # Length = number of reward objectives (may be expanded for multi-output scorers)
            pref_gate_init=getattr(config, 'pref_gate_init', 1e-3),
            conditioning_mode=conditioning_mode,
        )

    # load original weights; preference modules will be missing and that's fine
    missing, unexpected = cond_transformer.load_state_dict(base_transformer.state_dict(), strict=False)
    if missing:
        logger.info(f"Loaded base transformer weights with missing keys (expected for pref modules): {missing}")

    cond_transformer = cond_transformer.to(device)

    # IMPORTANT: pipeline must now use the conditioned model
    pipeline.transformer = cond_transformer

    transformer = pipeline.transformer

    pipeline.transformer.requires_grad_(not config.use_lora)

    # Enable gradient checkpointing to save massive amounts of VRAM
    transformer.enable_gradient_checkpointing()

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        lora_rank = getattr(config.train, "lora_rank", 32)
        lora_alpha = getattr(config.train, "lora_alpha", lora_rank * 2)
        if conditioning_mode in _ABLATION_MODES:
            modules_to_save = get_ablation_modules_to_save(conditioning_mode)
        else:
            modules_to_save = get_modules_to_save(conditioning_mode)
        transformer_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        # Always create PEFT model with our config (which includes modules_to_save for pref_*)
        transformer = get_peft_model(transformer, transformer_lora_config)
        
        if config.train.lora_path:
            # Load LoRA weights from checkpoint (may be from non-conditioned model without pref_* modules)
            # The saved checkpoint has keys WITHOUT adapter name (e.g., "...lora_A.weight")
            # But the PEFT model expects keys WITH adapter name (e.g., "...lora_A.default.weight")
            # We need to remap the keys to include the adapter name
            import safetensors.torch
            lora_weights_path = os.path.join(config.train.lora_path, "adapter_model.safetensors")
            if os.path.exists(lora_weights_path):
                lora_state_dict = safetensors.torch.load_file(lora_weights_path)
                
                # Remap keys: insert ".default" before ".weight" for lora_A and lora_B
                remapped_state_dict = {}
                for key, value in lora_state_dict.items():
                    if ".lora_A.weight" in key:
                        new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
                    elif ".lora_B.weight" in key:
                        new_key = key.replace(".lora_B.weight", ".lora_B.default.weight")
                    else:
                        new_key = key
                    remapped_state_dict[new_key] = value
                
                missing, unexpected = transformer.load_state_dict(remapped_state_dict, strict=False)
                logger.info(f"Loaded LoRA weights from {config.train.lora_path}")
                logger.info(f"  Remapped {len(lora_state_dict)} keys to include adapter name 'default'")
                logger.info(f"  Missing keys (expected for pref_* modules): {len(missing)}")
                logger.info(f"  Unexpected keys: {len(unexpected)}")
                
                # Verify LoRA weights were actually loaded
                if unexpected:
                    logger.warning(f"  WARNING: {len(unexpected)} unexpected keys - check key format!")
                
                # Create frozen "ref" adapter initialized with checkpoint-51 weights
                # This will be used as the KL reference model instead of base SD3
                transformer.add_adapter("ref", transformer_lora_config)
                
                # Remap keys for "ref" adapter (replace .default. with .ref.)
                ref_state_dict = {}
                for key, value in remapped_state_dict.items():
                    ref_key = key.replace(".default.", ".ref.")
                    ref_state_dict[ref_key] = value
                
                missing_ref, unexpected_ref = transformer.load_state_dict(ref_state_dict, strict=False)
                logger.info(f"Created 'ref' adapter with checkpoint-51 weights for KL reference")
                logger.info(f"  Missing keys: {len(missing_ref)}, Unexpected: {len(unexpected_ref)}")
                
                # Freeze ref adapter parameters
                for name, param in transformer.named_parameters():
                    if ".ref." in name:
                        param.requires_grad = False
                logger.info(f"  Froze 'ref' adapter parameters")
                
            else:
                raise FileNotFoundError(f"LoRA weights not found at {lora_weights_path}")
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    pipeline.transformer = transformer
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")
    pipeline.transformer = transformer_ddp.module

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
        if getattr(config, "loss_mode", "single_loss") == "per_objective":
            raise ValueError(
                "loss_mode='per_objective' requires per_prompt_stat_tracking=True "
                "(num_image_per_prompt must be > 1)"
            )

    num_pref_per_prompt = getattr(config, "num_pref_per_prompt", 1)
    if num_pref_per_prompt > 1:
        repeats = config.sample.num_image_per_prompt
        if repeats % num_pref_per_prompt != 0:
            raise ValueError(
                f"num_image_per_prompt ({repeats}) must be divisible by "
                f"num_pref_per_prompt ({num_pref_per_prompt})"
            )
        sub_group_size = repeats // num_pref_per_prompt
        if sub_group_size < 2:
            raise ValueError(
                f"Sub-group size {sub_group_size} too small (need >= 2). "
                f"Reduce num_pref_per_prompt or increase num_image_per_prompt."
            )
        if is_main_process(rank):
            logger.info(
                f"Multi-preference: {num_pref_per_prompt} preferences per prompt, "
                f"{sub_group_size} repeats per sub-group"
            )

    """if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)"""
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptAndPreferenceStatTracker(config.sample.global_std)
    else:
        stat_tracker = None

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score_conditioned")(device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score_conditioned")(device, config.reward_fn)

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            transformer_ddp.module.load_adapter(lora_path, adapter_name="old", is_trainable=False)
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                # By default we *refuse* to load a full transformer state dict when LoRA is enabled,
                # because it’s easy to accidentally point to a checkpoint from a different base model.
                resume_allow_full_model = bool(getattr(config, "resume_allow_full_model", False))
                if bool(getattr(config, "use_lora", False)) and not resume_allow_full_model:
                    raise ValueError(
                        "Checkpoint resume failed: found 'transformer_model.pt' but no 'lora/' folder. "
                        "This training script expects LoRA checkpoints under '<checkpoint>/lora'. "
                        "If you intentionally want to load a full transformer state dict, set "
                        "`config.resume_allow_full_model = True` (and ensure it matches the same base model)."
                    )

                ckpt_state = torch.load(model_ckpt_path, map_location="cpu")
                model_state = transformer_ddp.module.state_dict()
                # Detect and explain shape mismatches (e.g., hidden size 1536 vs 1152).
                for k, v in ckpt_state.items():
                    if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                        if tuple(v.shape) != tuple(model_state[k].shape):
                            raise ValueError(
                                "Checkpoint/model architecture mismatch while loading transformer weights.\n"
                                f"- checkpoint param: {k} has shape {tuple(v.shape)}\n"
                                f"- current model:   {k} has shape {tuple(model_state[k].shape)}\n"
                                f"- current pretrained model: {getattr(config.pretrained, 'model', None)}\n"
                                "This usually means the checkpoint was trained from a different SD3 variant "
                                "(different hidden size). Use a checkpoint produced from the same base model, "
                                "or resume from a LoRA checkpoint directory that contains 'lora/'."
                            )
                transformer_ddp.module.load_state_dict(ckpt_state, strict=True)

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Extract epoch and step from checkpoint name, e.g., "checkpoint-1000" -> global_step = 1000
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(f"Resumed global_step to {global_step}. Epoch estimation might be needed.")
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param



    # Build scalarizer from config (defaults to linear if not specified)
    _scalarize = make_scalarizer(config)
    if getattr(config, "scalarization", "linear") != "linear":
        logging.info(f"Using scalarization method: {config.scalarization}")

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []
        # Expand reward_fn to handle multi-output scorers (e.g., sketch_6obj_decomposed -> 6 objectives)
        expanded_objectives = expand_reward_fn_to_objectives(config.reward_fn)
        reward_names = [name for name, _ in expanded_objectives]
        weight_vec = torch.tensor([float(weight) for _, weight in expanded_objectives], device=device, dtype=torch.float32)

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            """# Sample preference vectors on simplex ONCE per (batch, prompt), deterministic across ranks.
            num_rewards = len(reward_names)
            batch_prompt_to_pref = {}
            preferences_list = []
            
            # 5.0 concentrates on the center (solving the "Curve" issue).
            # 1.0 would be uniform.
            dirichlet_alpha = 1.0 
            
            # Probability of sampling a "Soft Corner" (focus on one objective)
            corner_prob = 0.2

            for prompt in prompts:
                if prompt not in batch_prompt_to_pref:
                    # Deterministic seed per prompt+batch, rank-independent logic
                    seed_u32 = (int(config.seed) + int(epoch) * 1000003 + int(i) * 1009 + stable_prompt_hash_u32(prompt)) % (2**32)
                    
                    # SAFE FIX: Use numpy's seeded RNG. Does NOT touch global torch RNG state.
                    seed_u32 = int(seed_u32) % (2**32)
                    rng = np.random.default_rng(seed_u32)
                    
                    if num_rewards == 1:
                        pref = torch.ones((1,), dtype=torch.float32)
                    elif rng.random() < corner_prob:
                        # "Soft Corner": Focus on one reward, but don't go to 1.0 (anti-hacking)
                        idx = int(rng.integers(0, num_rewards))
                        # Distribute small weight to others, large weight to target
                        pref = torch.full((num_rewards,), 0.05 / (num_rewards - 1), dtype=torch.float32)
                        pref[idx] = 0.95
                    else:
                        # "Curve Fix": Gamma(alpha, 1) normalized -> Dirichlet(alpha)
                        # strictly uses local numpy rng
                        gamma_samples = rng.gamma(dirichlet_alpha, 1.0, size=num_rewards).astype(np.float32)
                        gamma_samples = torch.from_numpy(gamma_samples)
                        pref = gamma_samples / gamma_samples.sum()
                    
                    batch_prompt_to_pref[prompt] = pref.to(device=device, dtype=torch.float32)
                
                preferences_list.append(batch_prompt_to_pref[prompt])
            preferences = torch.stack(preferences_list, dim=0)  # (batch_size, num_rewards)
            """

            num_rewards = len(reward_names)
            num_pref_per_prompt = getattr(config, "num_pref_per_prompt", 1)
            if num_pref_per_prompt > 1:
                preferences, pref_slots = build_preferences_with_subgroups(
                    prompts_local=prompts,
                    num_rewards=num_rewards,
                    device=device,
                    epoch=epoch,
                    batch_idx=i,
                    base_seed=config.seed,
                    num_pref_per_prompt=num_pref_per_prompt,
                )
            else:
                preferences = build_consistent_preferences_for_global_batch(
                    prompts_local=prompts,
                    num_rewards=num_rewards,
                    device=device,
                    epoch=epoch,
                    batch_idx=i,
                    base_seed=config.seed,
                )
                pref_slots = torch.zeros(len(prompts), dtype=torch.long, device=device)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if i == 0 and epoch % config.save_freq == 0 and is_main_process(rank) and not config.debug:
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                        preference=preferences,
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "preferences": preferences,
                    "pref_slots": pref_slots,
                    "rewards_future": rewards_future,  # Store future
                }
            )

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            # Store reward vectors in a deterministic order matching config.reward_fn keys.
            if isinstance(rewards, dict):
                # Validate required keys exist and are non-nested.
                missing = [name for name in reward_names if name not in rewards]
                if missing:
                    raise KeyError(
                        f"Reward fn returned dict missing required keys {missing}. "
                        f"Expected at least {reward_names}. Got keys={list(rewards.keys())}"
                    )
                for name in reward_names:
                    if isinstance(rewards[name], dict):
                        raise TypeError(
                            f"Reward fn returned nested dict for key '{name}', expected a numeric vector. "
                            f"Got type={type(rewards[name])}"
                        )
                reward_vec = torch.stack(
                    [torch.as_tensor(rewards[name], device=device).float() for name in reward_names],
                    dim=1,
                )
                # Tensorize only leaf numeric entries; keep known nested structures separately.
                nested_keys = [k for k, v in rewards.items() if isinstance(v, dict)]
                unexpected_nested = [k for k in nested_keys if k != "weighted_scores"]
                if unexpected_nested:
                    raise TypeError(
                        f"Reward fn returned unexpected nested dict keys {unexpected_nested}. "
                        f"Only 'weighted_scores' is allowed to be nested. Got keys={list(rewards.keys())}"
                    )
                reward_dict = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items() if not isinstance(v, dict)}
                if "weighted_scores" in rewards:
                    # Convert weighted_scores values to tensors
                    sample_item["weighted_scores"] = {
                        k: torch.as_tensor(v, device=device).float() 
                        for k, v in rewards["weighted_scores"].items()
                    }
            else:
                reward_vec = torch.as_tensor(rewards, device=device).float()
                reward_dict = {}
            sample_item["reward_vec"] = reward_vec
            sample_item["rewards"] = reward_dict
            del sample_item["rewards_future"]

        # Collate samples
        def _to_tensor(x):
            """Ensure x is a tensor for concatenation."""
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x, device=device).float()
        
        collated_samples = {
            k: (
                torch.cat([_to_tensor(s[k]) for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([_to_tensor(s[k][sk]) for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # Keep raw rewards for logging
        collated_samples["reward_vec_raw"] = collated_samples["reward_vec"].clone()

        # Compute scalar objective locally for logging (objective = dot(p, w * r_vec)).
        objective_local = (collated_samples["preferences"] * collated_samples["reward_vec"] * weight_vec[None, :]).sum(dim=1)

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = objective_local[-len(images_to_log) :].detach().cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )

        # Gather normalized reward vectors + preferences across processes (needed to compute objective consistently).
        gathered_reward_vec_raw = gather_tensor_to_all(collated_samples["reward_vec_raw"], world_size).numpy()
        gathered_preferences = gather_tensor_to_all(collated_samples["preferences"], world_size).numpy()
        
        # In GDPO, we normalize per group (per prompt) inside the tracker, so we don't need global normalization here.
        gathered_reward_vec = gathered_reward_vec_raw

        if is_main_process(rank):  # logging
            # Log objective using RAW rewards (or we could normalize per batch for logging if desired, but raw is fine)
            objective_all = (gathered_preferences * gathered_reward_vec * weight_vec.detach().cpu().numpy()[None, :]).sum(axis=-1)
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{name}": gathered_reward_vec_raw[:, j].mean()  # Log raw rewards for interpretability
                        for j, name in enumerate(reward_names)
                    },
                    "reward_objective": float(objective_all.mean()),
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Compute objective and advantages inside tracker (pure arithmetic).
            # GDPO: Use update_from_reward_vectors_gdpo and pass raw rewards (tracker normalizes group-wise)
            use_per_obj = getattr(config, "loss_mode", "single_loss") == "per_objective"

            # Build GDPO group keys: (prompt, pref_slot) when using multi-preference.
            # This ensures each sub-group has uniform conditioning, so advantages
            # reflect stochastic quality only (not systematic preference effects).
            num_pref_per_prompt = getattr(config, "num_pref_per_prompt", 1)
            if num_pref_per_prompt > 1:
                pref_slots_all = gather_tensor_to_all(
                    collated_samples["pref_slots"], world_size
                ).numpy()
                gdpo_group_keys = [
                    f"{p}__pref{int(s)}" for p, s in zip(prompts_all_decoded, pref_slots_all)
                ]
            else:
                gdpo_group_keys = prompts_all_decoded

            result = stat_tracker.update_from_reward_vectors_gdpo(
                gdpo_group_keys,
                reward_vec=gathered_reward_vec_raw,
                preferences=gathered_preferences,
                weights=weight_vec.detach().cpu().numpy(),
                use_per_objective_loss=use_per_obj,
            )
            advantages_result, objective_scalar = result[0], result[1]
            objective_broadcast = np.repeat(objective_scalar[:, None], num_train_timesteps, axis=1)
            if use_per_obj:
                # advantages_result is (N, R) — broadcast to (N, R, T)
                advantages_per_obj_all = np.repeat(advantages_result[:, :, None], num_train_timesteps, axis=2)
                # Also compute scalar advantages for logging
                scalar_adv = (gathered_preferences * advantages_result).sum(axis=-1)
                scalar_adv = (scalar_adv - scalar_adv.mean()) / (scalar_adv.std() + 1e-4)
                advantages = np.repeat(scalar_adv[:, None], num_train_timesteps, axis=1)
            else:
                advantages = np.repeat(advantages_result[:, None], num_train_timesteps, axis=1)

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                gathered_obj_dict = {"objective": objective_broadcast}
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_obj_dict)
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            objective_all = (gathered_preferences * gathered_reward_vec * weight_vec.detach().cpu().numpy()[None, :]).sum(axis=-1)
            objective_broadcast = np.repeat(objective_all[:, None], num_train_timesteps, axis=1)
            advantages = (objective_broadcast - objective_broadcast.mean()) / (objective_broadcast.std() + 1e-4)
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        # Distribute per-objective advantages if in per_objective mode
        if getattr(config, "loss_mode", "single_loss") == "per_objective" and config.per_prompt_stat_tracking:
            collated_samples["advantages_per_obj"] = torch.from_numpy(
                advantages_per_obj_all.reshape(world_size, samples_per_gpu, num_rewards, -1)[rank]
            ).to(device)

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        del collated_samples["rewards"]
        del collated_samples["reward_vec"]
        del collated_samples["reward_vec_raw"]
        del collated_samples["prompt_ids"]
        if "pref_slots" in collated_samples:
            del collated_samples["pref_slots"]
        # Remove nested dict if present (can't be indexed with tensor)
        if "weighted_scores" in collated_samples:
            del collated_samples["weighted_scores"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                                preference=train_sample_batch["preferences"],
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                            preference=train_sample_batch["preferences"],
                        )[0]

                        with torch.no_grad():  # Reference model part
                            if config.use_lora:
                                if hasattr(config.train, 'lora_path') and config.train.lora_path:
                                    # Use checkpoint-51 as KL reference (frozen "ref" adapter)
                                    transformer_ddp.module.set_adapter("ref")
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                        preference=None,  # No preference conditioning for reference
                                    )[0]
                                    transformer_ddp.module.set_adapter("default")
                                else:
                                    # No checkpoint loaded - use base SD3 (disable adapter)
                                    with transformer_ddp.module.disable_adapter():
                                        ref_forward_prediction = transformer_ddp(
                                            hidden_states=xt,
                                            timestep=train_sample_batch["timesteps"][:, j_idx],
                                            encoder_hidden_states=embeds,
                                            pooled_projections=pooled_embeds,
                                            return_dict=False,
                                            preference=None,
                                        )[0]
                                    transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                raise ValueError("Full-model training is not supported here; set config.use_lora=True")
                    loss_terms = {}
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()

                    # Shared: positive/negative predictions and losses
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    if getattr(config, "loss_mode", "single_loss") == "per_objective" and "advantages_per_obj" in train_sample_batch:
                        # Per-objective NFT: apply NFT mapping per objective, then scalarize with preferences
                        adv_per_obj = train_sample_batch["advantages_per_obj"][:, :, j_idx]  # (B, R)
                        pref_batch = train_sample_batch["preferences"]  # (B, R)
                        per_obj_losses = []
                        for r_idx in range(num_rewards):
                            adv_r = torch.clamp(adv_per_obj[:, r_idx], -config.train.adv_clip_max, config.train.adv_clip_max)
                            r_val = 0.5 + 0.5 * torch.clamp(adv_r / config.train.adv_clip_max, -1.0, 1.0)
                            loss_r = r_val * positive_loss / config.beta + (1.0 - r_val) * negative_loss / config.beta
                            per_obj_losses.append(loss_r)
                        per_obj_losses = torch.stack(per_obj_losses, dim=1)  # (B, R)
                        weighted_loss = _scalarize(pref_batch, per_obj_losses)  # (B,)
                        policy_loss = (weighted_loss * config.train.adv_clip_max).mean()
                        loss = policy_loss
                        loss_terms["policy_loss"] = policy_loss.detach()
                        loss_terms["unweighted_policy_loss"] = weighted_loss.mean().detach()
                        for r_idx, rn in enumerate(reward_names):
                            loss_terms[f"policy_loss_{rn}"] = (per_obj_losses[:, r_idx] * config.train.adv_clip_max).mean().detach()
                    else:
                        # Single-loss: scalarize advantages first, then one NFT loss
                        advantages_clip = torch.clamp(
                            train_sample_batch["advantages"][:, j_idx],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        if hasattr(config.train, "adv_mode"):
                            if config.train.adv_mode == "positive_only":
                                advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                            elif config.train.adv_mode == "negative_only":
                                advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                            elif config.train.adv_mode == "one_only":
                                advantages_clip = torch.where(
                                    advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                                )
                            elif config.train.adv_mode == "binary":
                                advantages_clip = torch.sign(advantages_clip)
                        r = 0.5 + 0.5 * torch.clamp(advantages_clip / config.train.adv_clip_max, -1.0, 1.0)
                        ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                        policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()
                        loss = policy_loss
                        loss_terms["policy_loss"] = policy_loss.detach()
                        loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps

                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb_dict = {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                            }

                            wandb.log(wandb_dict)

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
    