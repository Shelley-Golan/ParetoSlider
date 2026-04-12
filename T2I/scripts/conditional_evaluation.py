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

import argparse
import os
import sys
import glob
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset, Subset
from peft import PeftModel

# Ensure we import the local `flow_grpo` package from this repo (and not a different checkout
# that might also be on PYTHONPATH).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flow_grpo.rewards import multi_score
from flow_grpo.diffusers_patch.transformer_sd3 import SD3Transformer2DModelWithConditioning
from flow_grpo.diffusers_patch.transformer_ablations import (
    ABLATION_MODES as _ABLATION_MODES,
    SD3AblationTransformer,
)
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict

import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def setup_distributed(rank, world_size):
    """Initializes the distributed process group."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def is_main_process(rank):
    """Checks if the current process is the main one (rank 0)."""
    return rank == 0


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "original_index": idx}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "original_index": idx}


def collate_fn(examples):
    prompts = [example["prompt"] for example in examples]
    metadatas = [example["metadata"] for example in examples]
    indices = [example["original_index"] for example in examples]
    return prompts, metadatas, indices


def _is_peft_adapter_dir(path: str) -> bool:
    """Check if directory contains a PEFT adapter (has adapter_config.json)."""
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def _load_state_dict_from_dir(path: str):
    """Load state dict from directory, trying safetensors first, then .bin."""
    safes = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if safes:
        from safetensors.torch import load_file
        return load_file(safes[0])
    bins = sorted(glob.glob(os.path.join(path, "*.bin")))
    if bins:
        return torch.load(bins[0], map_location="cpu")
    raise FileNotFoundError(f"No *.safetensors or *.bin found in: {path}")


def main(args):
    # --- Distributed Setup ---
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None

    if is_main_process(rank):
        print(f"Running distributed conditional evaluation with {world_size} GPUs.")
        if enable_amp:
            print(f"Using mixed precision: {args.mixed_precision}")
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model and Pipeline ---
    if is_main_process(rank):
        print("Loading model and pipeline...")

    if args.model_type == "sd3":
        pipeline = StableDiffusion3Pipeline.from_pretrained(args.base_model)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Read checkpoint metadata ---
    lora_path = os.path.join(args.checkpoint_path, "lora")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA directory not found at {lora_path}")

    conditioning_mode = args.conditioning_mode
    block_mod_form = args.block_mod_form
    use_pooled_text = args.use_pooled_text
    num_freqs = args.num_freqs
    mod_block_fraction = args.mod_block_fraction
    pref_dim = len(args.objective_names)

    metadata_path = os.path.join(args.checkpoint_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        pref_dim = int(meta.get("pref_dim", pref_dim))
        saved_mode = meta.get("conditioning_mode")
        if saved_mode and saved_mode != conditioning_mode:
            if is_main_process(rank):
                print(f"  Using conditioning_mode from metadata: '{saved_mode}' (CLI had '{conditioning_mode}')")
            conditioning_mode = saved_mode
        if "block_mod_form" in meta:
            block_mod_form = meta["block_mod_form"]
        if "use_pooled_text" in meta:
            use_pooled_text = bool(meta["use_pooled_text"])
        if "num_freqs" in meta:
            num_freqs = int(meta["num_freqs"])
        if "mod_block_fraction" in meta:
            mod_block_fraction = float(meta["mod_block_fraction"])
        if is_main_process(rank):
            print(f"Loaded metadata: pref_dim={pref_dim}, conditioning_mode={conditioning_mode}")

    # --- Create conditioned transformer ---
    base_config_dict = dict(pipeline.transformer.config)
    for key in ["_class_name", "_diffusers_version", "_name_or_path"]:
        base_config_dict.pop(key, None)

    if conditioning_mode in _ABLATION_MODES:
        if is_main_process(rank):
            print(f"Creating SD3AblationTransformer (pref_dim={pref_dim}, mode={conditioning_mode})")
        cond_transformer = SD3AblationTransformer(
            **base_config_dict,
            pref_dim=pref_dim,
            conditioning_mode=conditioning_mode,
            block_mod_form=block_mod_form,
            use_pooled_text=use_pooled_text,
            num_freqs=num_freqs,
            mod_block_fraction=mod_block_fraction,
        )
    else:
        if is_main_process(rank):
            print(f"Creating SD3Transformer2DModelWithConditioning (pref_dim={pref_dim}, mode={conditioning_mode})")
        cond_transformer = SD3Transformer2DModelWithConditioning(
            **base_config_dict,
            pref_dim=pref_dim,
            conditioning_mode=conditioning_mode,
        )

    # Load base SD3 weights (pref modules will be missing — expected)
    base_missing, _ = cond_transformer.load_state_dict(pipeline.transformer.state_dict(), strict=False)
    if is_main_process(rank):
        print(f"  Base load: {len(base_missing)} missing keys (expected for pref modules)")

    cond_transformer = cond_transformer.to(device)

    # Load PEFT adapter or legacy weights
    if _is_peft_adapter_dir(lora_path):
        if is_main_process(rank):
            print(f"Loading PEFT adapter from: {lora_path}")
        cond_transformer = PeftModel.from_pretrained(cond_transformer, lora_path, is_trainable=False)
        cond_transformer.set_adapter("default")
    else:
        if is_main_process(rank):
            print(f"Loading legacy weights from: {lora_path}")
        state = _load_state_dict_from_dir(lora_path)
        missing, unexpected = cond_transformer.load_state_dict(state, strict=False)
        if is_main_process(rank) and missing:
            pref_missing = [k for k in missing if "pref_" in k]
            if pref_missing:
                print(f"  WARNING: Missing pref weights: {pref_missing}")

    cond_transformer.eval()
    pipeline.transformer = cond_transformer

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline.transformer.to(device, dtype=text_encoder_dtype)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)
    pipeline.safety_checker = None

    # Extract text encoders/tokenizers for encode_prompt
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # --- Load Dataset with Distributed Sampler ---
    dataset_path = f"dataset/{args.dataset}"
    if is_main_process(rank):
        print(f"Loading dataset from: {dataset_path}")

    if args.dataset == "geneval":
        dataset = GenevalPromptDataset(dataset_path, split="test")
        all_reward_scorers = {"geneval": 1.0}
        eval_batch_size = 14
    elif args.dataset == "ocr":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {"ocr": 1.0}
        eval_batch_size = 16
    elif args.dataset == "pickscore":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {
            "imagereward": 1.0,
            "pickscore": 1.0,
            "aesthetic": 1.0,
            "unifiedreward": 1.0,
            "clipscore": 1.0,
            "hpsv2": 1.0,
        }
        eval_batch_size = 16
    elif args.dataset == "drawbench":
        dataset = TextPromptDataset(dataset_path, split="test")
        all_reward_scorers = {
            "imagereward": 1.0,
            "pickscore": 1.0,
            "aesthetic": 1.0,
            "unifiedreward": 1.0,
            "clipscore": 1.0,
            "hpsv2": 1.0,
        }
        eval_batch_size = 5

    if args.batch_size > 0:
        eval_batch_size = args.batch_size

    # --- Subsample prompts (same subset for every weight point) ---
    if args.num_samples and args.num_samples < len(dataset):
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(dataset), size=args.num_samples, replace=False)
        indices.sort()
        dataset = Subset(dataset, indices.tolist())
        if is_main_process(rank):
            print(f"Subsampled {args.num_samples} prompts (seed={args.seed}) from {len(indices)} candidates")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # --- Instantiate Reward Models ---
    scoring_fn = None
    if not args.generate_only:
        if is_main_process(rank):
            print("Initializing reward models...")
        scoring_fn = multi_score(device, all_reward_scorers)

    # --- Generate Preference Weights ---
    num_weights = args.num_weights
    weights_list = []
    for i in range(num_weights):
        w1 = i / (num_weights - 1) if num_weights > 1 else 0.5
        w2 = 1.0 - w1
        weights_list.append([w1, w2])

    if is_main_process(rank):
        print(f"\nSweeping {num_weights} preference weights: {weights_list}")

    all_weights_summary = []

    # --- Outer Loop: Preference Weights ---
    for wi, current_weights in enumerate(weights_list):
        weight_tag = "_".join(f"{w:.2f}" for w in current_weights)
        weight_subdir = os.path.join(args.output_dir, f"w_{weight_tag}")

        if is_main_process(rank):
            print(f"\n=== Weight point {wi + 1}/{num_weights}: {current_weights} ===")
            os.makedirs(weight_subdir, exist_ok=True)
            os.makedirs(os.path.join(weight_subdir, "images"), exist_ok=True)

        dist.barrier(device_ids=[local_rank])  # ensure all ranks wait for rank 0 to create directories

        results_this_rank = []

        for batch in tqdm(
            dataloader,
            desc=f"Evaluating w={current_weights} (Rank {rank})",
            disable=not is_main_process(rank),
        ):
            prompts, metadata, indices = batch
            current_batch_size = len(prompts)

            with torch.amp.autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    # Encode prompts (batch-native)
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders, tokenizers, prompts, max_sequence_length=128
                    )
                    prompt_embeds = prompt_embeds.to(device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

                    # Encode negative prompts
                    neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
                        text_encoders, tokenizers, [""] * current_batch_size, max_sequence_length=128
                    )
                    neg_prompt_embeds = neg_prompt_embeds.to(device)
                    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to(device)

                    # Preference tensor: same weights for entire batch
                    pref_tensor = torch.tensor(
                        [current_weights] * current_batch_size,
                        device=device, dtype=torch.float32,
                    )

                    # Generate images
                    images, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=neg_prompt_embeds,
                        negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        output_type="pt",
                        height=args.resolution,
                        width=args.resolution,
                        noise_level=args.noise_level,
                        deterministic=args.deterministic,
                        solver=args.solver,
                        model_type="sd3",
                        preference=pref_tensor,
                    )

            all_scores = None
            if scoring_fn is not None:
                all_scores, _ = scoring_fn(images, prompts, metadata, only_strict=False)

            for i in range(current_batch_size):
                sample_idx = indices[i]
                result_item = {
                    "sample_id": sample_idx,
                    "prompt": prompts[i],
                    "metadata": metadata[i] if metadata else {},
                    "preference_weights": current_weights,
                    "scores": {},
                }

                image_path = os.path.join(
                    weight_subdir, "images", f"{sample_idx:05d}.jpg"
                )
                pil_image = Image.fromarray(
                    (images[i].float().cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                )
                pil_image.save(image_path)
                result_item["image_path"] = image_path

                if all_scores is not None:
                    for score_name, score_values in all_scores.items():
                        if isinstance(score_values, torch.Tensor):
                            result_item["scores"][score_name] = score_values[i].detach().cpu().item()
                        else:
                            result_item["scores"][score_name] = float(score_values[i])

                results_this_rank.append(result_item)

            del images, all_scores
            torch.cuda.empty_cache()

        # --- Gather per-weight results across ranks ---
        dist.barrier()

        all_gathered = [None] * world_size
        dist.all_gather_object(all_gathered, results_this_rank)

        if is_main_process(rank):
            flat_results = [item for sublist in all_gathered for item in sublist]
            flat_results.sort(key=lambda x: x["sample_id"])

            # Save per-weight JSONL
            results_filepath = os.path.join(weight_subdir, "evaluation_results.jsonl")
            with open(results_filepath, "w") as f_out:
                for result_item in flat_results:
                    f_out.write(json.dumps(result_item) + "\n")

            # Compute per-weight averages (preserving sentinel -10.0 filtering)
            all_scores_agg = defaultdict(list)
            for result in flat_results:
                for score_name, score_value in result["scores"].items():
                    if isinstance(score_value, (int, float)):
                        all_scores_agg[score_name].append(score_value)

            average_scores = {
                name: float(np.mean(list(filter(lambda score: score != -10.0, scores))))
                for name, scores in all_scores_agg.items()
            }

            avg_scores_filepath = os.path.join(weight_subdir, "average_scores.json")
            with open(avg_scores_filepath, "w") as f_avg:
                json.dump(average_scores, f_avg, indent=4)

            print(f"  Weight {current_weights}: {len(flat_results)} samples evaluated")
            for name, avg_score in sorted(average_scores.items()):
                print(f"    {name:<20}: {avg_score:.4f}")

            all_weights_summary.append({
                "weights": current_weights,
                "average_scores": average_scores,
                "num_samples": len(flat_results),
            })

    # --- Final Summary ---
    if is_main_process(rank):
        summary_path = os.path.join(args.output_dir, "all_weights_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_weights_summary, f, indent=4)

        print("\n--- Summary Across All Weights ---")
        for entry in all_weights_summary:
            w_str = ", ".join(f"{w:.2f}" for w in entry["weights"])
            scores_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in sorted(entry["average_scores"].items())
            )
            print(f"  [{w_str}] -> {scores_str}")
        print("----------------------------------")
        print(f"Summary saved to {summary_path}")

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a conditioned NFT diffusion model in a distributed manner.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Local path to the checkpoint directory (e.g., 'logs/nft/sd3/.../checkpoints/checkpoint-9').",
    )
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-3.5-medium",
                        help="Base SD3 model id or local path.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="sd3",
        choices=["sd3"],
        help="Type of the base model ('sd3').",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["geneval", "ocr", "pickscore", "drawbench"], help="Dataset type."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation results and generated images.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=25, help="Number of inference steps for the diffusion pipeline."
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution of the generated images.")
    parser.add_argument(
        "--save_images", action="store_true", help="Include this flag to save generated images to the output directory."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between 'no', 'fp16', or 'bf16'.",
    )

    # --- Conditional evaluation args ---
    parser.add_argument("--objective_names", type=str, nargs="+", default=["objective_1", "objective_2"],
                        help="Names of the reward objectives (for labeling weight dimensions)")
    parser.add_argument("--num_weights", type=int, default=5,
                        help="Number of uniformly-spaced preference weight points")
    parser.add_argument("--conditioning_mode", type=str, default="temb_blk_shared",
                        choices=["hybrid", "adaln_both",
                                 "temb_only", "output_only", "temb_gated_output",
                                 "temb_blk_shared", "temb_blk_stage", "temb_blk_per"],
                        help="Conditioning mode (overridden by checkpoint metadata if available)")
    parser.add_argument("--block_mod_form", type=str, default="residual",
                        choices=["affine", "scale_only", "shift_only", "residual"],
                        help="Block modulation form (ablation modes only)")
    parser.add_argument("--use_pooled_text", action="store_true",
                        help="Use pooled text embeddings in block projector")
    parser.add_argument("--num_freqs", type=int, default=1,
                        help="Number of sinusoidal frequency bands")
    parser.add_argument("--mod_block_fraction", type=float, default=0.5,
                        help="Fraction of blocks that receive modulation")
    parser.add_argument("--solver", type=str, default="flow",
                        help="ODE solver (training default: flow)")
    parser.add_argument("--noise_level", type=float, default=0.7,
                        help="Noise level (training default: 0.7)")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic", default=True,
                        help="Disable deterministic sampling")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of prompts to randomly sample from the test set (0 = use all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt subsampling")
    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate and save images, skip reward scoring")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Override per-GPU batch size (0 = use dataset default)")

    args = parser.parse_args()
    main(args)