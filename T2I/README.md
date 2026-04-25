# ParetoSlider — Preference-Conditioned Pareto Training for Text-to-Image Diffusion

ParetoSlider fine-tunes **Stable Diffusion 3** with **LoRA** using online reward-driven policy gradients (NFT/flow-GRPO style), training a single model that learns a **family of Pareto-optimal solutions** across multiple objectives (e.g., photorealism vs. sketch style). At inference time, a **preference vector** on the objective simplex controls the trade-off — no retraining needed.

## Project Structure

```
ParetoSlider/
├── T2I/                                  # Main training & evaluation code
│   ├── config/
│   │   ├── base.py                       # Default hyperparameters (ml_collections)
│   │   └── nft.py                        # Named experiment presets
│   ├── scripts/
│   │   ├── train_pareto_nft_sd3.py       # Preference-conditioned training (distributed)
│   │   ├── train_nft_sd3.py              # Baseline single-objective training
│   │   ├── eval_preference.py            # Interactive preference sweep & visualization
│   │   └── conditional_evaluation.py     # Distributed evaluation with preference grid
│   ├── flow_grpo/                        # Core library
│   │   ├── rewards.py                    # Reward functions & multi_score_conditioned
│   │   ├── stat_tracking.py              # Per-prompt-and-preference stat tracker (GDPO)
│   │   ├── scalarization.py              # Loss scalarization (linear)
│   │   ├── ema.py                        # EMA for trainable parameters
│   │   ├── pickscore_scorer.py           # PickScore reward scorer
│   │   └── diffusers_patch/
│   │       ├── transformer_sd3.py        # SD3Transformer2DModelWithConditioning
│   │       └── pipeline_with_logprob.py  # Patched pipeline with log-prob + preference
│   └── dataset/                          # Bundled prompt datasets
│       ├── pickscore/                    # train.txt / test.txt
│       ├── drawbench/                    # train.txt / test.txt
│       ├── ocr/                          # train.txt / test.txt
│       └── geneval/                      # train_metadata.jsonl / test_metadata.jsonl
└── reward_server/                        # External Qwen-VL reward server
    ├── multi_reward_server.py
    ├── reward_server.py
    └── pyproject.toml
```

## Requirements

There is no `requirements.txt` — install the following manually:

```bash
pip install torch diffusers transformers peft accelerate wandb \
    ml-collections absl-py numpy pillow tqdm safetensors matplotlib
```

For the **Qwen-VL reward server** (only needed for style objectives like `qwen_style_sketch`):

```bash
cd reward_server
pip install -e .
```

### Hardware

- Multi-GPU setup required for distributed training (tested with 4–6 GPUs).
- Mixed precision (`fp16`) is enabled by default and significantly speeds up training.

### Models

The following HuggingFace models are downloaded automatically:


| Model                                     | Purpose                      |
| ----------------------------------------- | ---------------------------- |
| `stabilityai/stable-diffusion-3.5-medium` | Base diffusion model         |
| `yuvalkirstain/PickScore_v1`              | Photorealism reward scorer   |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`   | CLIP processor for PickScore |


## Conditional Training

### How It Works

1. **Sample images** from the current model, each conditioned on a random **preference vector** drawn from the objective simplex (e.g., `[0.7, 0.3]` for 70% photorealism, 30% sketch).
2. **Score** each image independently on every objective using `multi_score_conditioned`.
3. **Normalize** advantages within (prompt, preference-slot) subgroups via `PerPromptAndPreferenceStatTracker` (GDPO-style).
4. **Update** the model with per-objective NFT losses, scalarized by the preference vector.

The preference vector is injected into the SD3 transformer via two learnable conditioning paths (temb + shared block modulation), while the base model weights stay frozen under LoRA.

### Conditioning Architecture

The single supported conditioning mode uses two injection paths:

1. **Temb injection** — A small MLP projects the preference vector and adds it to the time/text embedding. Near-identity at init via small std on the last linear layer.
2. **Shared block modulation** — A `SliderProjector` (sinusoidal PE of the preference vector, optionally concatenated with pooled text embeddings, fed through a 4-layer MLP) produces a single modulation vector shared across all active transformer blocks. Each block is wrapped in an `ImageOnlyModulationBlock` that applies the modulation to the image stream only (never the text stream). The `block_mod_form` parameter controls how modulation is applied:


| `block_mod_form`     | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `residual` (default) | Residual injection after FF, gated by the block's native `gate_mlp` |
| `affine`             | Adds to both scale and shift in AdaLN (mod dim = inner_dim * 2)     |
| `scale_only`         | Adds to scale only                                                  |
| `shift_only`         | Adds to shift only                                                  |


### Quick Start

All commands must be run from the `T2I/` directory:

```bash
cd ParetoSlider/T2I
```

#### 1. Start the Reward Server (if using Qwen-VL style objectives)

```bash
cd ../reward_server
python multi_reward_server.py
```

The server listens on `http://127.0.0.1:12341` by default. Override with:

```bash
export QWEN_VL_REWARD_URL=http://<host>:<port>
```

#### 2. Launch Conditional Training

Training uses `torchrun` for distributed execution and `ml_collections` config flags:

```bash
torchrun --nproc_per_node=4 scripts/train_pareto_nft_sd3.py \
    --config=config/nft.py:sd3_qwen_style_sketch
```

The `--config` flag takes the form `config/nft.py:<function_name>` where available presets are:


| Preset Function                                    | Objectives                   | Loss Mode       | GPUs |
| -------------------------------------------------- | ---------------------------- | --------------- | ---- |
| `sd3_qwen_style_sketch`                            | photorealism + sketch        | `per_objective` | 4    |
| `sd3_qwen_sketch_photorealism_single_loss`         | photorealism + sketch        | `single_loss`   | 6    |
| `sd3_1_pickscore_photorealism_0_qwen_style_sketch` | photorealism only (baseline) | —               | 6    |


#### 3. Environment Variables


| Variable                 | Default                  | Description                                             |
| ------------------------ | ------------------------ | ------------------------------------------------------- |
| `NFT_LOGDIR`             | `./logs`                 | Root directory for checkpoints and logs                 |
| `NFT_LORA_PATH`          | `None`                   | Path to a pretrained LoRA checkpoint to initialize from |
| `QWEN_VL_REWARD_URL`     | `http://127.0.0.1:12341` | Qwen-VL reward server URL                               |
| `QWEN_VL_REWARD_TIMEOUT` | `1800`                   | HTTP timeout (seconds) for reward server requests       |
| `MASTER_ADDR`            | `localhost`              | Distributed training master address                     |
| `MASTER_PORT`            | `12355`                  | Distributed training master port                        |


### Key Configuration Fields

These can be overridden in a custom config function (see `config/nft.py` for examples):


| Field                                | Type    | Description                                                                                             |
| ------------------------------------ | ------- | ------------------------------------------------------------------------------------------------------- |
| `config.loss_mode`                   | `str`   | `"per_objective"` (recommended) or `"single_loss"`                                                      |
| `config.num_pref_per_prompt`         | `int`   | K distinct preference slots per prompt per batch (default: 1; use 3 for Pareto training)                |
| `config.reward_fn`                   | `dict`  | Maps reward function names to weights, e.g. `{"pickscore_photorealism": 1.0, "qwen_style_sketch": 1.0}` |
| `config.block_mod_form`              | `str`   | Block modulation type: `"residual"` (default), `"affine"`, `"scale_only"`, `"shift_only"`               |
| `config.mod_block_fraction`          | `float` | Fraction of transformer blocks that receive preference modulation (0.0–1.0, default: 1.0)               |
| `config.num_freqs`                   | `int`   | Number of sinusoidal frequency bands in `SliderProjector` PE (default: 1)                               |
| `config.beta`                        | `float` | Outer beta controlling decay schedule                                                                   |
| `config.train.beta`                  | `float` | KL penalty coefficient in NFT loss                                                                      |
| `config.train.lora_rank`             | `int`   | LoRA rank (default: 32 in presets)                                                                      |
| `config.sample.num_steps`            | `int`   | Denoising steps during training sampling (5–25 typical)                                                 |
| `config.sample.num_image_per_prompt` | `int`   | Images per prompt per batch (24 in presets)                                                             |


### Monitoring

Training logs to **Weights & Biases** under the project name `pareto-slider`. Track per-objective rewards, advantages, gradient norms, and sample images in real time.

## Evaluation

### Interactive Preference Sweep

Generate images across a range of preference weights and produce an HTML gallery:

```bash
python scripts/eval_preference.py \
    --checkpoint_path ./logs/nft/sd3/<run_name>/checkpoint_<N> \
    --base_model stabilityai/stable-diffusion-3.5-medium \
    --objective_names pickscore_photorealism qwen_style_sketch \
    --weight_mode sweep \
    --num_weights 11 \
    --num_seeds 3 \
    --prompt "A cat sitting on a windowsill" \
    --compute_scores \
    --output_dir ./eval_output
```

Key `eval_preference.py` options:


| Flag                         | Description                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| `--weight_mode`              | `sweep` (linear interpolation), `corners` (one-hot extremes), `simplex_grid` (uniform grid) |
| `--num_weights`              | Number of weight points to evaluate                                                         |
| `--num_seeds`                | Number of random seeds per weight                                                           |
| `--compute_scores`           | Also score images with reward functions                                                     |
| `--plot`                     | Generate Pareto front plots                                                                 |
| `--custom_weights`           | Comma-separated custom weight values                                                        |
| `--baseline_checkpoint_path` | Compare against a non-conditioned baseline                                                  |


### Distributed Conditional Evaluation

For large-scale evaluation across the full test set with a grid of preference weights:

```bash
torchrun --nproc_per_node=4 scripts/conditional_evaluation.py \
    --checkpoint_path ./logs/nft/sd3/<run_name>/checkpoint_<N> \
    --base_model stabilityai/stable-diffusion-3.5-medium \
    --dataset dataset/pickscore \
    --objective_names pickscore_photorealism qwen_style_sketch \
    --num_weights 11 \
    --output_dir ./eval_output \
    --save_images
```

Checkpoint `metadata.json` (saved during training) is auto-detected and overrides CLI flags for `pref_dim`, `block_mod_form`, `num_freqs`, and `mod_block_fraction`.

## Datasets

Prompt datasets are bundled under `T2I/dataset/`. Each dataset contains text prompts (one per line for `.txt`, one JSON object per line for `.jsonl`):


| Dataset     | Format                                         | Use Case                         |
| ----------- | ---------------------------------------------- | -------------------------------- |
| `pickscore` | `train.txt` / `test.txt`                       | General text-to-image prompts    |
| `drawbench` | `train.txt` / `test.txt`                       | DrawBench evaluation prompts     |
| `ocr`       | `train.txt` / `test.txt`                       | OCR-focused prompts              |
| `geneval`   | `train_metadata.jsonl` / `test_metadata.jsonl` | GenEval with structured metadata |


## Checkpoints

Checkpoints are saved every `config.save_freq` epochs to `<logdir>/nft/sd3/<run_name>/checkpoint_<epoch>/`. Each checkpoint contains:

- `lora/` — LoRA adapter weights (PEFT format)
- `metadata.json` — preference dimension, block modulation form, and architecture details

To resume training from a checkpoint:

```python
config.resume_from = "./logs/nft/sd3/<run_name>/checkpoint_30"
```

Or set it as an environment override when launching.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

```
NVIDIA CORPORATION & AFFILIATES
```

