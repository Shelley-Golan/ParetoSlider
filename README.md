<h1 align="center">ParetoSlider: Diffusion Models Post-Training for Continuous Reward Control</h1>

<p align="center">
  <img src="assets/teaser.png" alt="ParetoSlider Teaser" width="100%">
</p>

<br>

ParetoSlider trains a **single diffusion model** that learns an entire **Pareto front** across multiple competing objectives — for example, photorealism vs. artistic sketch style. Instead of committing to one fixed trade-off, the model accepts a **preference vector** at inference time, letting users smoothly slide between objectives with no retraining.

## TODO

- [x] T2I code release
- [ ] I2I coming soon
- [ ] T2V coming soon

---

## Key Ideas

- **Multi-objective reward training** — The model is fine-tuned online with multiple reward signals (e.g., PickScore for photorealism, Qwen-VL for sketch style) rather than collapsing them into a single scalar.
- **Preference-conditioned generation** — A preference vector on the objective simplex (e.g., `[0.7, 0.3]`) is injected into the SD3 transformer via learned temb injection and shared block modulation, giving continuous control over the output style.
- **Late scalarization** — Advantages are normalized within (prompt, preference) subgroups, so the model trains stably across different regions of the Pareto front.
- **Efficient LoRA fine-tuning** — Only LoRA adapters and lightweight preference modules are trained; the base SD3 weights stay frozen.

## Architecture

ParetoSlider builds on **Stable Diffusion 3.5 Medium** and adds two preference conditioning paths:

1. **Temb injection** — A small MLP maps the preference vector into the time/text embedding space.
2. **Shared block modulation** — A `SliderProjector` (sinusoidal PE + 4-layer MLP) produces a modulation vector that is applied to all transformer blocks via image-only residual injection.

Both paths are initialized near-identity so the model starts from the base SD3 behavior and gradually learns preference-dependent generation.

## Quick Start

```bash
cd T2I

# Install dependencies
pip install torch diffusers transformers peft accelerate wandb \
    ml-collections absl-py numpy pillow tqdm safetensors matplotlib

# Launch training (4 GPUs, photorealism vs. sketch)
torchrun --nproc_per_node=4 scripts/train_pareto_nft_sd3.py \
    --config=config/nft.py:sd3_qwen_style_sketch

# Evaluate with preference sweep
python scripts/eval_preference.py \
    --checkpoint_path ./logs/nft/sd3/<run_name>/checkpoint_<N> \
    --base_model stabilityai/stable-diffusion-3.5-medium \
    --objective_names pickscore_photorealism qwen_style_sketch \
    --weight_mode sweep --num_weights 11 --compute_scores
```

See [`T2I/README.md`](T2I/README.md) for full documentation on configuration, evaluation, datasets, and checkpoints.

## Repository Structure

```
ParetoSlider/
├── T2I/                    # Training, evaluation, and core library
│   ├── config/             # ml_collections configs and experiment presets
│   ├── scripts/            # Training and evaluation entry points
│   ├── flow_grpo/          # Rewards, stat tracking, patched SD3 pipeline
│   └── dataset/            # Bundled prompt datasets
├── reward_server/          # Qwen-VL reward server (for style objectives)
└── assets/                 # Figures and media
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 with CUDA
- 4+ GPUs (tested with 4–6 NVIDIA GPUs)
- HuggingFace `diffusers`, `transformers`, `peft`
- Weights & Biases for logging

## License

Apache License 2.0 — see [T2I/LICENSE](T2I/LICENSE).

## Acknowledgements

We thank [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) for their awesome code.

## Citation

```bibtex
@article{golan2025paretoSlider,
  title={{ParetoSlider}: Diffusion Models Post-Training for Continuous Reward Control},
  author={Golan, Shelly and Finkelson, Michael and Bereslavsky, Ariel and Nitzan, Yotam and Patashnik, Or},
  journal={arXiv preprint arXiv:2604.20816},
  year={2025}
}
```
