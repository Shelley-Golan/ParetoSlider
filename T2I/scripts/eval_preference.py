# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate images with different preference weights and save to disk.
Sweeps across preference weights for conditioned NFT models.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import glob

from diffusers import StableDiffusion3Pipeline

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    LoraConfig = None
    get_peft_model = None

# Ensure we import the local `flow_grpo` package from this repo
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flow_grpo.rewards import multi_score, MULTI_OUTPUT_SCORERS
from flow_grpo.diffusers_patch.transformer_sd3 import SD3Transformer2DModelWithConditioning
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def _is_peft_adapter_dir(path: str) -> bool:
    """Check if directory contains a PEFT adapter (has adapter_config.json)."""
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def _load_state_dict_from_dir(path: str):
    """Load state dict from directory, trying safetensors first, then .bin."""
    # Try safetensors first, then .bin
    safes = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if safes:
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("safetensors is required to load .safetensors checkpoints") from e
        return load_file(safes[0])

    bins = sorted(glob.glob(os.path.join(path, "*.bin")))
    if bins:
        return torch.load(bins[0], map_location="cpu")

    raise FileNotFoundError(f"No *.safetensors or *.bin found in: {path}")


# HTML gallery template
HTML_GALLERY_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Preference Gallery</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      margin: 0;
      padding: 16px;
      background: #111;
      color: #eee;
    }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .panel { border: 1px solid #444; padding: 12px; border-radius: 8px; margin-bottom: 12px; background: #1b1b1b; }
    label { font-size: 14px; }
    select, input[type="text"], input[type="number"] {
      padding: 6px 8px; border-radius: 6px; border: 1px solid #555; background: #222; color: #eee;
    }
    button {
      padding: 8px 12px; border: 1px solid #666; background: #2a2a2a; color: #eee; border-radius: 6px;
      cursor: pointer;
    }
    button:hover { background: #333; }
    .images-row { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }
    .image-panel { text-align: center; flex: 1; min-width: 300px; max-width: 500px; }
    .image-panel img { width: 100%; border-radius: 8px; border: 1px solid #333; }
    .image-panel h3 { margin: 0 0 8px 0; font-size: 14px; color: #aaa; }
    .meta { font-size: 13px; line-height: 1.4; white-space: pre-line; text-align: left; margin-top: 8px; }
    input[type="range"] { width: 100%; }
    #error { color: #ffb3b3; font-weight: 600; }
    .flex-col { display: flex; flex-direction: column; gap: 8px; }
    .prompt-text { font-size: 14px; margin-bottom: 8px; color: #ccc; }
    .hidden { display: none; }
  </style>
</head>
<body>
  <h2>Preference Gallery</h2>

  <div class="panel">
    <div class="row">
      <label for="promptSelect">Prompt:</label>
      <select id="promptSelect"></select>
      <label for="seedSelect">Seed:</label>
      <select id="seedSelect"></select>
    </div>
    <div class="flex-col">
      <label for="weightSlider">Weight: <span id="weightIdx">0</span>/<span id="weightMax">0</span> <span id="sampleInfo" style="font-size: 12px; color: #888;"></span></label>
      <input id="weightSlider" type="range" min="0" max="0" step="1" value="0" />
    </div>
  </div>

  <div class="panel">
    <div class="prompt-text" id="promptText"></div>
    <div class="images-row">
      <div class="image-panel" id="baselinePanel">
        <h3>Baseline (Original NFT)</h3>
        <img id="baselineImg" alt="baseline" />
        <div class="meta" id="baselineMeta"></div>
      </div>
      <div class="image-panel">
        <h3>Conditioned (Preference Sweep)</h3>
        <img id="prefImg" alt="preference" />
        <div class="meta" id="prefMeta"></div>
      </div>
    </div>
  </div>

  <script>
    const promptSelect = document.getElementById("promptSelect");
    const seedSelect = document.getElementById("seedSelect");
    const weightSlider = document.getElementById("weightSlider");
    const weightIdx = document.getElementById("weightIdx");
    const weightMax = document.getElementById("weightMax");
    const sampleInfo = document.getElementById("sampleInfo");
    const prefImgEl = document.getElementById("prefImg");
    const prefMetaEl = document.getElementById("prefMeta");
    const baselineImgEl = document.getElementById("baselineImg");
    const baselineMetaEl = document.getElementById("baselineMeta");
    const baselinePanel = document.getElementById("baselinePanel");
    const promptTextEl = document.getElementById("promptText");

    let results = [];
    let filteredGenerations = [];

    async function loadResults() {
      prefImgEl.src = "";
      prefMetaEl.textContent = "";
      baselineImgEl.src = "";
      baselineMetaEl.textContent = "";
      promptSelect.innerHTML = "";
      seedSelect.innerHTML = "";
      try {
        const res = await fetch("results.json");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        results = await res.json();
        if (!Array.isArray(results) || results.length === 0) {
          throw new Error("results.json is empty or malformed");
        }
        results.forEach((p, idx) => {
          const opt = document.createElement("option");
          opt.value = idx;
          const truncPrompt = (p.prompt ?? `Prompt ${idx}`).substring(0, 60);
          opt.textContent = truncPrompt + (p.prompt && p.prompt.length > 60 ? "..." : "");
          promptSelect.appendChild(opt);
        });
        onPromptChange();
      } catch (e) {
        console.error(e);
      }
    }

    function onPromptChange() {
      const pIdx = Number(promptSelect.value);
      const promptData = results[pIdx];
      if (!promptData || !promptData.generations || promptData.generations.length === 0) return;
      
      // Extract unique seeds and populate seed dropdown
      const seeds = [...new Set(promptData.generations.map(g => g.seed))].sort((a, b) => a - b);
      seedSelect.innerHTML = "";
      seeds.forEach(seed => {
        const opt = document.createElement("option");
        opt.value = seed;
        opt.textContent = `Seed ${seed}`;
        seedSelect.appendChild(opt);
      });
      
      onSeedChange();
    }

    function onSeedChange() {
      const pIdx = Number(promptSelect.value);
      const selectedSeed = Number(seedSelect.value);
      const promptData = results[pIdx];
      if (!promptData) return;
      
      // Filter generations by selected seed
      filteredGenerations = promptData.generations.filter(g => g.seed === selectedSeed);
      
      weightSlider.max = Math.max(0, filteredGenerations.length - 1);
      weightMax.textContent = filteredGenerations.length - 1;
      weightSlider.value = 0;
      updateImage();
    }

    function formatScores(scores) {
      if (!scores) return "";
      return Object.entries(scores).map(([k, v]) => `${k}: ${v.toFixed(3)}`).join("\\n");
    }

    function updateImage() {
      const pIdx = Number(promptSelect.value);
      const wIdx = Number(weightSlider.value);
      const selectedSeed = Number(seedSelect.value);
      weightIdx.textContent = wIdx;
      const promptData = results[pIdx];
      if (!promptData) return;
      
      promptTextEl.textContent = promptData.prompt;
      const promptDir = `prompt_${String(pIdx).padStart(3, "0")}`;
      
      // Update preference image from filtered generations
      const gen = filteredGenerations[wIdx];
      if (gen) {
        const imgPath = gen.image_path || "";
        const filename = imgPath.split("/").pop();
        prefImgEl.src = `${promptDir}/${filename}`;

        // Show weight info in the slider label (seed is now shown in dropdown)
        const weightKeys = Object.keys(gen).filter(k => k.endsWith("_weight"));
        const weights = weightKeys.map(k => `${k.replace("_weight", "")}: ${gen[k].toFixed(2)}`).join(", ");
        sampleInfo.textContent = weights ? `[${weights}]` : "";

        let metaText = "";
        for (const [key, value] of Object.entries(gen)) {
          if (key !== "image_path" && key !== "scores") {
            metaText += `${key}: ${typeof value === "number" ? value.toFixed(2) : value}\\n`;
          }
        }
        if (gen.scores) {
          metaText += "\\nScores:\\n" + formatScores(gen.scores);
        }
        prefMetaEl.textContent = metaText;
      }
      
      // Update baseline image if available - try to match the selected seed
      const baselineSeeds = promptData.baseline_seeds || [];
      let baseline = baselineSeeds.find(b => b.seed === selectedSeed);
      if (!baseline && promptData.baseline) {
        baseline = promptData.baseline;
      }
      
      if (baseline && baseline.image_path) {
        baselinePanel.classList.remove("hidden");
        const baselineFilename = baseline.image_path.split("/").pop();
        baselineImgEl.src = `${promptDir}/${baselineFilename}`;
        let baselineMeta = `type: ${baseline.type || "baseline"}\\n`;
        if (baseline.seed !== undefined) {
          baselineMeta += `seed: ${baseline.seed}\\n`;
        }
        if (baseline.scores) {
          baselineMeta += "\\nScores:\\n" + formatScores(baseline.scores);
        }
        baselineMetaEl.textContent = baselineMeta;
      } else {
        baselinePanel.classList.add("hidden");
      }
    }

    weightSlider.addEventListener("input", updateImage);
    promptSelect.addEventListener("change", onPromptChange);
    seedSelect.addEventListener("change", onSeedChange);

    // Arrow key support for quick browsing
    window.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") {
        weightSlider.value = Math.max(Number(weightSlider.min), Number(weightSlider.value) - 1);
        updateImage();
      } else if (e.key === "ArrowRight") {
        weightSlider.value = Math.min(Number(weightSlider.max), Number(weightSlider.value) + 1);
        updateImage();
      }
    });

    // Auto-load on page load
    loadResults();
  </script>
</body>
</html>
'''


def compute_pareto_front(points):
    """Return indices of Pareto-optimal points (maximize both objectives).
    
    Args:
        points: List of (score1, score2) tuples or numpy array of shape (N, 2)
        
    Returns:
        List of indices of Pareto-optimal points, sorted by first objective
    """
    points = np.array(points)
    pareto_indices = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            # q dominates p if q is >= p in all objectives and > p in at least one
            if i != j and q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    # Sort by first objective for drawing the front line
    pareto_indices.sort(key=lambda idx: points[idx, 0])
    return pareto_indices


def plot_pareto_front(results_path, output_path, objective_names):
    """Plot Pareto front from evaluation results.
    
    Aggregates points by weight (mean ± std over seeds) for cleaner analysis.
    
    Args:
        results_path: Path to results.json file
        output_path: Path to save the output PNG
        objective_names: List of objective names (e.g., ["clipscore", "pickscore"])
    """
    from collections import defaultdict
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if len(objective_names) != 2:
        print(f"Warning: Pareto plot only supports 2 objectives, got {len(objective_names)}")
        return
    
    obj1, obj2 = objective_names
    
    # Set up the plot with a clean style
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Color palette for multiple prompts
    colors = plt.cm.tab10.colors
    
    for pi, prompt_data in enumerate(results):
        prompt = prompt_data.get("prompt", f"Prompt {pi}")
        generations = prompt_data.get("generations", [])
        baseline = prompt_data.get("baseline")
        
        # Group scores by (w1, w2) to aggregate over seeds
        weight_groups = defaultdict(list)
        for gen in generations:
            scores = gen.get("scores", {})
            if obj1 in scores and obj2 in scores:
                w1 = gen.get(f"{obj1}_weight", None)
                w2 = gen.get(f"{obj2}_weight", None)
                if w1 is not None and w2 is not None:
                    key = (round(w1, 3), round(w2, 3))  # Round to avoid float precision issues
                    weight_groups[key].append((scores[obj1], scores[obj2]))
        
        if not weight_groups:
            print(f"Warning: No scores found for prompt {pi}")
            continue
        
        # Compute mean and std per weight
        mean_points = []
        std_points = []
        weight_labels = []
        for (w1, w2), score_list in sorted(weight_groups.items(), key=lambda x: -x[0][0]):
            scores_arr = np.array(score_list)
            mean_clip = np.mean(scores_arr[:, 0])
            mean_pick = np.mean(scores_arr[:, 1])
            std_clip = np.std(scores_arr[:, 0]) if len(scores_arr) > 1 else 0
            std_pick = np.std(scores_arr[:, 1]) if len(scores_arr) > 1 else 0
            mean_points.append((mean_clip, mean_pick))
            std_points.append((std_clip, std_pick))
            weight_labels.append((w1, w2))
        
        mean_points = np.array(mean_points)
        std_points = np.array(std_points)
        color = colors[pi % len(colors)]
        
        # Truncate prompt for legend
        label = prompt[:40] + "..." if len(prompt) > 40 else prompt
        
        # Plot mean points with error bars
        ax.errorbar(mean_points[:, 0], mean_points[:, 1],
                    xerr=std_points[:, 0], yerr=std_points[:, 1],
                    fmt='o', c=color, markersize=8, alpha=0.8,
                    ecolor=color, elinewidth=1.5, capsize=3, capthick=1.5,
                    label=label, zorder=2)
        
        # Add weight annotations next to each mean point
        for idx, ((x, y), (w1, w2)) in enumerate(zip(mean_points, weight_labels)):
            weight_text = f"({w1:.2f}, {w2:.2f})"
            ax.annotate(weight_text, (x, y), textcoords="offset points",
                       xytext=(8, 4), fontsize=8, color='#333',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='gray', alpha=0.7))
        
        # Find and highlight Pareto-optimal points (from mean points)
        pareto_indices = compute_pareto_front(mean_points)
        if pareto_indices:
            pareto_points = mean_points[pareto_indices]
            
            # Draw Pareto front line connecting means
            ax.plot(pareto_points[:, 0], pareto_points[:, 1], 
                    c=color, linewidth=2, alpha=0.8, zorder=3)
            
            # Highlight Pareto-optimal points with larger markers
            ax.scatter(pareto_points[:, 0], pareto_points[:, 1], 
                       c=[color], s=150, alpha=1.0, marker='*',
                       edgecolors='black', linewidth=1, zorder=4)
        
        # Plot baseline if available
        if baseline and baseline.get("scores"):
            b_scores = baseline["scores"]
            if obj1 in b_scores and obj2 in b_scores:
                ax.scatter(b_scores[obj1], b_scores[obj2], 
                           c=[color], s=200, marker='X', alpha=0.9,
                           edgecolors='black', linewidth=1.5, zorder=5,
                           label=f"Baseline ({label[:20]}...)" if len(label) > 20 else f"Baseline ({label})")
        
        all_points.extend(mean_points.tolist())
    
    # Labels and styling
    ax.set_xlabel(obj1.upper(), fontsize=12, fontweight='bold')
    ax.set_ylabel(obj2.upper(), fontsize=12, fontweight='bold')
    ax.set_title('Pareto Front Analysis (Mean ± Std over Seeds)', fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Add annotation explaining markers
    legend_text = "★ = Pareto-optimal (means)\n✕ = Baseline\nError bars = ±1 std"
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Pareto front plot saved to: {output_path}")


def plot_pareto_front_per_prompt(results_path, output_dir, objective_names):
    """Generate a separate Pareto front plot for EACH prompt.
    
    Args:
        results_path: Path to results.json file
        output_dir: Directory to save per-prompt plots
        objective_names: List of objective names (e.g., ["clipscore", "pickscore"])
    """
    from collections import defaultdict
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if len(objective_names) != 2:
        print(f"Warning: Pareto plot only supports 2 objectives, got {len(objective_names)}")
        return
    
    obj1, obj2 = objective_names
    
    # Create subdirectory for per-prompt plots
    plots_dir = os.path.join(output_dir, "pareto_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for pi, prompt_data in enumerate(results):
        prompt = prompt_data.get("prompt", f"Prompt {pi}")
        generations = prompt_data.get("generations", [])
        baseline = prompt_data.get("baseline")
        
        # Group scores by weight
        weight_groups = defaultdict(list)
        for gen in generations:
            scores = gen.get("scores", {})
            if obj1 in scores and obj2 in scores:
                w1 = gen.get(f"{obj1}_weight", None)
                w2 = gen.get(f"{obj2}_weight", None)
                if w1 is not None and w2 is not None:
                    key = (round(w1, 3), round(w2, 3))
                    weight_groups[key].append((scores[obj1], scores[obj2]))
        
        if not weight_groups:
            print(f"Warning: No scores found for prompt {pi}, skipping plot")
            continue
        
        # Create individual plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # Compute means and stds per weight
        mean_points = []
        std_points = []
        weight_labels = []
        for (w1, w2), score_list in sorted(weight_groups.items(), key=lambda x: -x[0][0]):
            scores_arr = np.array(score_list)
            mean_clip = np.mean(scores_arr[:, 0])
            mean_pick = np.mean(scores_arr[:, 1])
            std_clip = np.std(scores_arr[:, 0]) if len(scores_arr) > 1 else 0
            std_pick = np.std(scores_arr[:, 1]) if len(scores_arr) > 1 else 0
            mean_points.append((mean_clip, mean_pick))
            std_points.append((std_clip, std_pick))
            weight_labels.append(f"({w1:.2f}, {w2:.2f})")
        
        mean_points = np.array(mean_points)
        std_points = np.array(std_points)
        
        # Plot points with error bars
        ax.errorbar(mean_points[:, 0], mean_points[:, 1],
                    xerr=std_points[:, 0], yerr=std_points[:, 1],
                    fmt='o', c='steelblue', markersize=10, alpha=0.8,
                    ecolor='steelblue', elinewidth=1.5, capsize=3, capthick=1.5,
                    label='Preference weights', zorder=2)
        
        # Add weight labels to points
        for i, label in enumerate(weight_labels):
            ax.annotate(label, (mean_points[i, 0], mean_points[i, 1]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
        
        # Compute and plot Pareto front
        pareto_idx = compute_pareto_front(mean_points)
        if pareto_idx:
            pareto_pts = mean_points[pareto_idx]
            ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], 'r--', linewidth=2, 
                    label='Pareto Front', zorder=3)
            ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], s=150, c='red', 
                       marker='*', label='Pareto-optimal', zorder=4)
        
        # Plot baseline if available
        if baseline and baseline.get("scores"):
            bx = baseline["scores"].get(obj1)
            by = baseline["scores"].get(obj2)
            if bx is not None and by is not None:
                ax.scatter(bx, by, s=200, c='green', marker='X', 
                           label='Baseline', zorder=5, edgecolors='darkgreen', linewidths=2)
        
        ax.set_xlabel(obj1.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel(obj2.upper(), fontsize=12, fontweight='bold')
        
        # Truncate prompt for title
        title_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        ax.set_title(f"Pareto Front: {title_prompt}", fontsize=10, fontweight='bold', pad=10)
        
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Save per-prompt plot
        safe_name = f"pareto_prompt_{pi:03d}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, safe_name), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"Per-prompt Pareto plots saved to: {plots_dir}/")


def plot_pairwise_pareto_fronts(results_path, output_dir, objective_names):
    """Generate 2D Pareto plots for all pairs of objectives.
    
    For N objectives, generates C(N,2) = N*(N-1)/2 pairwise plots.
    Useful for analyzing trade-offs in high-dimensional Pareto optimization.
    
    Args:
        results_path: Path to results.json file
        output_dir: Base output directory
        objective_names: List of all objective names (expanded)
    """
    from itertools import combinations
    from collections import defaultdict
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if len(objective_names) < 2:
        print(f"Warning: Need at least 2 objectives for pairwise plots, got {len(objective_names)}")
        return
    
    # Create subdirectory for pairwise plots
    plots_dir = os.path.join(output_dir, "pareto_pairwise")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate all pairs
    pairs = list(combinations(objective_names, 2))
    print(f"Generating {len(pairs)} pairwise Pareto plots...")
    
    for obj1, obj2 in pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        colors = plt.cm.tab10.colors
        
        for pi, prompt_data in enumerate(results):
            prompt = prompt_data.get("prompt", f"Prompt {pi}")
            generations = prompt_data.get("generations", [])
            baseline_seeds = prompt_data.get("baseline_seeds", [])
            
            # Group scores by weight configuration
            weight_groups = defaultdict(list)
            for gen in generations:
                scores = gen.get("scores", {})
                if obj1 in scores and obj2 in scores:
                    # Create weight key from all weights
                    weight_keys = sorted([k for k in gen.keys() if k.endswith("_weight")])
                    weight_tuple = tuple(round(gen.get(k, 0), 3) for k in weight_keys)
                    weight_groups[weight_tuple].append((scores[obj1], scores[obj2]))
            
            if not weight_groups:
                continue
            
            # Compute mean and std per weight configuration
            mean_points = []
            std_points = []
            for weight_tuple, score_list in weight_groups.items():
                scores_arr = np.array(score_list)
                mean_points.append((np.mean(scores_arr[:, 0]), np.mean(scores_arr[:, 1])))
                std_points.append((
                    np.std(scores_arr[:, 0]) if len(scores_arr) > 1 else 0,
                    np.std(scores_arr[:, 1]) if len(scores_arr) > 1 else 0
                ))
            
            mean_points = np.array(mean_points)
            std_points = np.array(std_points)
            color = colors[pi % len(colors)]
            label = prompt[:30] + "..." if len(prompt) > 30 else prompt
            
            # Plot mean points with error bars
            ax.errorbar(mean_points[:, 0], mean_points[:, 1],
                        xerr=std_points[:, 0], yerr=std_points[:, 1],
                        fmt='o', c=color, markersize=8, alpha=0.8,
                        ecolor=color, elinewidth=1, capsize=2,
                        label=label, zorder=2)
            
            # Compute and draw Pareto front
            pareto_indices = compute_pareto_front(mean_points)
            if pareto_indices:
                pareto_points = mean_points[pareto_indices]
                ax.plot(pareto_points[:, 0], pareto_points[:, 1],
                        c=color, linewidth=2, alpha=0.7, zorder=3)
                ax.scatter(pareto_points[:, 0], pareto_points[:, 1],
                           c=[color], s=120, marker='*', edgecolors='black',
                           linewidth=0.5, zorder=4)
            
            # Plot baseline if available
            baseline_scores = []
            for baseline in baseline_seeds:
                b_scores = baseline.get("scores", {})
                if obj1 in b_scores and obj2 in b_scores:
                    baseline_scores.append((b_scores[obj1], b_scores[obj2]))
            
            if baseline_scores:
                baseline_arr = np.array(baseline_scores)
                baseline_mean = baseline_arr.mean(axis=0)
                ax.scatter(baseline_mean[0], baseline_mean[1],
                           c=[color], s=150, marker='X', alpha=0.9,
                           edgecolors='black', linewidth=1.5, zorder=5)
        
        ax.set_xlabel(obj1, fontsize=11, fontweight='bold')
        ax.set_ylabel(obj2, fontsize=11, fontweight='bold')
        ax.set_title(f'{obj1} vs {obj2}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        
        plt.tight_layout()
        safe_name = f"pair_{obj1}_vs_{obj2}.png"
        plt.savefig(os.path.join(plots_dir, safe_name), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"Pairwise Pareto plots saved to: {plots_dir}/")


def load_model(checkpoint_path, base_model, device="cuda", mixed_precision="bf16", num_objectives=2, conditioning_mode="temb_blk_shared",
               block_mod_form="residual", use_pooled_text=False, num_freqs=1, mod_block_fraction=1.0):
    """Load the preference-conditioned model from checkpoint.

    Supports PEFT LoRA checkpoints with pref_mlp and pref_adaln saved via modules_to_save.

    Args:
        checkpoint_path: Path to checkpoint directory containing lora/ subdirectory
        base_model: Base SD3 model id or local path
        device: Device to load model on
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        num_objectives: Number of preference objectives (pref_dim)
    """
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("Loading base pipeline...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(base_model)

    lora_path = os.path.join(checkpoint_path, "lora")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"Checkpoint does not contain lora/: {lora_path}")

    # Extract base transformer config
    base_config_dict = dict(pipeline.transformer.config)
    for key in ["_class_name", "_diffusers_version", "_name_or_path"]:
        base_config_dict.pop(key, None)

    # Read checkpoint metadata (saved by train_cond_nft_sd3.py)
    pref_dim = num_objectives
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        pref_dim = int(meta.get("pref_dim", pref_dim))
        saved_mode = meta.get("conditioning_mode")
        if saved_mode and saved_mode != conditioning_mode:
            print(f"  NOTE: checkpoint metadata has conditioning_mode='{saved_mode}' "
                  f"(CLI had '{conditioning_mode}')")
            conditioning_mode = saved_mode
        if "block_mod_form" in meta:
            block_mod_form = meta["block_mod_form"]
        if "use_pooled_text" in meta:
            use_pooled_text = bool(meta["use_pooled_text"])
        if "num_freqs" in meta:
            num_freqs = int(meta["num_freqs"])
        if "mod_block_fraction" in meta:
            mod_block_fraction = float(meta["mod_block_fraction"])
        print(f"Loaded metadata: pref_dim={pref_dim}, conditioning_mode={conditioning_mode}")
    else:
        # Fallback: try config.json inside lora dir
        ckpt_cfg_path = os.path.join(lora_path, "config.json")
        if os.path.exists(ckpt_cfg_path):
            with open(ckpt_cfg_path, "r") as f:
                ckpt_cfg = json.load(f)
            pref_dim = int(ckpt_cfg.get("pref_dim", pref_dim))
            print(f"Loaded config from checkpoint: pref_dim={pref_dim}")

    # Check if this is a PEFT adapter checkpoint
    if _is_peft_adapter_dir(lora_path):
        # ----- PEFT adapter path -----
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is not installed but checkpoint is a PEFT adapter")

        print(f"Creating SD3Transformer2DModelWithConditioning (pref_dim={pref_dim})")
        transformer = SD3Transformer2DModelWithConditioning(
            **base_config_dict,
            pref_dim=pref_dim,
            block_mod_form=block_mod_form,
            use_pooled_text=use_pooled_text,
            num_freqs=num_freqs,
            mod_block_fraction=mod_block_fraction,
        )

        # Load base SD3 transformer weights
        print("Loading base SD3 transformer weights...")
        base_missing, _ = transformer.load_state_dict(pipeline.transformer.state_dict(), strict=False)
        print(f"  Base load: {len(base_missing)} missing (expected for pref modules)")

        transformer = transformer.to(device)

        # Load PEFT adapter using from_pretrained (reads adapter_config.json from checkpoint)
        print(f"Loading PEFT adapter from: {lora_path}")
        transformer = PeftModel.from_pretrained(transformer, lora_path, is_trainable=False)
        transformer.set_adapter("default")

    else:
        # ----- Diffusers save_pretrained path (legacy) -----
        print(f"Creating SD3Transformer2DModelWithConditioning (pref_dim={pref_dim})")
        transformer = SD3Transformer2DModelWithConditioning(
            **base_config_dict,
            pref_dim=pref_dim,
            block_mod_form=block_mod_form,
            use_pooled_text=use_pooled_text,
            num_freqs=num_freqs,
            mod_block_fraction=mod_block_fraction,
        )

        # Load base SD3 transformer weights
        print("Loading base SD3 transformer weights...")
        base_missing, _ = transformer.load_state_dict(pipeline.transformer.state_dict(), strict=False)
        print(f"  Base load: {len(base_missing)} missing (expected for pref modules)")

        # Load checkpoint weights on top
        print(f"Loading checkpoint weights from: {lora_path}")
        state = _load_state_dict_from_dir(lora_path)
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        
        print(f"Checkpoint load result: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
        if missing:
            pref_missing = [k for k in missing if 'pref_' in k]
            if pref_missing:
                print(f"  WARNING: Missing pref weights: {pref_missing}")
        if unexpected:
            print(f"  Unexpected keys (first 10): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

        transformer = transformer.to(device)

    transformer.eval()
    pipeline.transformer = transformer

    # Summary: confirm which architecture was loaded
    cls_name = type(transformer).__name__
    # Unwrap PEFT if present
    if hasattr(transformer, "base_model") and hasattr(transformer.base_model, "model"):
        cls_name = type(transformer.base_model.model).__name__
    print(f"\n>>> Eval transformer class: {cls_name}")
    print(f">>> block_mod_form={block_mod_form}, use_pooled_text={use_pooled_text}, "
          f"num_freqs={num_freqs}, mod_block_fraction={mod_block_fraction}")
    print()

    # Move other components to device
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.text_encoder_3.to(device, dtype=dtype)
    pipeline.safety_checker = None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return pipeline, dtype


def load_baseline_model(checkpoint_path, base_model, device="cuda", mixed_precision="bf16"):
    """Load the original (non-conditioned) NFT model from checkpoint.
    
    Supports both PEFT adapter checkpoints and diffusers-style save_pretrained checkpoints.
    
    Args:
        checkpoint_path: Path to checkpoint directory containing lora/ subdirectory
        base_model: Base SD3 model id or local path
        device: Device to load model on
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
    """
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("Loading baseline pipeline (non-conditioned)...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(base_model)

    lora_path = os.path.join(checkpoint_path, "lora")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"Checkpoint does not contain lora/: {lora_path}")

    if _is_peft_adapter_dir(lora_path):
        # ----- PEFT adapter path -----
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is not installed but checkpoint is a PEFT adapter (adapter_config.json found)")

        # Configure LoRA (without modules_to_save since no pref_adaln)
        target_modules = [
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        pipeline.transformer = pipeline.transformer.to(device)
        print(f"Loading baseline LoRA weights from: {lora_path}")
        transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
        transformer.load_adapter(lora_path, adapter_name="default", is_trainable=False)
        transformer.set_adapter("default")
        pipeline.transformer = transformer

    else:
        # ----- Diffusers save_pretrained path -----
        print(f"Loading baseline transformer weights from (diffusers-style): {lora_path}")

        # Try from_pretrained first
        from diffusers import SD3Transformer2DModel
        transformer = None
        try:
            transformer = SD3Transformer2DModel.from_pretrained(lora_path)
        except Exception:
            # Fallback: load state dict into base transformer
            state = _load_state_dict_from_dir(lora_path)
            missing, unexpected = pipeline.transformer.load_state_dict(state, strict=False)
            if missing:
                print(f"Missing keys when loading baseline checkpoint: {missing[:8]}{'...' if len(missing) > 8 else ''}")
            if unexpected:
                print(f"Unexpected keys when loading baseline checkpoint: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
            transformer = pipeline.transformer

        pipeline.transformer = transformer

    # Move components to device
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.text_encoder_3.to(device, dtype=dtype)

    pipeline.transformer.to(device)
    pipeline.transformer.eval()
    pipeline.safety_checker = None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return pipeline, dtype


def generate_baseline(
    pipeline,
    prompt,
    num_steps=40,
    guidance_scale=1.0,
    resolution=512,
    dtype=torch.bfloat16,
    device="cuda",
    seed=None,
    solver="flow",
    noise_level=0.7,
    deterministic=True,
):
    """Generate an image with the baseline (non-conditioned) model.
    
    Uses the standard diffusers pipeline call since the baseline transformer
    doesn't support the preference argument.
    """
    
    generator = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    
    use_amp = dtype in (torch.float16, torch.bfloat16)
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
        with torch.no_grad():
            # Use standard pipeline call (no preference argument)
            output = pipeline(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
                height=resolution,
                width=resolution,
            )
            images = output.images
    
    image = images[0].float().cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image), images


def generate_with_preference(
    pipeline,
    prompt,
    preference_weights,
    num_steps=25,
    guidance_scale=1.0,
    resolution=512,
    dtype=torch.bfloat16,
    device="cuda",
    seed=None,
    solver="flow",
    noise_level=0.7,
    deterministic=True,
):
    """Generate an image with specific preference weights.
    
    Default sampling settings match training config for fair evaluation.
    """
    
    # Seeding for reproducibility
    generator = None
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Preference weights as float32 tensor
    pref_tensor = torch.tensor([preference_weights], device=device, dtype=torch.float32)
    
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    
    use_amp = dtype in (torch.float16, torch.bfloat16)
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
        with torch.no_grad():
            # Encode prompts
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, [prompt], max_sequence_length=128
            )
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            
            neg_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, [""], max_sequence_length=128
            )
            neg_prompt_embeds = neg_prompt_embeds.to(device)
            neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.to(device)
            
            # Generate image
            images, _, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
                height=resolution,
                width=resolution,
                noise_level=noise_level,
                deterministic=deterministic,
                solver=solver,
                model_type="sd3",
                preference=pref_tensor,
            )
    
    # Convert to PIL image
    image = images[0].float().cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image), images


def main():
    parser = argparse.ArgumentParser(description="Generate images with preference conditioning")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint directory (e.g., logs/nft/sd3/.../checkpoints/checkpoint-1000)")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-3.5-medium",
                        help="Base SD3 model id or local path. Must match what the checkpoint was trained from.")
    parser.add_argument("--output_dir", type=str, default="./preference_outputs",
                        help="Output directory for generated images")
    parser.add_argument("--prompt", type=str, nargs="+", required=True,
                        help="Text prompt(s) for image generation. Pass multiple prompts separated by spaces, or use quotes for prompts with spaces.")
    parser.add_argument("--num_weights", type=int, default=5,
                        help="Number of preference weight points to sweep")
    
    # Reward objective names (for labeling)
    parser.add_argument("--objective_names", type=str, nargs="+", default=["clipscore", "pickscore"],
                        help="Names of the reward objectives (must match training order)")
    
    # Sampling settings - defaults match training config
    parser.add_argument("--num_steps", type=int, default=40,
                        help="Number of inference steps (training default: 25)")
    parser.add_argument("--solver", type=str, default="flow",
                        help="ODE solver (training default: flow)")
    parser.add_argument("--noise_level", type=float, default=0.7,
                        help="Noise level (training default: 0.7)")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic", default=True,
                        help="Disable deterministic sampling (default: deterministic=True)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resolution of generated images")
    parser.add_argument("--seed", type=int, default=42,
                        help="Starting random seed")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds to run per (prompt, weight) combination")
    
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--compute_scores", action="store_true",
                        help="Compute reward scores for generated images")
    parser.add_argument("--baseline_checkpoint_path", type=str, default=None,
                        help="Path to baseline (non-conditioned) checkpoint for comparison")
    parser.add_argument("--plot", action="store_true",
                        help="Generate Pareto front plot from results (requires --compute_scores)")
    parser.add_argument("--plot_per_prompt", action="store_true",
                        help="Generate separate Pareto front plot for each prompt (requires --compute_scores)")
    parser.add_argument("--custom_weights", type=float, nargs="+", default=None,
                        help="Custom preference weights to use instead of sweep (e.g., --custom_weights 1.0 1.0)")
    parser.add_argument("--no_normalize_weights", action="store_true",
                        help="Don't normalize custom weights to sum to 1 (for extrapolation experiments)")
    parser.add_argument("--weight_mode", type=str, default="sweep",
                        choices=["sweep", "corners", "simplex_grid"],
                        help="Weight generation mode: 'sweep' (default) linearly interpolates between objectives "
                             "(pairwise edges for >2 objectives), 'corners' generates weights that maximize each "
                             "objective in turn, 'simplex_grid' generates a regular grid over the full simplex "
                             "including interior points where 3+ objectives mix (grid resolution controlled by "
                             "--num_weights)")
    parser.add_argument("--simplex_include_edges", action="store_true",
                        help="With --weight_mode simplex_grid, also include pairwise edge sweeps "
                             "(default: only interior + vertex points)")
    parser.add_argument("--plot_pairwise", action="store_true",
                        help="Generate pairwise 2D Pareto plots for all objective pairs (useful for >2 objectives)")
    parser.add_argument("--conditioning_mode", type=str, default="temb_blk_shared",
                        help="Conditioning mode used during training (must match checkpoint)")
    parser.add_argument("--block_mod_form", type=str, default="affine",
                        choices=["affine", "scale_only", "shift_only", "residual"],
                        help="Block modulation form (ablation modes only)")
    parser.add_argument("--use_pooled_text", action="store_true",
                        help="Use pooled text embeddings in block projector (ablation modes only)")
    parser.add_argument("--num_freqs", type=int, default=64,
                        help="Number of sinusoidal frequency bands (ablation modes only)")
    parser.add_argument("--mod_block_fraction", type=float, default=1.0,
                        help="Fraction of blocks that receive modulation (ablation modes only)")

    args = parser.parse_args()
    
    # Expand objective names for multi-output scorers
    # e.g., "sketch_6obj_decomposed" -> ["pickscore_photorealism", "sketch_style", ...]
    expanded_objective_names = []
    for name in args.objective_names:
        if name in MULTI_OUTPUT_SCORERS:
            expanded_objective_names.extend(MULTI_OUTPUT_SCORERS[name])
        else:
            expanded_objective_names.append(name)
    
    num_objectives = len(expanded_objective_names)
    prompts = args.prompt  # Already a list from nargs="+"
    
    print(f"Objective names (input): {args.objective_names}")
    print(f"Objective names (expanded): {expanded_objective_names}")
    print(f"Number of objectives: {num_objectives}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    pipeline, dtype = load_model(
        args.checkpoint_path,
        base_model=args.base_model,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_objectives=num_objectives,
        conditioning_mode=args.conditioning_mode,
        block_mod_form=args.block_mod_form,
        use_pooled_text=args.use_pooled_text,
        num_freqs=args.num_freqs,
        mod_block_fraction=args.mod_block_fraction,
    )
    
    # Load baseline model if specified
    baseline_pipeline = None
    baseline_dtype = None
    if args.baseline_checkpoint_path:
        print("\nLoading baseline model for comparison...")
        baseline_pipeline, baseline_dtype = load_baseline_model(
            args.baseline_checkpoint_path,
            base_model=args.base_model,
            device=args.device,
            mixed_precision=args.mixed_precision,
        )
    
    # Load scoring function if needed
    scoring_fn = None
    if args.compute_scores:
        print("Loading reward models...")
        # Use original objective_names for multi_score (it handles expansion internally)
        score_dict = {name: 1.0 for name in args.objective_names}
        scoring_fn = multi_score(args.device, score_dict)
    
    # Generate preference weights
    weights_list = []
    if args.custom_weights is not None:
        # Use custom weights, normalized to sum to 1 (training preferences were on the simplex)
        if len(args.custom_weights) != num_objectives:
            raise ValueError(f"--custom_weights must have {num_objectives} values, got {len(args.custom_weights)}")
        w = np.array(args.custom_weights, dtype=np.float32)
        w = np.clip(w, 1e-6, None)  # Avoid zeros
        w = (w / w.sum()).tolist()  # Normalize to simplex
        print(f"Custom weights (normalized): {w}")
        weights_list.append(w)
    elif args.weight_mode == "corners":
        # Corner weights: maximize each objective in turn
        # Each corner has one objective at ~0.95 and others share ~0.05
        print(f"Generating {num_objectives} corner weights (one per objective)")
        for i in range(num_objectives):
            floor = 0.01  # Small floor to avoid numerical edge cases
            w = [floor] * num_objectives
            w[i] = 1.0 - floor * (num_objectives - 1)  # Dominant weight
            # Normalize to ensure sum = 1
            total = sum(w)
            w = [x / total for x in w]
            weights_list.append(w)
            print(f"  Corner {i} ({expanded_objective_names[i]}): {[f'{x:.3f}' for x in w]}")
    elif args.weight_mode == "simplex_grid":
        # Regular grid over the full simplex, including interior points where
        # 3+ objectives are simultaneously nonzero.
        # Grid resolution N = num_weights: generates all (i1/N, i2/N, ..., ik/N)
        # with i1+i2+...+ik = N.  Total points = C(N + k-1, k-1).
        #
        # Example counts (k=4 objectives):
        #   N=4 ->  35 points    N=6 ->  84 points
        #   N=8 -> 165 points    N=10 -> 286 points
        N = args.num_weights
        k = num_objectives

        def _simplex_grid_recursive(n, k, prefix=()):
            """Yield all k-tuples of non-negative ints summing to n."""
            if k == 1:
                yield prefix + (n,)
                return
            for i in range(n + 1):
                yield from _simplex_grid_recursive(n - i, k - 1, prefix + (i,))

        seen = set()
        for indices in _simplex_grid_recursive(N, k):
            w = [round(i / N, 6) for i in indices]
            key = tuple(w)
            if key not in seen:
                seen.add(key)
                weights_list.append(w)

        if args.simplex_include_edges:
            # Also add the dense pairwise edge sweeps (same as "sweep" mode
            # for >2 objectives) so we get smooth 1D sliders along each edge
            # even if the grid resolution is coarser than num_weights.
            from itertools import combinations as _combs
            for a, b in _combs(range(k), 2):
                for j in range(N):
                    t = j / max(N - 1, 1)
                    w = [0.0] * k
                    w[a] = round(1.0 - t, 6)
                    w[b] = round(t, 6)
                    key = tuple(w)
                    if key not in seen:
                        seen.add(key)
                        weights_list.append(w)

        # Sort: descending lex (matches slider gallery ordering)
        weights_list.sort(key=lambda w: [-x for x in w])

        n_interior = sum(1 for w in weights_list if sum(x > 1e-6 for x in w) >= 3)
        print(f"Simplex grid: N={N}, {k} objectives -> {len(weights_list)} points "
              f"({n_interior} with 3+ objectives active)")
        for w in weights_list[:8]:
            print(f"  {[f'{x:.3f}' for x in w]}")
        if len(weights_list) > 8:
            print(f"  ... ({len(weights_list) - 8} more)")
    else:
        # Sweep mode: interpolate from first objective to second with exact 0/1 endpoints
        for i in range(args.num_weights):
            w1 = 1.0 - (i / (args.num_weights - 1)) if args.num_weights > 1 else 0.5
            # Allow exact 0 and 1 endpoints
            w1 = max(0.0, min(1.0, w1))
            w2 = 1.0 - w1
            if num_objectives == 2:
                weights_list.append([w1, w2])
            else:
                # For >2 objectives, generate pairwise sweeps along each edge
                # of the simplex (num_weights points per pair, deduplicated).
                from itertools import combinations
                seen = set()
                weights_list = []  # reset — ignore outer loop points
                for a, b in combinations(range(num_objectives), 2):
                    for j in range(args.num_weights):
                        t = j / max(args.num_weights - 1, 1)
                        w = [0.0] * num_objectives
                        w[a] = round(1.0 - t, 4)
                        w[b] = round(t, 4)
                        key = tuple(w)
                        if key not in seen:
                            seen.add(key)
                            weights_list.append(w)
                break  # pairwise generation done, exit outer loop
    
    results = []
    
    total_images = len(prompts) * len(weights_list) * args.num_seeds
    print(f"\nGenerating {len(prompts)} prompt(s) x {len(weights_list)} weight(s) x {args.num_seeds} seed(s) = {total_images} images")
    print(f"Output directory: {args.output_dir}\n")
    
    for pi, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        prompt_results = {"prompt": prompt, "generations": [], "baseline": None, "baseline_seeds": []}
        
        # Create subdirectory for this prompt
        prompt_dir = os.path.join(args.output_dir, f"prompt_{pi:03d}")
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Generate baseline images if baseline model is loaded (one per seed for fair comparison)
        if baseline_pipeline is not None:
            # Move baseline back to GPU if it was moved to CPU
            if next(baseline_pipeline.transformer.parameters()).device.type == "cpu":
                print("  Moving baseline model back to GPU...")
                baseline_pipeline.transformer.to(args.device, dtype=baseline_dtype)
                baseline_pipeline.vae.to(args.device, dtype=torch.float32)
                baseline_pipeline.text_encoder.to(args.device, dtype=baseline_dtype)
                baseline_pipeline.text_encoder_2.to(args.device, dtype=baseline_dtype)
                baseline_pipeline.text_encoder_3.to(args.device, dtype=baseline_dtype)
            print(f"  Generating {args.num_seeds} baseline image(s)...")
            
            baseline_scores_list = []
            for si in range(args.num_seeds):
                current_seed = args.seed + si
                baseline_pil, baseline_tensor = generate_baseline(
                    baseline_pipeline,
                    prompt,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    resolution=args.resolution,
                    dtype=baseline_dtype,
                    device=args.device,
                    seed=current_seed,
                    solver=args.solver,
                    noise_level=args.noise_level,
                    deterministic=args.deterministic,
                )
                
                baseline_path = os.path.join(prompt_dir, f"baseline_seed{current_seed:04d}.png")
                baseline_pil.save(baseline_path)
                
                baseline_result = {"image_path": baseline_path, "type": "baseline", "seed": current_seed}
                
                if scoring_fn:
                    scores, _ = scoring_fn(baseline_tensor, [prompt], [{}], only_strict=False)
                    baseline_result["scores"] = {k: float(v[0]) if hasattr(v, '__getitem__') else float(v) 
                                                 for k, v in scores.items() if k != "avg"}
                    baseline_scores_list.append(baseline_result["scores"])
                
                prompt_results["baseline_seeds"].append(baseline_result)
                
                # Clear baseline tensor
                del baseline_tensor
                torch.cuda.empty_cache()
            
            # Set baseline to first seed as representative (includes image_path for HTML display)
            if prompt_results["baseline_seeds"]:
                prompt_results["baseline"] = dict(prompt_results["baseline_seeds"][0])
                prompt_results["baseline"]["type"] = "baseline_representative"
                
                # Also compute and store average scores across seeds
                if baseline_scores_list:
                    avg_baseline_scores = {}
                    for key in baseline_scores_list[0].keys():
                        avg_baseline_scores[key] = float(np.mean([s[key] for s in baseline_scores_list]))
                    prompt_results["baseline_average_scores"] = avg_baseline_scores
                    prompt_results["baseline"]["num_seeds_averaged"] = args.num_seeds
            
            # Move baseline to CPU to free GPU memory for conditioned generation
            print("  Moving baseline model to CPU to free GPU memory...")
            baseline_pipeline.transformer.to("cpu")
            baseline_pipeline.vae.to("cpu")  
            baseline_pipeline.text_encoder.to("cpu")
            baseline_pipeline.text_encoder_2.to("cpu")
            baseline_pipeline.text_encoder_3.to("cpu")
            torch.cuda.empty_cache()
        
        # Generate conditioned images for each weight and seed
        for wi, preference_weights in enumerate(tqdm(weights_list, desc="  Weights", leave=False)):
            for si in range(args.num_seeds):
                current_seed = args.seed + si
                
                pil_image, image_tensor = generate_with_preference(
                    pipeline,
                    prompt,
                    preference_weights,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    resolution=args.resolution,
                    dtype=dtype,
                    device=args.device,
                    seed=current_seed,
                    solver=args.solver,
                    noise_level=args.noise_level,
                    deterministic=args.deterministic,
                )
                
                # Build filename from weights and seed
                weight_parts = [f"{name}{w:.2f}" for name, w in zip(expanded_objective_names, preference_weights)]
                img_filename = "_".join(weight_parts) + f"_seed{current_seed:04d}.png"
                img_path = os.path.join(prompt_dir, img_filename)
                pil_image.save(img_path)
                
                gen_result = {
                    "image_path": img_path,
                    "seed": current_seed,
                }
                # Add individual weights to result
                for name, w in zip(expanded_objective_names, preference_weights):
                    gen_result[f"{name}_weight"] = float(w)
                
                # Compute scores if requested
                if scoring_fn:
                    scores, _ = scoring_fn(image_tensor, [prompt], [{}], only_strict=False)
                    gen_result["scores"] = {k: float(v[0]) if hasattr(v, '__getitem__') else float(v) 
                                            for k, v in scores.items() if k != "avg"}
                
                prompt_results["generations"].append(gen_result)
                
                # Clear GPU memory to prevent OOM during long evaluation runs
                del image_tensor
                torch.cuda.empty_cache()
        
        results.append(prompt_results)
        
        # Save prompt text
        with open(os.path.join(prompt_dir, "prompt.txt"), 'w') as f:
            f.write(prompt)
    
    # Save all results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save HTML gallery
    gallery_path = os.path.join(args.output_dir, "gallery.html")
    with open(gallery_path, 'w') as f:
        f.write(HTML_GALLERY_TEMPLATE)
    
    # Generate Pareto front plot if requested (only works for 2 objectives)
    if args.plot:
        if args.compute_scores:
            if num_objectives == 2:
                plot_path = os.path.join(args.output_dir, "pareto_front.png")
                plot_pareto_front(results_path, plot_path, expanded_objective_names)
            else:
                print(f"\nNote: --plot only supports 2 objectives. Use --plot_pairwise for {num_objectives} objectives.")
        else:
            print("\nWarning: --plot requires --compute_scores to generate Pareto front plot")
    
    # Generate per-prompt Pareto plots if requested (only works for 2 objectives)
    if args.plot_per_prompt:
        if args.compute_scores:
            if num_objectives == 2:
                plot_pareto_front_per_prompt(results_path, args.output_dir, expanded_objective_names)
            else:
                print(f"\nNote: --plot_per_prompt only supports 2 objectives. Use --plot_pairwise for {num_objectives} objectives.")
        else:
            print("\nWarning: --plot_per_prompt requires --compute_scores to generate Pareto plots")
    
    # Generate pairwise Pareto plots for all objective pairs
    if args.plot_pairwise:
        if args.compute_scores:
            plot_pairwise_pareto_fronts(results_path, args.output_dir, expanded_objective_names)
        else:
            print("\nWarning: --plot_pairwise requires --compute_scores to generate Pareto plots")
    
    print(f"\n✓ Done! Results saved to {args.output_dir}")
    print(f"  - {len(prompts) * len(weights_list) * args.num_seeds} conditioned images generated")
    if args.baseline_checkpoint_path:
        print(f"  - {len(prompts) * args.num_seeds} baseline images generated")
    print(f"  - Results JSON: {results_path}")
    print(f"  - HTML Gallery: {gallery_path}")
    if args.plot and args.compute_scores and num_objectives == 2:
        print(f"  - Pareto Front: {os.path.join(args.output_dir, 'pareto_front.png')}")
    if args.plot_per_prompt and args.compute_scores and num_objectives == 2:
        print(f"  - Per-prompt Pareto plots: {os.path.join(args.output_dir, 'pareto_plots/')}")
    if args.plot_pairwise and args.compute_scores:
        print(f"  - Pairwise Pareto plots: {os.path.join(args.output_dir, 'pareto_pairwise/')}")
    print(f"\nTo view the gallery, run:")
    print(f"  cd {args.output_dir} && python -m http.server 8000")
    print(f"  Then open http://localhost:8000/gallery.html")


if __name__ == "__main__":
    main()