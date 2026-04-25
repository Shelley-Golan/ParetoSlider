import math
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)


_VALID_BLOCK_MOD_FORMS = frozenset({"affine", "scale_only", "shift_only", "residual"})


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class SliderProjector(nn.Module):
    """Projector: sinusoidal PE of preference + pooled text -> modulation vector.

    pref (B, pref_dim) -> sin/cos PE -> Linear to pe_extender_dim
    concat with pooled_projections (B, pooled_dim) -> 4-layer MLP with ReLU -> out_dim

    Each pref component gets num_freqs bands of (sin, cos),
    giving pref_dim * num_freqs * 2 total PE dims.
    """

    def __init__(self, pref_dim: int, pooled_dim: int, out_dim: int,
                 pe_extender_dim: int = 768,
                 num_freqs: int = 1, last_layer_init_std: float = 0.0,
                 n_layers: int = 4, hidden_dim: int = None):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = torch.pow(2.0, torch.arange(num_freqs, dtype=torch.float32))
        self.register_buffer("freqs", freqs)  # (num_freqs,)
        pe_dim = pref_dim * num_freqs * 2
        self.pe_extender = nn.Linear(pe_dim, pe_extender_dim)
        h = hidden_dim if hidden_dim is not None else out_dim
        in_dim = pe_extender_dim + pooled_dim
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.projector = nn.Sequential(*layers)
        if last_layer_init_std > 0:
            nn.init.normal_(self.projector[-1].weight, mean=0.0, std=last_layer_init_std)
            nn.init.zeros_(self.projector[-1].bias)

    def forward(self, pref: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        scaled = pref.unsqueeze(-1) * self.freqs * math.pi  # (B, pref_dim, num_freqs)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (B, pref_dim, num_freqs*2)
        pe = pe.flatten(1)  # (B, pref_dim * num_freqs * 2)
        pe_ext = self.pe_extender(pe)  # (B, pe_extender_dim)
        combined = torch.cat([pe_ext, pooled], dim=-1)  # (B, pe_extender_dim + pooled_dim)
        return self.projector(combined)  # (B, out_dim)


# ---------------------------------------------------------------------------
# Per-block modulation wrapper (image-stream only)
# ---------------------------------------------------------------------------

class ImageOnlyModulationBlock(nn.Module):
    """Wraps a JointTransformerBlock with image-only preference modulation.

    Reads ``_pref_modulation`` from ``joint_attention_kwargs``.
    Only modulates the image stream, never the text stream.

    Supported ``block_mod_form`` values:
        - ``"affine"``:     adds to both (scale_mlp, shift_mlp), mod dim = inner_dim * 2
        - ``"scale_only"``: adds to scale_mlp only, mod dim = inner_dim
        - ``"shift_only"``: adds to shift_mlp only, mod dim = inner_dim
        - ``"residual"``:   external residual injection after FF residual,
                            multiplied by the block's own ``gate_mlp`` so it
                            participates in the native per-block gating.
                            mod dim = inner_dim
    """

    def __init__(self, original_block, block_mod_form: str = "affine"):
        super().__init__()
        if block_mod_form not in _VALID_BLOCK_MOD_FORMS:
            raise ValueError(
                f"Unknown block_mod_form={block_mod_form!r}. "
                f"Supported: {sorted(_VALID_BLOCK_MOD_FORMS)}"
            )
        self.norm1 = original_block.norm1
        self.norm1_context = original_block.norm1_context
        self.attn = original_block.attn
        self.norm2 = original_block.norm2
        self.ff = original_block.ff

        self.context_pre_only = original_block.context_pre_only
        self.use_dual_attention = getattr(original_block, "use_dual_attention", False)
        self._chunk_size = getattr(original_block, "_chunk_size", None)
        self._chunk_dim = getattr(original_block, "_chunk_dim", 0)

        if not self.context_pre_only:
            self.norm2_context = original_block.norm2_context
            self.ff_context = original_block.ff_context

        if self.use_dual_attention:
            self.attn2 = original_block.attn2

        self.block_mod_form = block_mod_form

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = dict(joint_attention_kwargs) if joint_attention_kwargs else {}
        modulation = joint_attention_kwargs.pop("_pref_modulation", None)

        # --- AdaLN norm: image stream ---
        if self.use_dual_attention:
            (
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp,
                norm_hidden_states2, gate_msa2,
            ) = self.norm1(hidden_states, emb=temb)
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb
            )

        # --- AdaLN norm: text stream (unchanged) ---
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            (
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # --- Image-only modulation (affine / scale_only / shift_only) ---
        if modulation is not None and self.block_mod_form != "residual":
            if self.block_mod_form == "affine":
                img_scale, img_shift = modulation.chunk(2, dim=-1)
                scale_mlp = scale_mlp + img_scale
                shift_mlp = shift_mlp + img_shift
            elif self.block_mod_form == "scale_only":
                scale_mlp = scale_mlp + modulation
            elif self.block_mod_form == "shift_only":
                shift_mlp = shift_mlp + modulation

        # --- Joint attention ---
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        # --- Image FF ---
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            from diffusers.models.attention import _chunked_feed_forward
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Residual injection (after FF), gated by the block's native gate_mlp
        if modulation is not None and self.block_mod_form == "residual":
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * modulation[:, None, :]

        # --- Text FF (identical to original, no modulation) ---
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )

            if self._chunk_size is not None:
                from diffusers.models.attention import _chunked_feed_forward
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Main conditioned transformer (temb_blk_shared only)
# ---------------------------------------------------------------------------

class SD3Transformer2DModelWithConditioning(SD3Transformer2DModel):
    """SD3 Transformer with temb + shared block modulation preference conditioning.

    Preference is injected in two ways:
      1. **Temb path**: pref_mlp projects the preference vector and adds it to the
         time/text embedding (near-identity init via small std on last linear).
      2. **Block modulation**: SliderProjector maps the preference (+ optional pooled
         text) to a single shared modulation vector applied to all active blocks via
         ImageOnlyModulationBlock wrappers.
    """

    def __init__(
        self,
        # --- Standard SD3 params (forwarded to base class) ---
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple = (),
        qk_norm: Optional[str] = None,
        # --- Conditioning params ---
        pref_dim: int = 2,
        pref_gate_init: float = 1e-3,
        conditioning_mode: str = "temb_blk_shared",
        block_mod_form: str = "residual",
        use_pooled_text: bool = False,
        num_freqs: int = 1,
        mod_block_fraction: float = 1.0,
        **kwargs,
    ):
        for key in ("conditioning_mode", "pref_gate_init", "block_mod_form",
                     "use_pooled_text", "num_freqs", "mod_block_fraction"):
            kwargs.pop(key, None)

        super().__init__(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            pooled_projection_dim=pooled_projection_dim,
            out_channels=out_channels,
            pos_embed_max_size=pos_embed_max_size,
            dual_attention_layers=dual_attention_layers,
            qk_norm=qk_norm,
            **kwargs,
        )

        if block_mod_form not in _VALID_BLOCK_MOD_FORMS:
            raise ValueError(
                f"Unknown block_mod_form={block_mod_form!r}. "
                f"Supported: {sorted(_VALID_BLOCK_MOD_FORMS)}"
            )
        if not (0 < mod_block_fraction <= 1.0):
            raise ValueError(
                f"mod_block_fraction must be in (0, 1.0], got {mod_block_fraction}"
            )

        self.register_to_config(
            pref_dim=pref_dim,
            pref_gate_init=pref_gate_init,
            conditioning_mode=conditioning_mode,
            block_mod_form=block_mod_form,
            use_pooled_text=use_pooled_text,
            num_freqs=num_freqs,
            mod_block_fraction=mod_block_fraction,
        )

        self.pref_dim = pref_dim
        self.conditioning_mode = conditioning_mode
        self.block_mod_form = block_mod_form
        self.use_pooled_text = use_pooled_text
        inner_dim = self.inner_dim

        # Temb injection (near-identity via small-init on last linear)
        self.pref_mlp = nn.Sequential(
            nn.Linear(pref_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )
        nn.init.normal_(self.pref_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.pref_mlp[-1].bias)

        # Block modulation via shared SliderProjector
        num_active = max(1, min(int(num_layers * mod_block_fraction), num_layers))
        mod_dim = inner_dim * 2 if block_mod_form == "affine" else inner_dim
        self._mod_dim = mod_dim

        block_to_group = torch.full((num_layers,), -1, dtype=torch.long)
        block_to_group[:num_active] = 0
        self.register_buffer("_block_to_group", block_to_group, persistent=False)

        pooled_dim = pooled_projection_dim if use_pooled_text else 0
        self.pref_blk_projector = SliderProjector(
            pref_dim=pref_dim,
            pooled_dim=pooled_dim,
            out_dim=mod_dim,
            pe_extender_dim=768,
            num_freqs=num_freqs,
            hidden_dim=mod_dim,
        )

        self._replace_blocks_with_image_only_mod(block_mod_form)

    def _replace_blocks_with_image_only_mod(self, block_mod_form: str):
        new_blocks = nn.ModuleList()
        for block in self.transformer_blocks:
            new_blocks.append(ImageOnlyModulationBlock(block, block_mod_form))
        self.transformer_blocks = new_blocks

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        pooled_projections: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states=None,
        joint_attention_kwargs=None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        preference: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        patch_size = self.config.patch_size

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)

        # ---- Preference conditioning ----
        blk_mods = None

        if preference is not None:
            if (preference.ndim != 2
                    or preference.shape[0] != temb.shape[0]
                    or preference.shape[1] != self.pref_dim):
                raise ValueError(
                    f"preference must be shape (B, {self.pref_dim}) with B={temb.shape[0]}, "
                    f"got {tuple(preference.shape)}"
                )
            pref = preference.to(device=temb.device, dtype=temb.dtype)

            # (a) Temb injection
            temb = temb + self.pref_mlp(pref)

            # (b) Block modulation (shared vector across all active blocks)
            B = pref.shape[0]
            if self.use_pooled_text:
                pooled = pooled_projections.to(device=temb.device, dtype=temb.dtype)
            else:
                pooled = torch.zeros(B, 0, device=temb.device, dtype=temb.dtype)

            raw_mod = self.pref_blk_projector(pref, pooled)  # (B, mod_dim)

            num_blocks = len(self.transformer_blocks)
            blk_mods = [None] * num_blocks
            for blk_idx in range(num_blocks):
                g = self._block_to_group[blk_idx].item()
                if g >= 0:
                    blk_mods[blk_idx] = raw_mod  # shared: same vector for all active blocks

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # IP-Adapter support
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)
            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        # ---- Transformer blocks ----
        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers

            block_kwargs = dict(joint_attention_kwargs) if joint_attention_kwargs else {}
            if blk_mods is not None:
                block_kwargs["_pref_modulation"] = blk_mods[index_block]

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    block_kwargs if block_kwargs else None,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=block_kwargs if block_kwargs else None,
                )

            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        # ---- Output projection ----
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Reshape to image
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# PEFT helper
# ---------------------------------------------------------------------------

def get_modules_to_save() -> List[str]:
    """Return the list of module names for PEFT ``modules_to_save``."""
    return ["pref_mlp", "pref_blk_projector"]
