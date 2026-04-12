import math
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class ScalarGate(nn.Module):
    def __init__(self, init_val: float = 1e-3):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor([init_val]))

    def forward(self, x=None) -> torch.Tensor:
        return self.gate


class SliderProjector(nn.Module):
    """Projector: sinusoidal PE of preference + pooled text → modulation vector.

    Matches KontinuousKontext (Snap Research) released code:
      pref (B, pref_dim) → sin/cos PE → Linear to pe_extender_dim
      concat with pooled_projections (B, pooled_dim) → 4-layer MLP with ReLU → out_dim

    Each pref component gets num_freqs bands of (sin, cos),
    giving pref_dim * num_freqs * 2 total PE dims.
    Default num_freqs=1 matches KK's actual implementation.
    """

    def __init__(self, pref_dim: int, pooled_dim: int, out_dim: int,
                 pe_extender_dim: int = 768,
                 num_freqs: int = 1, last_layer_init_std: float = 0.0,
                 n_layers: int = 4, hidden_dim: int = None):
        super().__init__()
        self.num_freqs = num_freqs
        # Register frequency bands: 2^0, 2^1, ..., 2^(num_freqs-1)
        freqs = torch.pow(2.0, torch.arange(num_freqs, dtype=torch.float32))
        self.register_buffer("freqs", freqs)  # (num_freqs,)
        # PE output: pref_dim * num_freqs * 2 (sin + cos per freq per component)
        pe_dim = pref_dim * num_freqs * 2
        self.pe_extender = nn.Linear(pe_dim, pe_extender_dim)
        # 4-layer MLP: hidden layers use hidden_dim (capped), final layer expands to out_dim.
        # hidden_dim defaults to out_dim for backward compat with single-group configs.
        h = hidden_dim if hidden_dim is not None else out_dim
        in_dim = pe_extender_dim + pooled_dim
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.projector = nn.Sequential(*layers)
        # Set last_layer_init_std > 0 to use small-init on last layer.
        # Default 0.0 keeps standard Kaiming init (matching KK).
        if last_layer_init_std > 0:
            nn.init.normal_(self.projector[-1].weight, mean=0.0, std=last_layer_init_std)
            nn.init.zeros_(self.projector[-1].bias)

    def forward(self, pref: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        # pref: (B, pref_dim), pooled: (B, pooled_dim)
        # Multi-frequency PE: each component × each freq → sin, cos
        # pref_expanded: (B, pref_dim, 1) * freqs: (num_freqs,) → (B, pref_dim, num_freqs)
        scaled = pref.unsqueeze(-1) * self.freqs * math.pi  # (B, pref_dim, num_freqs)
        pe = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (B, pref_dim, num_freqs*2)
        pe = pe.flatten(1)  # (B, pref_dim * num_freqs * 2)
        pe_ext = self.pe_extender(pe)  # (B, pe_extender_dim)
        combined = torch.cat([pe_ext, pooled], dim=-1)  # (B, pe_extender_dim + pooled_dim)
        return self.projector(combined)  # (B, out_dim)


def get_modules_to_save(conditioning_mode: str = "hybrid") -> List[str]:
    mode_to_modules = {
        "hybrid": ["pref_mlp", "pref_adaln", "pref_gate"],
        "adaln_both": ["pref_projector", "pref_post_adaln"],
        "temb": ["pref_mlp", "pref_gate"],
        "adaln": ["pref_adaln"],
        "pooled": ["pref_pooled_proj"],
        "token_concat": ["pref_token_proj"],
    }
    return mode_to_modules.get(conditioning_mode, mode_to_modules["hybrid"])


# ---------------------------------------------------------------------------
# Per-block modulation wrapper
# ---------------------------------------------------------------------------

class JointTransformerBlockWithModulation(nn.Module):
    """JointTransformerBlock with KK-style additive modulation on both streams.

    Expects ``_pref_modulation`` in ``joint_attention_kwargs`` with shape
    ``(B, inner_dim * 4)`` split as ``[img_scale, img_shift, txt_scale, txt_shift]``.

    For ``context_pre_only`` blocks (last block), only image modulation is applied.
    """

    def __init__(self, original_block):
        super().__init__()
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

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = dict(joint_attention_kwargs) if joint_attention_kwargs else {}
        modulation_condn = joint_attention_kwargs.pop("_pref_modulation", None)

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

        # --- AdaLN norm: text stream ---
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            (
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # --- Additive modulation (KK-style, no gates) ---
        if modulation_condn is not None:
            img_scale, img_shift, txt_scale, txt_shift = modulation_condn.chunk(4, dim=-1)
            scale_mlp = scale_mlp + img_scale
            shift_mlp = shift_mlp + img_shift
            if not self.context_pre_only:
                c_scale_mlp = c_scale_mlp + txt_scale
                c_shift_mlp = c_shift_mlp + txt_shift

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

        # --- Text FF ---
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
# Main conditioned transformer
# ---------------------------------------------------------------------------

class SD3Transformer2DModelWithConditioning(SD3Transformer2DModel):

    def __init__(
        self,
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
        pref_dim: int = 2,
        pref_gate_init: float = 1e-3,
        conditioning_mode: str = "hybrid",
        **kwargs,
    ):
        kwargs.pop("conditioning_mode", None)
        kwargs.pop("pref_gate_init", None)

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

        self.register_to_config(
            pref_dim=pref_dim,
            pref_gate_init=pref_gate_init,
            conditioning_mode=conditioning_mode,
        )
        self.pref_dim = pref_dim
        self.conditioning_mode = conditioning_mode
        inner_dim = self.inner_dim

        if conditioning_mode == "hybrid":
            self.pref_mlp = nn.Sequential(
                nn.Linear(pref_dim, inner_dim),
                nn.SiLU(),
                nn.Linear(inner_dim, inner_dim),
            )
            nn.init.normal_(self.pref_mlp[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.pref_mlp[-1].bias)

            self.pref_adaln = nn.Sequential(
                nn.Linear(pref_dim, inner_dim),
                nn.SiLU(),
                nn.Linear(inner_dim, inner_dim * 2),
            )
            nn.init.normal_(self.pref_adaln[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.pref_adaln[-1].bias)

            self.pref_gate = ScalarGate(init_val=pref_gate_init)

        elif conditioning_mode == "adaln_both":
            # Per-block: SliderProjector → inner_dim * 4 for FF scale/shift in both streams
            self.pref_projector = SliderProjector(
                pref_dim=pref_dim,
                pooled_dim=pooled_projection_dim,
                out_dim=inner_dim * 4,
                pe_extender_dim=768,
            )
            self._replace_blocks_with_modulation()
            # Post-block: simple 2-layer MLP (same arch as hybrid's pref_adaln — proven to work)
            self.pref_post_adaln = nn.Sequential(
                nn.Linear(pref_dim, inner_dim),
                nn.SiLU(),
                nn.Linear(inner_dim, inner_dim * 2),
            )
            nn.init.normal_(self.pref_post_adaln[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.pref_post_adaln[-1].bias)

        else:
            raise ValueError(
                f"Unknown conditioning_mode={conditioning_mode!r}. "
                f"Supported: 'hybrid', 'adaln_both'"
            )

    def _replace_blocks_with_modulation(self):
        new_blocks = nn.ModuleList()
        for block in self.transformer_blocks:
            new_blocks.append(JointTransformerBlockWithModulation(block))
        self.transformer_blocks = new_blocks

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states=None,
        joint_attention_kwargs=None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        preference: torch.FloatTensor = None,
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
        pref_scale = None
        pref_shift = None

        if preference is not None:
            if preference.ndim != 2 or preference.shape[0] != temb.shape[0] or preference.shape[1] != self.pref_dim:
                raise ValueError(
                    f"preference must be shape (B, {self.pref_dim}) with B={temb.shape[0]}, "
                    f"got {tuple(preference.shape)}"
                )
            pref = preference.to(device=temb.device, dtype=temb.dtype)

            if self.conditioning_mode == "hybrid":
                gate = self.pref_gate(temb)
                temb = temb + gate * self.pref_mlp(pref)
                pref_mod = self.pref_adaln(pref)
                pref_scale, pref_shift = pref_mod.chunk(2, dim=-1)
                pref_scale = pref_scale.unsqueeze(1)
                pref_shift = pref_shift.unsqueeze(1)

            elif self.conditioning_mode == "adaln_both":
                pooled = pooled_projections.to(device=temb.device, dtype=temb.dtype)
                # Per-block modulation (fine-grained, grows over training)
                modulation = self.pref_projector(pref, pooled)  # (B, inner_dim * 4)
                if joint_attention_kwargs is None:
                    joint_attention_kwargs = {}
                joint_attention_kwargs["_pref_modulation"] = modulation
                # Post-block AdaLN (strong signal, immediate effect)
                post_mod = self.pref_post_adaln(pref)  # (B, inner_dim * 2)
                pref_scale, pref_shift = post_mod.chunk(2, dim=-1)
                pref_scale = pref_scale.unsqueeze(1)
                pref_shift = pref_shift.unsqueeze(1)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # IP-Adapter support
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)
            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        # Transformer blocks
        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        # Output projection
        hidden_states = self.norm_out(hidden_states, temb)

        # Post-block AdaLN (hybrid: pref_adaln, adaln_both: pref_post_adaln)
        if pref_scale is not None:
            hidden_states = hidden_states * (1 + pref_scale) + pref_shift

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
