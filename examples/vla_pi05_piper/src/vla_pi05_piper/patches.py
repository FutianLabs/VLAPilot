"""Pi0.5 inference patches: lean loading, TCS smoothing, EMA, guidance toggle.

Three orthogonal smoothing layers:
  1. RTC guidance — gradient-based inpainting inside denoiser (toggleable)
  2. TCS (χ₀)   — linear blend between old/new chunks on ActionQueue merge
  3. EMA         — exponential smoothing on each action popped from queue

Usage:
    from nanobot.vla.pi05_patches import apply_all_patches, reset_state, flush_queue

    apply_all_patches(tcs_min_overlap=8, ema_alpha=0.5, rtc_guidance=True)
    policy = PI05Policy.from_pretrained(path)   # lean loading is patched
    install_warmup(policy)                       # torch.compile warmup
    ...
    flush_queue(action_queue)                    # on task switch
"""

from __future__ import annotations

import gc
import logging
import os
import time as _time

import torch

logger = logging.getLogger(__name__)

# ── Smoothing state (module singleton) ────────────────────────────────────

_state: dict = {"tcs_last_action": None, "ema_prev": None}


def reset_state():
    _state["tcs_last_action"] = None
    _state["ema_prev"] = None


def flush_queue(aq):
    """Reset ActionQueue + smoothing state. Call on task switch."""
    with aq.lock:
        aq.queue = None
        aq.original_queue = None
        aq.last_index = 0
    reset_state()


# ── Lean model loading ────────────────────────────────────────────────────

def _lean_from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
    """Memory-efficient PI05Policy loading: meta device → safetensors → assign."""
    from lerobot.configs.policies import PreTrainedConfig
    from safetensors.torch import load_file

    GemmaRotaryEmbedding = None
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
    except ImportError:
        pass

    if config is None:
        config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
    saved_device = config.device
    config.device = None

    with torch.device("meta"):
        model = cls(config, **kwargs)
    logger.info("Model skeleton created on meta device (0 memory)")

    model_path = str(pretrained_name_or_path)
    model_file = (
        os.path.join(model_path, "model.safetensors") if os.path.isdir(model_path) else model_path
    )
    state_dict = load_file(model_file)

    fixed = model._fix_pytorch_state_dict_keys(state_dict, config)
    remapped = {
        (f"model.{k}" if not k.startswith("model.") else k): v for k, v in fixed.items()
    }
    model.load_state_dict(remapped, strict=False, assign=True)
    del state_dict, fixed, remapped
    gc.collect()

    for _name, module in model.named_modules():
        is_rotary = GemmaRotaryEmbedding is not None and isinstance(module, GemmaRotaryEmbedding)
        if not is_rotary and hasattr(module, "inv_freq") and hasattr(module, "config"):
            is_rotary = True
        if is_rotary:
            module.__init__(module.config, device="cpu")
        if hasattr(module, "position_ids"):
            buf = getattr(module, "position_ids")
            if buf is not None and buf.device == torch.device("meta"):
                module.register_buffer(
                    "position_ids", torch.arange(buf.shape[1]).unsqueeze(0), persistent=False
                )

    meta = torch.device("meta")
    for name, param in list(model.named_parameters()):
        if param.device == meta:
            parts = name.rsplit(".", 1)
            parent = model
            for p in parts[0].split("."):
                parent = getattr(parent, p)
            setattr(
                parent,
                parts[1],
                torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype),
                    requires_grad=param.requires_grad,
                ),
            )
    for name, buf in list(model.named_buffers()):
        if buf.device == meta:
            parts = name.rsplit(".", 1)
            parent = model
            for p in parts[0].split("."):
                parent = getattr(parent, p)
            parent.register_buffer(parts[1], torch.zeros(buf.shape, dtype=buf.dtype))

    config.device = saved_device
    model.eval()
    logger.info("Lean loading complete")
    return model


def patch_lean_loading(*, torch_compile: bool = True):
    """Monkey-patch PI05Policy.from_pretrained with lean loading + torch.compile."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def _patched(cls, pretrained_name_or_path, *, config=None, **kwargs):
        model = _lean_from_pretrained(cls, pretrained_name_or_path, config=config, **kwargs)
        if torch_compile:
            for m in model.modules():
                m._forward_hooks.clear()
            model.model.denoise_step = torch.compile(model.model.denoise_step, mode="default")
            logger.info("torch.compile enabled on denoise_step")
        return model

    PI05Policy.from_pretrained = classmethod(_patched)


# ── Latency tracker warmup ────────────────────────────────────────────────

def patch_latency_warmup(skip: int = 2):
    """Skip first N latency samples from torch.compile warmup."""
    from lerobot.policies.rtc.latency_tracker import LatencyTracker

    _orig_add = LatencyTracker.add

    def _add(self, latency):
        if not hasattr(self, "_warmup_count"):
            self._warmup_count = 0
        if self._warmup_count < skip:
            self._warmup_count += 1
            logger.info("[Warmup] Skip latency %d/%d: %.3fs", self._warmup_count, skip, latency)
            return
        _orig_add(self, latency)

    LatencyTracker.add = _add


# ── TCS (χ₀ Temporal Chunk-wise Smoothing) ────────────────────────────────

def patch_tcs(min_overlap: int = 8):
    """Monkey-patch ActionQueue._replace_actions_queue with χ₀ linear blending.

    On each new chunk arrival, linearly interpolate the overlap region between
    old buffer remainder and new chunk:  w_old=linspace(1→0), w_new=1-w_old.
    Ref: χ₀ (arxiv 2602.09021) Algorithm 1, Sec III-E.
    """
    from lerobot.policies.rtc.action_queue import ActionQueue

    def _replace_with_tcs(self, original_actions, processed_actions, real_delay):
        old_remaining = None
        if self.queue is not None and self.last_index < len(self.queue):
            old_remaining = self.queue[self.last_index:].clone()
        elif _state["tcs_last_action"] is not None:
            old_remaining = _state["tcs_last_action"].unsqueeze(0).expand(min_overlap, -1).clone()

        new_processed = processed_actions[real_delay:].clone()
        self.original_queue = original_actions[real_delay:].clone()

        if old_remaining is not None and len(old_remaining) > 0 and len(new_processed) > 0:
            if len(old_remaining) < min_overlap:
                pad = old_remaining[-1:].expand(min_overlap - len(old_remaining), -1)
                old_remaining = torch.cat([old_remaining, pad], dim=0)

            overlap_len = min(len(old_remaining), len(new_processed))
            if overlap_len > 1:
                w_old = torch.linspace(
                    1.0, 0.0, overlap_len,
                    device=new_processed.device, dtype=new_processed.dtype,
                ).unsqueeze(-1)
                blended = w_old * old_remaining[:overlap_len] + (1 - w_old) * new_processed[:overlap_len]
                if overlap_len < len(new_processed):
                    self.queue = torch.cat([blended, new_processed[overlap_len:]], dim=0)
                else:
                    self.queue = blended
            else:
                self.queue = new_processed
        else:
            self.queue = new_processed

        self.last_index = 0

    ActionQueue._replace_actions_queue = _replace_with_tcs


# ── EMA smoothing ─────────────────────────────────────────────────────────

def patch_ema(alpha: float = 0.5):
    """Monkey-patch ActionQueue.get with per-action EMA. alpha=1.0 disables."""
    from lerobot.policies.rtc.action_queue import ActionQueue

    _orig_get = ActionQueue.get

    def _get_with_ema(self):
        action = _orig_get(self)
        if action is None:
            return None
        _state["tcs_last_action"] = action.clone()
        if alpha < 1.0:
            prev = _state["ema_prev"]
            if prev is not None and prev.shape == action.shape:
                action = alpha * action + (1.0 - alpha) * prev
            _state["ema_prev"] = action.clone()
        return action

    ActionQueue.get = _get_with_ema


# ── RTC guidance toggle ──────────────────────────────────────────────────

def patch_guidance_off():
    """Disable RTC gradient-based guidance (skip autograd). Queue mgmt stays active."""
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    def _denoise_no_guidance(
        self, x_t, prev_chunk_left_over, inference_delay,
        time, original_denoise_step_partial, execution_horizon=None,
    ):
        return original_denoise_step_partial(x_t)

    RTCProcessor.denoise_step = _denoise_no_guidance
    logger.info("RTC guidance DISABLED (no autograd overhead)")


# ── torch.compile warmup ─────────────────────────────────────────────────

def install_warmup(policy):
    """Run 2 warmup passes to JIT-compile the denoiser before real inference."""
    orig = policy.predict_action_chunk
    warmup_done = False

    def _wrapper(*args, **kwargs):
        nonlocal warmup_done
        if not warmup_done:
            logger.info("[Warmup] Pass 1/2 (no prev_chunk)...")
            t0 = _time.perf_counter()
            result = orig(*args, **kwargs)
            logger.info("[Warmup] Pass 1 done in %.1fs", _time.perf_counter() - t0)

            logger.info("[Warmup] Pass 2/2 (with prev_chunk)...")
            t0 = _time.perf_counter()
            kw = dict(kwargs)
            kw["prev_chunk_left_over"] = result.squeeze(0)
            kw["inference_delay"] = 2
            orig(*args[:1], **kw) if args else orig(**kw)
            logger.info("[Warmup] Pass 2 done in %.1fs", _time.perf_counter() - t0)
            warmup_done = True
        return orig(*args, **kwargs)

    policy.predict_action_chunk = _wrapper


# -- torch.compile persistent disk cache ---------------------------------

def enable_compile_cache(cache_dir: str | None = None) -> None:
    """Enable the torch.compile Inductor FX graph disk cache.

    The first run still compiles and writes the cache. Later starts can load
    from disk and save several minutes. PyTorch upgrades or model structure
    changes invalidate the cache automatically and trigger recompilation.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/torchinductor_vla")
    os.makedirs(cache_dir, exist_ok=True)
    # This environment variable must be set before torch._inductor is imported.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    try:
        import torch._inductor.config as _inductor_cfg
        _inductor_cfg.fx_graph_cache = True
        logger.info("torch.compile persistent cache enabled, dir: %s", cache_dir)
    except Exception as e:
        logger.warning("Failed to enable fx_graph_cache (non-fatal): %s", e)


# ── Convenience entry point ──────────────────────────────────────────────

def apply_all_patches(
    *,
    tcs_min_overlap: int = 8,
    ema_alpha: float = 0.5,
    rtc_guidance: bool = True,
    warmup_skip: int = 2,
    torch_compile: bool = True,
    compile_cache_dir: str | None = None,
):
    """Apply all patches at once. Call before loading the model."""
    if torch_compile:
        enable_compile_cache(cache_dir=compile_cache_dir)
    patch_lean_loading(torch_compile=torch_compile)
    patch_latency_warmup(skip=warmup_skip)
    patch_tcs(min_overlap=tcs_min_overlap)
    patch_ema(alpha=ema_alpha)
    if not rtc_guidance:
        patch_guidance_off()
    logger.info(
        "Patches: TCS(overlap=%d) EMA(α=%.2f) guidance=%s compile=%s",
        tcs_min_overlap, ema_alpha, "ON" if rtc_guidance else "OFF", torch_compile,
    )
