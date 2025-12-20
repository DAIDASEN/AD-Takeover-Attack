from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _add_drivemm_to_syspath() -> Path:
    """
    Ensure we can `import llava` from `DriveMM/llava` without installing as a package.
    """
    repo_root = Path(__file__).resolve().parents[1]
    drivemm_root = repo_root / "DriveMM"
    if str(drivemm_root) not in sys.path:
        sys.path.insert(0, str(drivemm_root))
    return repo_root


def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ["float16", "fp16", "half"]:
        return torch.float16
    if s in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if s in ["float32", "fp32"]:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


def get_cuda_mem_mb() -> Tuple[float, float]:
    """
    Returns (allocated_MB, reserved_MB). If CUDA is not available, returns (0,0).
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserv = torch.cuda.memory_reserved() / (1024**2)
    return float(alloc), float(reserv)


@dataclass
class VisionOnlyModel:
    """
    DriveMM Vision-only runtime:
    - vision tower (SigLIP/CLIP/etc)
    - multimodal projector (mm_projector)

    IMPORTANT: This class intentionally does NOT load any LLM / language model.
    """

    config: Any
    vision_tower: nn.Module
    mm_projector: nn.Module
    image_processor: Any
    device: torch.device
    dtype: torch.dtype

    @torch.no_grad()
    def eval(self) -> "VisionOnlyModel":
        self.vision_tower.eval()
        self.mm_projector.eval()
        return self

    def encode_image_tokens(self, image_tensor_bchw: torch.Tensor) -> torch.Tensor:
        """
        Input: (B,C,H,W) in vision-tower expected normalized space (e.g. SigLIP: [-1,1]).
        Output: (B, N, D) after projector.
        """
        if image_tensor_bchw.ndim != 4:
            raise ValueError(f"encode_image_tokens expects (B,C,H,W), got {tuple(image_tensor_bchw.shape)}")

        x = image_tensor_bchw.to(device=self.device, dtype=self.dtype)
        feats = self.vision_tower(x)  # (B, N, mm_hidden)
        feats = self.mm_projector(feats)  # (B, N, hidden_size)
        return feats

    def encode_image(self, image_tensor_bchw: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """
        Input: (B,C,H,W)
        Output:
          - pool='none' -> (B, N, D)
          - pool='mean' -> (B, D)
        """
        tokens = self.encode_image_tokens(image_tensor_bchw)
        if pool == "none":
            return tokens
        if pool == "mean":
            return tokens.mean(dim=1)
        raise ValueError(f"Unknown pool: {pool}")

    def encode_video_tokens(self, video_tensor_btchw: torch.Tensor) -> torch.Tensor:
        """
        Input: (B,T,C,H,W) in vision-tower expected normalized space.
        Output: (B,T,N,D) after projector.
        """
        if video_tensor_btchw.ndim != 5:
            raise ValueError(f"encode_video_tokens expects (B,T,C,H,W), got {tuple(video_tensor_btchw.shape)}")

        b, t, c, h, w = video_tensor_btchw.shape
        x = video_tensor_btchw.reshape(b * t, c, h, w).to(device=self.device, dtype=self.dtype)
        feats = self.vision_tower(x)  # (B*T, N, mm_hidden)
        feats = self.mm_projector(feats)  # (B*T, N, hidden_size)
        feats = feats.reshape(b, t, feats.shape[1], feats.shape[2])  # (B,T,N,D)
        return feats

    def encode_video(self, video_tensor_btchw: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """
        Input: (B,T,C,H,W)
        Output:
          - pool='none' -> (B,T,N,D)
          - pool='mean' -> (B, D) pooled over (T,N)
          - pool='time_mean' -> (B, N, D) pooled over T only
        """
        tokens = self.encode_video_tokens(video_tensor_btchw)
        if pool == "none":
            return tokens
        if pool == "time_mean":
            return tokens.mean(dim=1)
        if pool == "mean":
            return tokens.mean(dim=(1, 2))
        raise ValueError(f"Unknown pool: {pool}")


def load_vision_only_model(
    ckpt_dir: str = str(Path("DriveMM") / "ckpt" / "DriveMM"),
    device: str = "cuda:0",
    dtype: Optional[str] = None,
    try_load_local_weights: bool = True,
) -> VisionOnlyModel:
    """
    Loads ONLY (vision tower + mm_projector).

    Weight loading strategy:
    - Vision tower is always loaded from HF name in config (e.g. google/siglip-...).
    - If ckpt contains a standalone 'mm_projector.bin', it will be loaded.
    - Else if ckpt contains local safetensors shards (model-000xx-of-000xx.safetensors), we will load only
      the 'model.vision_tower.*' and 'model.mm_projector.*' subsets (requires `safetensors` package).
    - If neither exists, projector stays randomly initialized (still functional for feature-alignment pipeline).
    """

    _add_drivemm_to_syspath()
    from llava.model.multimodal_encoder.builder import build_vision_tower
    from llava.model.multimodal_projector.builder import build_vision_projector

    ckpt_dir_p = Path(ckpt_dir)
    cfg_path = ckpt_dir_p / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")

    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))

    # dtype: prefer caller -> config -> fp16
    if dtype is None:
        dtype = cfg_dict.get("torch_dtype", "float16")
    torch_dtype = _torch_dtype_from_str(dtype)

    cfg = SimpleNamespace(**cfg_dict)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build modules (NO LLM).
    vision_tower = build_vision_tower(cfg, delay_load=False)
    # Ensure the tower is on the right device/dtype.
    vision_tower.to(device=dev, dtype=torch_dtype)
    vision_tower.requires_grad_(False)

    mm_projector = build_vision_projector(cfg, vision_cfg=getattr(vision_tower, "config", None))
    mm_projector.to(device=dev, dtype=torch_dtype)
    mm_projector.requires_grad_(False)

    image_processor = getattr(vision_tower, "image_processor", None)

    # 1) mm_projector.bin (some LLaVA checkpoints ship this)
    mm_projector_bin = ckpt_dir_p / "mm_projector.bin"
    if mm_projector_bin.exists():
        state = torch.load(str(mm_projector_bin), map_location="cpu")
        # try common key styles
        mapped: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("model.mm_projector."):
                mapped[k.replace("model.", "", 1)] = v
            elif k.startswith("mm_projector."):
                mapped[k] = v
        if mapped:
            mm_projector.load_state_dict(
                {k.replace("mm_projector.", "", 1): v.to(dtype=torch_dtype) for k, v in mapped.items() if k.startswith("mm_projector.")},
                strict=False,
            )

    # 2) Optional: partial load from safetensors shards (if present)
    if try_load_local_weights:
        index_json = ckpt_dir_p / "model.safetensors.index.json"
        if index_json.exists():
            idx = json.loads(index_json.read_text(encoding="utf-8"))
            weight_map: Dict[str, str] = idx.get("weight_map", {})
            # We only care about these prefixes:
            wanted_prefixes = ("model.vision_tower.", "model.mm_projector.")
            wanted_keys = [k for k in weight_map.keys() if k.startswith(wanted_prefixes)]
            shard_files = sorted({weight_map[k] for k in wanted_keys})
            shard_paths = [ckpt_dir_p / sf for sf in shard_files if (ckpt_dir_p / sf).exists()]

            if shard_paths:
                try:
                    from safetensors.torch import load_file as safetensors_load_file  # type: ignore
                except Exception:
                    safetensors_load_file = None

                if safetensors_load_file is not None:
                    # Load only required tensors from each shard.
                    # NOTE: this keeps memory usage lower than loading full LLM weights.
                    vt_state: Dict[str, torch.Tensor] = {}
                    proj_state: Dict[str, torch.Tensor] = {}
                    for sp in shard_paths:
                        shard_sd = safetensors_load_file(str(sp), device="cpu")
                        for k in list(shard_sd.keys()):
                            if not k.startswith(wanted_prefixes):
                                continue
                            v = shard_sd[k]
                            k2 = k.replace("model.", "", 1)  # -> vision_tower.* or mm_projector.*
                            if k2.startswith("vision_tower."):
                                vt_state[k2] = v
                            elif k2.startswith("mm_projector."):
                                proj_state[k2] = v

                    if vt_state:
                        vision_tower.load_state_dict(vt_state, strict=False)
                    if proj_state:
                        # mm_projector module expects keys without leading 'mm_projector.'
                        mm_projector.load_state_dict(
                            {k.replace("mm_projector.", "", 1): v for k, v in proj_state.items()},
                            strict=False,
                        )

    return VisionOnlyModel(
        config=cfg,
        vision_tower=vision_tower,
        mm_projector=mm_projector,
        image_processor=image_processor,
        device=dev,
        dtype=torch_dtype,
    ).eval()


def pil_to_siglip_tensor(image_pil, image_processor) -> torch.Tensor:
    """
    Converts a PIL RGB image into a (1,C,H,W) normalized tensor using the model's image_processor.
    """
    out = image_processor.preprocess(image_pil, return_tensors="pt")
    # SigLipImageProcessor returns BatchFeature with list/pt conversion.
    if isinstance(out, dict):
        pv = out["pixel_values"]
    else:
        pv = out.data["pixel_values"]
    if isinstance(pv, list):
        pv = torch.tensor(np.stack(pv, axis=0))
    if pv.ndim == 3:
        pv = pv.unsqueeze(0)
    return pv


def denormalize_siglip(pixel_values: torch.Tensor, image_processor) -> torch.Tensor:
    """
    Inverse of SigLipImageProcessor preprocess (approx): x = (pv*std + mean) * 255
    Input: (T,C,H,W) or (B,T,C,H,W) in normalized space.
    Output: uint8 tensor in RGB, same shape.
    """
    mean = torch.tensor(getattr(image_processor, "image_mean", (0.5, 0.5, 0.5)), device=pixel_values.device, dtype=pixel_values.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor(getattr(image_processor, "image_std", (0.5, 0.5, 0.5)), device=pixel_values.device, dtype=pixel_values.dtype).view(1, 1, 3, 1, 1)

    x = pixel_values
    if x.ndim == 4:
        x = x.unsqueeze(0)
    x = (x * std) + mean
    x = x.clamp(0.0, 1.0) * 255.0
    x = x.round().to(torch.uint8)
    if pixel_values.ndim == 4:
        return x[0]
    return x



