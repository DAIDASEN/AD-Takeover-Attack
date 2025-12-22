from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from model_utils import denormalize_siglip, get_cuda_mem_mb, load_vision_only_model, pil_to_siglip_tensor
from PIL import Image


def _find_default_bddx_video() -> Optional[Path]:
    candidates = [
        Path("..") / "BDD-X" / "BDDX_Test" / "results" / "takeover",
        Path("..") / "BDD-X" / "BDDX_Test" / "results" / "no_takeover",
        Path("..") / "BDD-X",
        Path("..") / "data" / "BDDX_Processed" / "videos",
    ]
    for base in candidates:
        base = base.resolve()
        if not base.exists():
            continue
        mp4s = sorted(base.rglob("*.mp4"))
        if mp4s:
            return mp4s[0]
    return None


def read_video_frames_cv2(video_path: Path, num_frames: int = 8, max_size: Optional[int] = None) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # fallback: read sequentially and sample first num_frames
        idxs = list(range(num_frames))
    else:
        idxs = [int(round(i)) for i in np.linspace(0, total - 1, num_frames)]

    frames: List[Image.Image] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if max_size is not None:
            h, w = frame_rgb.shape[:2]
            s = max(h, w)
            if s > max_size:
                scale = max_size / float(s)
                frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def frames_to_pixel_values(frames: List[Image.Image], image_processor) -> torch.Tensor:
    # preprocess per-frame to avoid huge RAM spikes
    pvs = []
    for im in frames:
        pv = pil_to_siglip_tensor(im.convert("RGB"), image_processor)[0]  # (C,H,W)
        pvs.append(pv)
    x = torch.stack(pvs, dim=0)  # (T,C,H,W)
    return x


def save_mp4_from_pixel_values(
    pixel_values_tchw: torch.Tensor,
    image_processor,
    out_path: Path,
    fps: int = 5,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_u8 = denormalize_siglip(pixel_values_tchw, image_processor)  # (T,C,H,W) uint8 RGB
    x_u8 = x_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (T,H,W,3) RGB
    t, h, w, _ = x_u8.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")
    for i in range(t):
        bgr = cv2.cvtColor(x_u8[i], cv2.COLOR_RGB2BGR)
        vw.write(bgr)
    vw.release()


def cosine_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # supports a:(B,D) b:(B,D) or b:(1,D)
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    cos = (a_n * b_n).sum(dim=-1)
    return 1.0 - cos.mean()


def main():
    ap = argparse.ArgumentParser(description="PGD video feature alignment attack (Vision Tower + Projector only).")
    ap.add_argument("--ckpt_dir", type=str, default=str(Path("..") / "DriveMM" / "ckpt" / "DriveMM"))
    ap.add_argument("--target", type=str, default="target_embedding.pt", help="Path to target_embedding.pt (relative to this folder).")
    ap.add_argument("--video", type=str, default=None, help="Input video (.mp4). If not set, auto-pick from BDD-X.")
    ap.add_argument("--frames", type=int, default=8, help="Number of frames to sample for the attack.")
    ap.add_argument("--steps", type=int, default=150, help="PGD steps (100~200 recommended).")
    ap.add_argument("--epsilon", type=float, default=8.0 / 255.0, help="L_inf epsilon in raw pixel space (0~1).")
    ap.add_argument("--step_size", type=float, default=2.0 / 255.0, help="PGD step size in raw pixel space (0~1).")
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out_dir", type=str, default="adv_output")
    ap.add_argument("--save_mp4", action="store_true", help="Also save adversarial video as mp4.")
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parent
    out_dir = workdir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = (workdir / args.target) if not Path(args.target).is_absolute() else Path(args.target)
    if not target_path.exists():
        raise FileNotFoundError(f"Target embedding not found: {target_path}. Run extract_target_embedding.py first.")

    payload = torch.load(str(target_path), map_location="cpu")
    target = payload["target"]  # (1,D) or (1,N,D)
    pool = payload.get("pool", "mean")

    if pool != "mean":
        raise ValueError("attack_main currently expects target pool='mean'. Re-run extract_target_embedding.py with --pool mean.")
    if target.ndim != 2:
        raise ValueError(f"Expected target shape (1,D), got {tuple(target.shape)}")

    model = load_vision_only_model(ckpt_dir=args.ckpt_dir, device=args.device, dtype=args.dtype)

    if args.video is None:
        v = _find_default_bddx_video()
        if v is None:
            raise FileNotFoundError("No default BDD-X mp4 found. Please pass --video path/to.mp4")
        video_path = v
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

    frames = read_video_frames_cv2(video_path, num_frames=args.frames)
    pv_tchw = frames_to_pixel_values(frames, model.image_processor)  # (T,C,H,W) float32 in normalized space
    video = pv_tchw.unsqueeze(0)  # (1,T,C,H,W)

    # Convert epsilon/step_size from raw pixel space -> normalized space.
    # For SigLIP default: pv = (x - 0.5)/0.5, where x in [0,1] => delta_pv = delta_x / 0.5 = 2*delta_x
    eps_norm = float(args.epsilon) * 2.0
    step_norm = float(args.step_size) * 2.0

    original = video.to(device=model.device, dtype=torch.float32).detach()
    adv = original.clone().detach().requires_grad_(True)
    target_dev = target.to(device=model.device, dtype=torch.float32)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        if adv.grad is not None:
            adv.grad.zero_()

        # Forward with autocast to save VRAM; gradients still flow to `adv`.
        with torch.cuda.amp.autocast(enabled=(model.device.type == "cuda"), dtype=model.dtype):
            feats = model.encode_video(adv.to(dtype=model.dtype), pool="mean").to(torch.float32)  # (1,D)
            loss = cosine_loss(feats, target_dev)

        loss.backward()
        grad = adv.grad
        if grad is None:
            raise RuntimeError("No grad on adv input (unexpected).")

        with torch.no_grad():
            adv = adv - step_norm * grad.sign()
            # Project to epsilon-ball around original (L_inf)
            adv = torch.max(torch.min(adv, original + eps_norm), original - eps_norm)
            # Clamp to valid normalized range (SigLIP)
            adv = adv.clamp(-1.0, 1.0)
            adv.requires_grad_(True)

        if step % 10 == 0 or step == 1:
            alloc, reserv = get_cuda_mem_mb()
            peak = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            print(f"[step {step:04d}/{args.steps}] loss={loss.item():.6f}  cuda_alloc={alloc:.1f}MB reserv={reserv:.1f}MB peak={peak:.1f}MB")

    adv_final = adv.detach().cpu()
    adv_pt = out_dir / "adv_video.pt"
    torch.save(
        {
            "adv_pixel_values": adv_final,  # (1,T,C,H,W) normalized
            "orig_pixel_values": original.detach().cpu(),
            "video_path": str(video_path),
            "target_path": str(target_path),
            "epsilon_raw": float(args.epsilon),
            "step_size_raw": float(args.step_size),
            "epsilon_norm": float(eps_norm),
            "step_size_norm": float(step_norm),
            "frames": int(args.frames),
            "steps": int(args.steps),
        },
        str(adv_pt),
    )
    print(f"[OK] Saved adversarial tensor to: {adv_pt}")

    if args.save_mp4:
        adv_mp4 = out_dir / "adv_video.mp4"
        save_mp4_from_pixel_values(adv_final[0], model.image_processor, adv_mp4, fps=args.fps)
        print(f"[OK] Saved adversarial video to: {adv_mp4}")


if __name__ == "__main__":
    main()


