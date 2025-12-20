from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model_utils import load_vision_only_model, pil_to_siglip_tensor


def main():
    ap = argparse.ArgumentParser(description="Extract and save target (anchor) embedding for video feature alignment attack.")
    ap.add_argument("--ckpt_dir", type=str, default=str(Path("..") / "DriveMM" / "ckpt" / "DriveMM"), help="DriveMM checkpoint dir (has config.json).")
    ap.add_argument(
        "--anchor_img",
        type=str,
        default=str(Path("..") / "Verbose_Images" / "imgs" / "COCO_val2014_000000002006.jpg"),
        help="Anchor image path.",
    )
    ap.add_argument("--out", type=str, default="target_embedding.pt", help="Output path (inside Video_Feature_Attack_4060/ by default).")
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "float16", "bfloat16", "float32"], help="Model dtype.")
    ap.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0.")
    ap.add_argument("--noise_std", type=float, default=0.02, help="Additive Gaussian noise std in normalized space (SigLIP ~[-1,1]).")
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "none"], help="Pooling for target embedding.")
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parent
    out_path = (workdir / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_vision_only_model(ckpt_dir=args.ckpt_dir, device=args.device, dtype=args.dtype)

    img_p = Path(args.anchor_img)
    if not img_p.exists():
        raise FileNotFoundError(f"Anchor image not found: {img_p}")

    img = Image.open(str(img_p)).convert("RGB")
    pv = pil_to_siglip_tensor(img, model.image_processor)  # (1,C,H,W) float32

    # add noise as requested (initial noisy anchor)
    if args.noise_std > 0:
        pv = pv + torch.randn_like(pv) * float(args.noise_std)
        pv = pv.clamp(-1.0, 1.0)

    with torch.no_grad():
        target = model.encode_image(pv, pool=args.pool).detach().cpu()

    payload = {
        "target": target,
        "pool": args.pool,
        "anchor_img": str(img_p),
        "noise_std": float(args.noise_std),
        "dtype": str(model.dtype).replace("torch.", ""),
        "mm_hidden_size": getattr(model.config, "mm_hidden_size", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
        "mm_vision_tower": getattr(model.config, "mm_vision_tower", None),
        "mm_projector_type": getattr(model.config, "mm_projector_type", None),
    }
    torch.save(payload, str(out_path))
    print(f"[OK] Saved target embedding to: {out_path}")
    print(f"     target shape: {tuple(target.shape)}  pool={args.pool}")


if __name__ == "__main__":
    main()


