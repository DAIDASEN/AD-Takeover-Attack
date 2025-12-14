import argparse
import os
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVision2Seq
from datasets import create_dataloader
from attacker import VideoVerboseAttacker
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Run video-level latency attack")
    parser.add_argument("--data_root", type=str, default="./takeover")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--num_keyframes", type=int, default=3)
    parser.add_argument("--propagation_mode", type=str, default="neighbor")
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--num_videos", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./attack_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}")
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
    except:
        # Fallback for other VLMs
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # DataLoader
    loader = create_dataloader(
        root_dir=args.data_root,
        split=args.split,
        batch_size=1,
        clip_len=args.clip_len,
        num_workers=0 # Simple
    )

    # Attacker
    # LLaVA-1.5 expects the prompt to follow the format: "USER: <image>\n<prompt>\nASSISTANT:"
    # and must include the <image> token.
    prompt_template = "USER: <image>\nYou are an autonomous driving assistant. Describe in detail the current driving scene and whether a human driver should take over control.\nASSISTANT:"
    
    attacker = VideoVerboseAttacker(
        model=model,
        processor=processor,
        prompt=prompt_template,
        epsilon=args.epsilon / 255.0,
        step_size=args.step_size / 255.0,
        num_steps=args.num_steps,
        num_keyframes=args.num_keyframes,
        propagation_mode=args.propagation_mode,
        max_new_tokens=args.max_new_tokens,
        device=device
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    results_log = []
    
    count = 0
    for batch in tqdm(loader):
        if count >= args.num_videos:
            break
            
        video = batch["video"][0] # [T, 3, H, W]
        video_id = batch["video_id"][0]
        
        print(f"Attacking video {video_id}...")
        
        # Run attack
        # Use autocast
        if device == "cuda":
            with torch.amp.autocast('cuda'):
                result = attacker.attack_single_video(video)
        else:
            result = attacker.attack_single_video(video)
            
        # Log
        log_entry = {
            "video_id": video_id,
            "token_len_clean": result["token_len_clean"],
            "token_len_adv": result["token_len_adv"],
            "time_clean": result["time_clean"],
            "time_adv": result["time_adv"],
            "final_loss": result["loss_history"][-1] if result["loss_history"] else 0.0,
            "keyframe_indices": result["keyframe_indices"]
        }
        results_log.append(log_entry)
        print(f"Result: {log_entry}")
        
        # Save artifacts
        vid_out_dir = os.path.join(args.output_dir, video_id)
        os.makedirs(vid_out_dir, exist_ok=True)
        
        # Save delta
        torch.save(result["delta_keyframes"], os.path.join(args.output_dir, f"{video_id}_delta.pt"))
        
        # Save adv frames (as png)
        adv_video = result["adv_video"] # [T, 3, H, W]
        # Convert to [T, H, W, 3] uint8
        adv_video_np = (adv_video.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        
        for i in range(adv_video_np.shape[0]):
            im = Image.fromarray(adv_video_np[i])
            im.save(os.path.join(vid_out_dir, f"frame_{i:03d}_adv.png"))
            
        count += 1

    # Save logs
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results_log, f, indent=2)

if __name__ == "__main__":
    main()
