import argparse
import os
import torch
import json
import time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVision2Seq
from attacker import VideoVerboseAttacker

def parse_args():
    parser = argparse.ArgumentParser(description="Test pre-computed video-level latency attack")
    parser.add_argument("--data_root", type=str, default="./takeover")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--results_dir", type=str, default="./attack_results")
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--propagation_mode", type=str, default="neighbor")
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
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()

    # Load results.json to get metadata
    results_path = os.path.join(args.results_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return
        
    with open(results_path, "r") as f:
        results_data = json.load(f)

    # We need an attacker instance just for helper methods (prepare inputs, estimate tokens)
    prompt_template = "USER: <image>\nYou are an autonomous driving assistant. Describe in detail the current driving scene and whether a human driver should take over control.\nASSISTANT:"
    
    attacker = VideoVerboseAttacker(
        model=model,
        processor=processor,
        prompt=prompt_template,
        max_new_tokens=args.max_new_tokens,
        device=device,
        propagation_mode=args.propagation_mode
    )

    print(f"Testing {len(results_data)} videos from {args.results_dir}...")
    
    test_log = []
    
    for entry in tqdm(results_data):
        video_id = entry["video_id"]
        keyframe_indices = entry["keyframe_indices"]
        
        # Load video
        video_path = os.path.join(args.data_root, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found. Skipping.")
            continue
            
        # Replicating loading logic from datasets.py
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            frames = [np.zeros((336, 336, 3), dtype=np.uint8)]
            
        # Sample/Pad to clip_len
        total_frames = len(frames)
        if total_frames >= args.clip_len:
            indices = np.linspace(0, total_frames - 1, args.clip_len, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < args.clip_len:
                frames += frames
            frames = frames[:args.clip_len]
            
        # Transform
        processed_frames = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (336, 336))
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            processed_frames.append(frame)
        
        video_tensor = torch.stack(processed_frames).to(device) # [T, 3, H, W]
        
        # Load Delta
        delta_path = os.path.join(args.results_dir, f"{video_id}_delta.pt")
        if not os.path.exists(delta_path):
            print(f"Warning: Delta {delta_path} not found. Skipping.")
            continue
            
        delta = torch.load(delta_path, map_location=device)
        
        # 1. Clean Inference
        start_time = time.time()
        clean_tokens = attacker._estimate_tokens_for_video(video_tensor)
        time_clean = time.time() - start_time
        
        # 2. Adversarial Inference
        # video_tensor is already on device
        adv_video = attacker._propagate_delta_to_video(video_tensor, keyframe_indices, delta)
        
        start_time = time.time()
        adv_tokens = attacker._estimate_tokens_for_video(adv_video)
        time_adv = time.time() - start_time
        
        print(f"Video {video_id}: Clean {clean_tokens} toks ({time_clean:.2f}s) -> Adv {adv_tokens} toks ({time_adv:.2f}s)")
        
        test_log.append({
            "video_id": video_id,
            "clean_tokens": clean_tokens,
            "adv_tokens": adv_tokens,
            "time_clean": time_clean,
            "time_adv": time_adv,
            "diff_tokens": adv_tokens - clean_tokens
        })

    # Save test results
    with open(os.path.join(args.results_dir, "test_results.json"), "w") as f:
        json.dump(test_log, f, indent=2)
        
    print("Test complete. Results saved to test_results.json")

if __name__ == "__main__":
    main()
