import argparse
import os
import shutil
import torch
import json
import numpy as np
import av
from tqdm import tqdm
from transformers import (
    LlavaNextVideoForConditionalGeneration, 
    LlavaNextVideoProcessor,
    AutoModelForCausalLM, 
    AutoTokenizer
)

# ==============================================================================
# [Prompt 配置]
# ==============================================================================
LLAVA_PROMPT = (
    "Analyze this driving video. A manual takeover is when a driver must intervene "
    "to prevent accidents or system failures. Does this scene require a manual takeover? "
    "Explain your reasoning briefly and then conclude with 'Yes' or 'No'."
)

QWEN_SYSTEM_PROMPT = (
    "You are an expert driving scenario classifier. You will be given a detailed "
    "description of a driving scene. Your task is to determine if the description "
    "implies a manual takeover is required. \n"
    "Output ONLY 'Yes' if a takeover is needed, or 'No' if the scene is safe/automated. "
    "Do not provide any explanation, only the single word."
)

# ==============================================================================
# [工具函数]
# ==============================================================================
def load_video(video_path, num_frames=16):
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        container.seek(0)
        start_index, end_index = indices[0], indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index: break
            if i >= start_index and i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
        return np.stack(frames)
    except Exception as e:
        print(f"[Error] Failed to load {video_path}: {e}")
        return None

class Arbiter:
    def __init__(self, model_path, device):
        print(f"[Init] Loading Arbiter (Qwen): {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device
        )
        self.device = device

    def judge(self, llava_response):
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Description: {llava_response}"}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=10, do_sample=False)
        response = self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        
        clean_res = response.strip().strip('.').lower()
        if "yes" in clean_res: return "Yes"
        if "no" in clean_res: return "No"
        return "Unknown"

# ==============================================================================
# [主程序]
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="videos")
    parser.add_argument("--llava-path", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--qwen-path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=str, default="classified_videos")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    # 初始化目录
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "classification_log.jsonl")
    
    for label in ["takeover", "safe", "unknown"]:
        os.makedirs(os.path.join(args.output_dir, label), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 LLaVA 模型
    print(f"[Init] Loading LLaVA-NeXT-Video: {args.llava_path}")
    processor = LlavaNextVideoProcessor.from_pretrained(args.llava_path, use_fast=True)
    llava_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        args.llava_path, torch_dtype=torch.float16, device_map="auto"
    )

    # 2. 加载 Qwen 仲裁模型
    arbiter = Arbiter(args.qwen_path, device)

    # 3. 获取视频列表
    all_files = sorted([f for f in os.listdir(args.data_root) if f.lower().endswith('.mp4')])
    video_files = all_files[:args.limit]
    print(f"[Data] Processing {len(video_files)} videos.")

    stats = {"Yes": 0, "No": 0, "Unknown": 0}

    # 4. 循环处理
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        for video_name in tqdm(video_files, desc="Classifying"):
            video_full_path = os.path.join(args.data_root, video_name)
            video_frames = load_video(video_full_path)
            
            if video_frames is None:
                continue

            # --- LLaVA 视觉推理 ---
            conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": LLAVA_PROMPT}]}]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=prompt, videos=[list(video_frames)], return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_ids = llava_model.generate(**inputs, max_new_tokens=150, do_sample=False)
            llava_raw_response = processor.decode(output_ids[0], skip_special_tokens=True).strip()

            # --- Qwen 逻辑仲裁 ---
            final_label = arbiter.judge(llava_raw_response)
            stats[final_label] += 1

            # --- 记录对话信息 ---
            log_entry = {
                "video_name": video_name,
                "llava_reasoning": llava_raw_response,
                "arbiter_decision": final_label,
                "status": "success"
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            log_file.flush() # 确保实时写入磁盘

            # --- 执行文件移动 ---
            dest_folder_name = "takeover" if final_label == "Yes" else ("safe" if final_label == "No" else "unknown")
            dest_path = os.path.join(args.output_dir, dest_folder_name, video_name)
            shutil.move(video_full_path, dest_path)

    # 打印最终汇总
    print("\n" + "="*40)
    print(f"Classification Finished.")
    print(f"Log saved to: {log_file_path}")
    print(f"Results: Takeover(Yes): {stats['Yes']}, Safe(No): {stats['No']}, Unknown: {stats['Unknown']}")
    print("="*40)

if __name__ == "__main__":
    main()