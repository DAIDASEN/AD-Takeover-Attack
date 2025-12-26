import argparse
import torch
import os
import av
import numpy as np
from model_utils import load_vision_only_model, get_vision_embeddings

def read_video_pyav(container, indices):
    """读取视频帧的辅助函数"""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def load_video(video_path, num_frames=8):
    """加载视频并采样"""
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    video = read_video_pyav(container, indices)
    return video # List of numpy arrays

def main():
    parser = argparse.ArgumentParser()
    # 默认使用 HF 上的 LLaVA-Video 模型，你也可以换成你的本地路径
    parser.add_argument("--ckpt-dir", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--video-path", type=str, required=True, help="Path to target video")
    parser.add_argument("--save-path", type=str, default="target_embedding.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Extracting features for video: {args.video_path}")

    # 1. 加载模型
    model, processor = load_vision_only_model(args.ckpt_dir, device=args.device)

    # 2. 处理视频
    # 读取原始视频帧 (List of numpy arrays)
    video_frames = load_video(args.video_path, num_frames=8) # 默认取8帧，可根据显存调整
    
    # 使用 HF processor 进行预处理 (归一化、Resize等)
    # text 参数是必须的，但在提取视觉特征时不会用到 input_ids
    conversation = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe this video."}]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(text=prompt, videos=list(video_frames), return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"]

    # 3. 提取特征
    with torch.no_grad():
        target_embedding = get_vision_embeddings(model, pixel_values)

    # 4. 保存
    print(f"Target embedding shape: {target_embedding.shape}")
    torch.save(target_embedding, args.save_path)
    print(f"Saved to {args.save_path}")

if __name__ == "__main__":
    main()