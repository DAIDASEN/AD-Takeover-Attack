import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import av
from model_utils import load_vision_only_model, get_vision_embeddings

def read_video_pyav(container, indices):
    """
    使用 PyAV 读取指定索引的视频帧
    """
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
    """
    加载视频并进行均匀采样
    """
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    # 均匀采样索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    video = read_video_pyav(container, indices)
    return video # List of numpy arrays (T, H, W, C)

def save_video(frames, output_path, fps=10):
    """
    将 numpy 帧序列保存为视频文件
    frames: numpy array (T, H, W, C) in uint8 0-255
    """
    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    height, width = frames.shape[1], frames.shape[2]
    stream.width = width
    stream.height = height
    # stream.pix_fmt = "yuv420p"

    for frame in frames:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    
    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()

class VideoNormalizer(nn.Module):
    """
    可微的视频归一化层，用于在攻击循环中处理 tensor
    """
    def __init__(self, mean, std):
        super().__init__()
        # 假设输入形状为 (Batch, Time, Channel, Height, Width)
        # 将 mean/std 调整为 (1, 1, 3, 1, 1) 以便广播
        self.register_buffer('mean', torch.tensor(mean).view(1, 1, 3, 1, 1)) 
        self.register_buffer('std', torch.tensor(std).view(1, 1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std

class Pgd_Attack:
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.device = args.device
        self.eps = args.eps / 255.0  # 将 eps 转换到 0-1 空间
        self.alpha = args.alpha / 255.0
        self.steps = args.steps
        
        # 获取 Processor 中的归一化参数
        # 注意：不同模型的 processor 结构可能略有不同，这里适配常见的 CLIP/SigLIP 结构
        if hasattr(processor, "image_processor"):
            mean = processor.image_processor.image_mean
            std = processor.image_processor.image_std
        else:
            # 如果找不到，使用 CLIP 的默认值作为后备
            print("[Warning] Could not find mean/std in processor, using CLIP defaults.")
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            
        self.normalizer = VideoNormalizer(mean, std).to(self.device)

    def forward(self, video_frames_numpy, target_embedding):
        """
        video_frames_numpy: list of np.array (H, W, 3) 原始像素
        target_embedding: 目标特征向量
        """
        
        # 1. 准备原始视频 Tensor (0-1 范围，未归一化)
        # 我们使用 processor 进行 Resize 和 Rescale (0-255 -> 0-1)，但不做 Normalize
        # 因为我们需要在 Normalize 之前加上扰动
        
        text_prompt = "dummy" # 占位符，Processor 需要 text 参数
        
        # 临时修改 processor 配置以跳过归一化
        original_do_norm = self.processor.image_processor.do_normalize
        self.processor.image_processor.do_normalize = False
        self.processor.image_processor.do_rescale = True # 确保转为 0-1 float
        
        try:
            inputs = self.processor(text=text_prompt, videos=list(video_frames_numpy), return_tensors="pt", padding=True)
        finally:
            # 恢复配置，以免影响其他代码
            self.processor.image_processor.do_normalize = original_do_norm
            
        # 获取基础视频 tensor: (Batch, Time, Channel, Height, Width)
        origin_video = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
        
        # 2. 初始化扰动 delta
        delta = torch.zeros_like(origin_video).to(self.device)
        delta.requires_grad = True
        
        # 这里使用 Sign-SGD 风格的更新，或者也可以用 Adam
        # 为了符合标准 PGD，这里演示手动梯度更新
        
        target_embedding = target_embedding.to(self.device, dtype=self.model.dtype).detach()

        print(f"Start Attack... Steps: {self.steps}, Eps: {self.eps*255:.2f}/255, Alpha: {self.alpha*255:.2f}/255")
        
        for step in range(self.steps):
            # 1. 构造对抗样本
            adv_video = origin_video + delta
            
            # 截断到合法图像范围 [0, 1]
            adv_video_clamped = torch.clamp(adv_video, 0.0, 1.0)
            
            # 2. 归一化 (这一步是可微的)
            norm_video = self.normalizer(adv_video_clamped)
            
            # 3. 提取特征
            current_embedding = get_vision_embeddings(self.model, norm_video)
            
            # 4. 计算损失 (MSE: 让当前特征接近目标特征)
            # 注意：对于非目标攻击(Untargeted Attack)，这里应该是 maximize distance
            # 对于目标攻击(Targeted Attack)，这里是 minimize distance (MSE)
            loss = nn.MSELoss()(current_embedding, target_embedding)
            
            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")
                
            # 5. 反向传播
            loss.backward()
            
            # 6. PGD 更新 (Gradient Descent: delta = delta - alpha * sign(grad))
            with torch.no_grad():
                grad = delta.grad.detach()
                delta.data = delta.data - self.alpha * grad.sign()
                
                # 7. 投影 (Projection)
                # 限制扰动幅度在 epsilon 之内
                delta.data = torch.clamp(delta.data, -self.eps, self.eps)
                # 再次限制最终图像在 [0, 1] 之间
                delta.data = torch.clamp(origin_video + delta.data, 0.0, 1.0) - origin_video
                
                # 清零梯度
                delta.grad.zero_()

        # 生成最终对抗样本
        final_adv_video = torch.clamp(origin_video + delta, 0.0, 1.0)
        return final_adv_video

def main():
    parser = argparse.ArgumentParser()
    # 默认使用 HuggingFace 上的 LLaVA-Video 模型
    parser.add_argument("--model-path", type=str, default="llava-hf/LLaVA-NeXT-Video-7B-hf")
    parser.add_argument("--video-path", type=str, required=True, help="Path to input video")
    parser.add_argument("--target-feature-path", type=str, required=True, help="Path to target .pt feature file")
    parser.add_argument("--output-dir", type=str, default="attack_results")
    parser.add_argument("--eps", type=float, default=8.0, help="Perturbation budget (0-255)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Step size (0-255)")
    parser.add_argument("--steps", type=int, default=50, help="Number of attack steps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to sample")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    print(f"Loading model from {args.model_path}...")
    model, processor = load_vision_only_model(args.model_path, device=args.device)
    
    # 2. 加载目标特征
    print(f"Loading target features from {args.target_feature_path}")
    target_embedding = torch.load(args.target_feature_path, map_location=args.device)
    
    # 3. 加载视频
    print(f"Loading video from {args.video_path}")
    video_frames = load_video(args.video_path, num_frames=args.num_frames)
    
    # 4. 初始化攻击器
    attacker = Pgd_Attack(model, processor, args)
    
    # 5. 运行攻击
    adv_video_tensor = attacker.forward(video_frames, target_embedding)
    
    # 6. 保存结果
    # adv_video_tensor 形状为 (B, T, C, H, W)，值域 [0, 1]
    # 需要转换为 (T, H, W, C) 且值域 [0, 255]
    
    # 假设 Batch Size = 1
    adv_video_np = adv_video_tensor.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
    adv_video_uint8 = (adv_video_np * 255).clip(0, 255).astype(np.uint8)
    
    video_name = os.path.basename(args.video_path)
    save_path = os.path.join(args.output_dir, f"adv_{video_name}")
    print(f"Saving adversarial video to {save_path}")
    
    save_video(adv_video_uint8, save_path)
    print("Done.")

if __name__ == "__main__":
    main()