import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

def load_vision_only_model(model_path, device="cuda", dtype=torch.float16):
    """
    加载 Hugging Face 的 LLaVA-Video 模型和处理器。
    为了适应 4060 (8GB显存)，默认尝试 4-bit 量化加载，如果不需要量化可手动移除 load_in_4bit。
    """
    print(f"[INFO] Loading HF LLaVA-Video model from: {model_path}")
    
    try:
        # 尝试加载处理器
        processor = LlavaNextVideoProcessor.from_pretrained(model_path)
        
        # 加载模型 (自动处理显存分配)
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            # 如果显存不足，取消注释下面这行使用 4bit 量化
            # load_in_4bit=True, 
            low_cpu_mem_usage=True
        )
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    return model, processor

def get_vision_embeddings(model, pixel_values):
    """
    专门用于提取视频/图像特征的辅助函数。
    模拟 LLaVA 原有的 encode_images 功能：Vision Tower -> Projector -> Embeddings
    """
    # 确保 pixel_values 类型正确
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)
    
    # 1. 通过 Vision Tower 提取特征
    # LLaVA-NeXT Video 通常直接处理 pixel_values
    outputs = model.vision_tower(pixel_values, output_hidden_states=True)
    
    # 2. 获取最后一层特征
    selected_image_feature = outputs.last_hidden_state
    
    # 3. 通过多模态投影层 (Projector) 映射到 LLM 空间
    image_features = model.multi_modal_projector(selected_image_feature)
    
    return image_features