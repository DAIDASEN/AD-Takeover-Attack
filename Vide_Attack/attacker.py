import torch
import numpy as np
import cv2
import time
from typing import List, Dict, Any, Optional
from PIL import Image
from attack_loss import verbose_losses_from_generate_output, total_verbose_loss

def select_keyframes(
    video: torch.Tensor,
    num_keyframes: int = 3,
    strategy: str = "uniform",  # ["uniform", "random", "optical_flow"]
) -> List[int]:
    """
    Args:
        video: Tensor[T, 3, H, W] in [0, 1]
    Returns:
        sorted list of frame indices, e.g. [4, 8, 12]
    """
    T = video.shape[0]
    if T < num_keyframes:
        return [int(i) for i in range(T)]

    if strategy == "uniform":
        indices = np.linspace(0, T - 1, num_keyframes, dtype=int)
        return [int(i) for i in sorted(list(set(indices)))]
    
    elif strategy == "random":
        indices = np.random.choice(T, num_keyframes, replace=False)
        return [int(i) for i in sorted(list(indices))]
    
    elif strategy == "optical_flow":
        # Compute motion score for each frame
        # Convert to numpy, [0, 255], uint8, grayscale
        video_np = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        scores = []
        
        # Score for frame i is flow from i to i+1 (or i-1 to i)
        # Let's define score[i] as flow magnitude between i and i+1
        # For the last frame, use 0 or same as previous
        
        prev_gray = cv2.cvtColor(video_np[0], cv2.COLOR_RGB2GRAY)
        
        motion_scores = np.zeros(T)
        
        for i in range(T - 1):
            curr_gray = cv2.cvtColor(video_np[i+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores[i] = np.mean(mag)
            prev_gray = curr_gray
            
        # Last frame score = 0 (or copy previous)
        motion_scores[-1] = 0
        
        # Pick top-k
        indices = np.argsort(motion_scores)[-num_keyframes:]
        return [int(i) for i in sorted(list(indices))]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

class VideoVerboseAttacker:
    def __init__(
        self,
        model,
        processor,
        prompt: str,
        epsilon: float = 4/255.0,
        step_size: float = 1/255.0,
        num_steps: int = 10,
        num_keyframes: int = 3,
        keyframe_strategy: str = "uniform",   # or "random", "optical_flow"
        propagation_mode: str = "neighbor",   # or "broadcast"
        max_new_tokens: int = 2048,
        device: str = "cuda",
        eos_token_id: Optional[int] = None
    ):
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_keyframes = num_keyframes
        self.keyframe_strategy = keyframe_strategy
        self.propagation_mode = propagation_mode
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        if eos_token_id is None:
            if hasattr(self.model.config, "eos_token_id"):
                self.eos_token_id = self.model.config.eos_token_id
            else:
                # Fallback, maybe 2 for LLaMA/Opt
                self.eos_token_id = 2 
        else:
            self.eos_token_id = eos_token_id

        # Get normalization stats from processor if possible
        try:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor:
                self.mean = torch.tensor(image_processor.image_mean, device=self.device).view(1, 3, 1, 1)
                self.std = torch.tensor(image_processor.image_std, device=self.device).view(1, 3, 1, 1)
            else:
                # Default CLIP stats
                self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
                self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        except Exception:
             # Default CLIP stats
            self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    def attack_single_video(self, video: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            video: Tensor[T, 3, H, W], values in [0,1]
        Returns:
            Dict with results
        """
        video = video.to(self.device)
        T, C, H, W = video.shape

        keyframe_indices = select_keyframes(video, self.num_keyframes, self.keyframe_strategy)
        K = len(keyframe_indices)

        # Initialize delta
        delta = torch.zeros(K, C, H, W, device=self.device)
        delta.requires_grad_(True)

        loss_history = []

        # Baseline tokens
        start_time = time.time()
        baseline_tokens = self._estimate_tokens_for_video(video)
        time_clean = time.time() - start_time

        for step in range(self.num_steps):
            if delta.grad is not None:
                delta.grad.zero_()

            total_loss = 0.0
            step_token_lens = []
            
            # Accumulate gradients over keyframes
            for idx_k, frame_idx in enumerate(keyframe_indices):
                clean_frame = video[frame_idx]  # [3,H,W]
                # Add delta
                adv_frame = (clean_frame + delta[idx_k]).clamp(0.0, 1.0)
                
                # Prepare inputs (differentiable)
                inputs = self._prepare_vlm_inputs(adv_frame)
                
                # 1. Generate sequence (no grad) to get the trajectory
                with torch.no_grad():
                    gen_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=1.0,
                        return_dict_in_generate=True
                    )
                
                # Record token length
                input_len = inputs['input_ids'].shape[1]
                generated_len = gen_outputs.sequences.shape[1] - input_len
                step_token_lens.append(generated_len)
                
                # 2. Forward pass on the generated sequence (with grad)
                full_ids = gen_outputs.sequences
                
                # Extend attention mask if needed
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    new_tokens_len = full_ids.shape[1] - attention_mask.shape[1]
                    if new_tokens_len > 0:
                        ones = torch.ones((attention_mask.shape[0], new_tokens_len), device=self.device, dtype=attention_mask.dtype)
                        attention_mask = torch.cat([attention_mask, ones], dim=1)
                
                # Forward pass to get gradients
                # We use the SAME pixel_values (which has grad)
                # We do NOT pass image_sizes in kwargs to avoid the bug
                out_grad = self.model(
                    input_ids=full_ids,
                    pixel_values=inputs['pixel_values'],
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 3. Extract logits and hidden states for the GENERATED tokens
                # input_len already calculated above
                
                # Logits: logits[t] predicts token at t+1
                # We want predictions for tokens at input_len, input_len+1, ...
                # These are predicted by logits at input_len-1, input_len, ...
                # So we slice logits from input_len-1 to -1
                relevant_logits = out_grad.logits[:, input_len-1 : -1, :]
                
                # Construct fake output for loss function
                class FakeOutput:
                    pass
                fake_output = FakeOutput()
                
                # scores: list of [B, V]
                fake_output.scores = [relevant_logits[:, t, :] for t in range(relevant_logits.shape[1])]
                
                # hidden_states: tuple of tuples
                # out_grad.hidden_states is tuple of [B, L, H] (per layer)
                # We need to slice and transpose
                sliced_layers = [layer[:, input_len:, :] for layer in out_grad.hidden_states]
                T_new = relevant_logits.shape[1]
                
                fake_hidden_states = []
                for t in range(T_new):
                    step_layers = []
                    for layer in sliced_layers:
                        step_layers.append(layer[:, t:t+1, :]) # Keep dim
                    fake_hidden_states.append(tuple(step_layers))
                fake_output.hidden_states = tuple(fake_hidden_states)
                
                loss_eos, loss_unc, loss_div = verbose_losses_from_generate_output(
                    fake_output, eos_token_id=self.eos_token_id
                )
                loss = total_verbose_loss(loss_eos, loss_unc, loss_div)
                
                total_loss = total_loss + loss

            total_loss = total_loss / K
            loss_history.append(total_loss.item())
            
            # Print step progress
            print(f"  Step {step+1}/{self.num_steps} | Loss: {total_loss.item():.4f} | Avg Len: {sum(step_token_lens)/len(step_token_lens):.1f} | Lens: {step_token_lens}")

            # Backward
            if total_loss.requires_grad:
                total_loss.backward()
                if delta.grad is not None:
                    with torch.no_grad():
                        delta -= self.step_size * delta.grad.sign()
                        delta.clamp_(-self.epsilon, self.epsilon)
            
        # Propagate
        adv_video = self._propagate_delta_to_video(video, keyframe_indices, delta)
        
        start_time = time.time()
        adv_tokens = self._estimate_tokens_for_video(adv_video)
        time_adv = time.time() - start_time

        return {
            "adv_video": adv_video.detach().cpu(),
            "keyframe_indices": keyframe_indices,
            "delta_keyframes": delta.detach().cpu(),
            "loss_history": loss_history,
            "token_len_clean": baseline_tokens,
            "token_len_adv": adv_tokens,
            "time_clean": time_clean,
            "time_adv": time_adv,
        }

    def _prepare_vlm_inputs(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frame: [3, H, W] tensor in [0, 1]
        Returns:
            inputs dict for model.generate
        """
        # Normalize
        # frame is [3, H, W]
        # mean, std are [1, 3, 1, 1]
        # We need [1, 3, H, W]
        pixel_values = (frame.unsqueeze(0) - self.mean) / self.std
        
        # Cast to model dtype if needed (e.g. float16)
        pixel_values = pixel_values.to(self.model.dtype)

        # Text inputs
        # We can cache this if prompt is fixed, but processor might handle padding etc.
        # We use processor for text, but manually supply pixel_values
        text_inputs = self.processor(text=self.prompt, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Note: image_sizes is needed for LLaVA's forward pass to handle aspect ratios/padding correctly
        # BUT, it should NOT be passed to the vision tower directly.
        # The LLaVA model's forward method takes image_sizes, but the vision tower inside might not.
        # However, when we pass **inputs to model.generate, it passes everything to model.forward.
        # model.forward (LlavaForConditionalGeneration) takes image_sizes.
        # The error "CLIPVisionModel.forward() got an unexpected keyword argument 'image_sizes'"
        # suggests that image_sizes is being passed down to the vision tower where it shouldn't be.
        # This usually happens if the model architecture passes **kwargs blindly.
        # In recent transformers, LlavaForConditionalGeneration handles this.
        # Let's check if we are using a version where this is an issue.
        # Actually, if we look at the traceback:
        # modeling_llava.py line 190, in get_image_features:
        # image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)
        # This **kwargs includes image_sizes if we passed it to forward!
        # So we need to make sure image_sizes is consumed or not passed to vision_tower.
        # But we can't change the library code.
        # Wait, LLaVA-1.5 uses CLIPVisionModel.
        # If we look at transformers source, get_image_features usually pops image_sizes?
        # Or maybe we shouldn't pass image_sizes in kwargs if the model doesn't support it?
        # But without image_sizes, we got the previous error.
        
        # Workaround: The previous error "Image features and image tokens do not match" 
        # happened because the processor wasn't used to prepare pixel_values, so the model
        # didn't know the original image size or how many patches to expect if it does dynamic stuff.
        # But here we are resizing to 336x336 fixed.
        # The issue might be that we are manually creating pixel_values.
        
        # Let's try to NOT pass image_sizes in the dict, but rely on the fact that we are using standard 336x336.
        # Wait, the previous error said: "tokens: 1, features 2359296".
        # 2359296 features is HUGE. 336*336*3 = 338688 pixels.
        # 2359296 / (336*336) = 20.9...
        # Actually 2359296 is exactly 576 * 4096? No.
        # 576 is 24*24 patches.
        # If we have 1 image, we expect 576 tokens.
        # The error "tokens: 1" means it found 1 <image> token in text, but got features for... something else?
        # Or maybe "features 2359296" is the total number of elements in the tensor?
        # If hidden size is 4096, 2359296 / 4096 = 576.
        # So it has 576 features (correct for 336x336 / 14x14 patch).
        # But "tokens: 1" refers to the number of image tokens in the input_ids?
        # No, LLaVA expands <image> into 576 tokens.
        # If we just provide "USER: <image> ...", that is 1 token in input_ids.
        # The model needs to expand it.
        # The error "Image features and image tokens do not match" usually comes from `get_placeholder_mask`.
        # It checks if the number of image tokens in `input_ids` matches the number of images.
        # We have 1 image and 1 <image> token. That should match.
        
        # The REAL issue with the previous error might be that `pixel_values` shape was wrong?
        # We passed [1, 3, 336, 336].
        # The error "tokens: 1, features 2359296"
        # In `modeling_llava.py`:
        # if image_features.shape[0] != num_images:
        #    raise ValueError(f"Image features and image tokens do not match: tokens: {num_images}, features {image_features.shape[0]}")
        # So it thinks we have 2359296 images??
        # That means `image_features` first dimension is 2359296.
        # That happens if `image_features` is flattened?
        # Or if `pixel_values` was not [B, C, H, W] but something else?
        # We did `pixel_values = (frame.unsqueeze(0) - self.mean) / self.std`.
        # frame is [3, 336, 336]. pixel_values is [1, 3, 336, 336].
        
        # Let's look at the NEW error: `CLIPVisionModel.forward() got an unexpected keyword argument 'image_sizes'`.
        # This confirms that `image_sizes` IS being passed to `vision_tower`.
        # This is a bug in some versions of `transformers` or how we call it.
        # However, we MUST provide `image_sizes` for LLaVA to work correctly with aspect ratio preservation,
        # BUT we are doing fixed resize in `datasets.py`.
        # If we do fixed resize, maybe we don't need `image_sizes` if we disable the aspect ratio logic?
        # But the model logic seems to depend on it.
        
        # Let's try to use the processor to generate the inputs, instead of manual normalization.
        # This ensures all keys are correct.
        # But we need to backprop through the image.
        # Processor returns a tensor. We can't backprop through processor.
        # BUT, we can use processor to get the *correct structure* and then replace `pixel_values` with our differentiable one.
        
        # Let's try this approach:
        # 1. Convert tensor frame back to PIL (or just keep it as tensor if processor supports it).
        # 2. Call processor.
        # 3. Replace `pixel_values` in the result with our `adv_frame` (normalized manually).
        
        # But `adv_frame` is already normalized?
        # No, `adv_frame` in `attack_single_video` is `(clean_frame + delta).clamp(0,1)`.
        # It is in [0, 1].
        # The processor usually does Rescale (x*255) -> Normalize.
        # Or just Normalize if input is 0-1?
        # LLaVA processor usually expects 0-255 PIL or numpy.
        
        # Let's stick to manual normalization but fix the arguments.
        # The error `CLIPVisionModel` getting `image_sizes` suggests we shouldn't pass `image_sizes` to `generate` if it propagates to vision tower.
        # BUT we need it for `LlavaMetaForCausalLM`.
        # The issue is likely that `LlavaModel` passes `**kwargs` to `vision_tower`.
        # We can try to NOT pass `image_sizes` and see if we can fix the "features mismatch" another way.
        
        # Re-visiting "features 2359296".
        # If `image_features` has shape (B*N, ...), and we have 1 image.
        # 2359296 is huge.
        # Wait, 2359296 = 576 * 4096.
        # So `image_features` is [576, 4096] (or [1, 576, 4096] flattened?).
        # If the code expects `image_features.shape[0]` to be `num_images` (which is 1),
        # but it gets 2359296, it means `image_features` is NOT [Batch, Seq, Dim] but maybe [Batch*Seq, Dim]?
        # OR, `image_features` is [1, 576, 4096] and `shape[0]` is 1.
        # Why did it say features 2359296?
        # Maybe `image_features` was flattened to 1D?
        
        # Let's try to use the processor to get the inputs.
        # This is the safest way to get the right keys.
        
        # Convert frame (Tensor [3, H, W] in [0,1]) to PIL for processor
        # We only use this to get the auxiliary keys (like image_sizes, attention_mask etc), 
        # and then we swap in our differentiable pixel_values.
        
        frame_cpu = frame.detach().cpu()
        # Convert to uint8 [0, 255] for processor
        frame_np = (frame_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_pil = Image.fromarray(frame_np)
        
        inputs = self.processor(text=self.prompt, images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Now replace pixel_values with our differentiable version
        # Our frame is [3, H, W] in [0, 1].
        # We need to normalize it using the SAME stats as the processor.
        # We already set self.mean and self.std.
        # pixel_values = (frame.unsqueeze(0) - self.mean) / self.std
        
        # Check if processor did resizing.
        # If processor resized, our `frame` (which is 336x336) matches.
        # If processor did padding, our `frame` might not match if we just normalize.
        # But we forced 336x336 in dataset.
        # LLaVA processor usually resizes to 336x336.
        
        pixel_values = (frame.unsqueeze(0) - self.mean) / self.std
        pixel_values = pixel_values.to(self.model.dtype)
        
        inputs["pixel_values"] = pixel_values
        
        return inputs

    def _estimate_tokens_for_video(self, video: torch.Tensor) -> int:
        T = video.shape[0]
        mid_frame = video[T // 2]
        
        # We don't need grad here
        with torch.no_grad():
            inputs = self._prepare_vlm_inputs(mid_frame)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                return_dict_in_generate=True
            )
        
        # Sequence length
        # outputs.sequences is [batch, seq_len]
        # We want new tokens count.
        # The input prompt length needs to be subtracted.
        input_len = inputs["input_ids"].shape[1]
        total_len = outputs.sequences.shape[1]
        return total_len - input_len

    def _propagate_delta_to_video(self, video, keyframe_indices, delta_keyframes):
        """
        video: [T, 3, H, W]
        keyframe_indices: list of K ints
        delta_keyframes: [K, 3, H, W]
        """
        if self.propagation_mode == "broadcast":
            delta_shared = delta_keyframes.mean(dim=0)  # [3,H,W]
            adv_video = (video + delta_shared.unsqueeze(0)).clamp(0.0, 1.0)
            return adv_video
            
        elif self.propagation_mode == "neighbor":
            adv_video = video.clone()
            T = video.shape[0]
            keyframe_indices = np.array(keyframe_indices)
            
            for t in range(T):
                # Find nearest keyframe
                dists = np.abs(keyframe_indices - t)
                nearest_idx = np.argmin(dists)
                # delta_keyframes[nearest_idx]
                adv_video[t] = (video[t] + delta_keyframes[nearest_idx]).clamp(0.0, 1.0)
            return adv_video
        
        else:
            raise ValueError(f"Unknown propagation mode: {self.propagation_mode}")
