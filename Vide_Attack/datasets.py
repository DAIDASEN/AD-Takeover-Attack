import os
import json
import random
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List
from torch.utils.data import Dataset, DataLoader

class DrivingVideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        clip_len: int = 16,
        transform: Optional[Callable] = None,
        seed: int = 42
    ):
        """
        Args:
            root_dir: directory that contains all .mp4 files and splits.json.
            split: one of {"train", "val", "test"}.
            clip_len: number of frames per video clip.
            transform: torchvision-like transform applied to each frame.
            seed: random seed for splitting.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.clip_len = clip_len
        self.transform = transform
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")

        self.splits_file = self.root_dir / "splits.json"
        self._prepare_splits(seed)
        
        with open(self.splits_file, "r") as f:
            splits = json.load(f)
        
        if split not in splits:
            raise ValueError(f"Split {split} not found in {self.splits_file}")
            
        self.video_filenames = splits[split]
        self.video_paths = [self.root_dir / fn for fn in self.video_filenames]

    def _prepare_splits(self, seed: int):
        if self.splits_file.exists():
            return

        # List all mp4 files
        all_videos = sorted(list(self.root_dir.glob("*.mp4")))
        all_filenames = [p.name for p in all_videos]
        
        if not all_filenames:
            raise RuntimeError(f"No .mp4 files found in {self.root_dir}")

        # Shuffle and split
        random.seed(seed)
        random.shuffle(all_filenames)
        
        n = len(all_filenames)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        # Remaining for test
        
        train_files = all_filenames[:n_train]
        val_files = all_filenames[n_train : n_train + n_val]
        test_files = all_filenames[n_train + n_val :]
        
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        with open(self.splits_file, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Created splits.json with {len(train_files)} train, {len(val_files)} val, {len(test_files)} test videos.")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path = self.video_paths[idx]
        video_id = video_path.stem
        
        frames = self._load_video(str(video_path))
        
        # Sample or pad
        total_frames = len(frames)
        if total_frames >= self.clip_len:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.clip_len, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Loop/pad
            # Simple strategy: loop until enough, then slice
            while len(frames) < self.clip_len:
                frames += frames
            frames = frames[:self.clip_len]
            
        # Apply transforms
        # frames is list of H,W,C (BGR from cv2)
        # Convert to RGB and then Tensor
        
        processed_frames = []
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Transform expects PIL or Tensor usually, but let's assume transform handles it or we do basic conversion
            # The prompt says: "Convert each frame to RGB, apply transform: Resize..., Convert to Tensor..."
            # If transform is provided, we use it. If not, we do basic.
            # But usually transform is a composition.
            # Let's assume transform takes a PIL image or we convert to tensor first.
            # To be safe and standard with torchvision, let's convert to PIL if transform is present, or just Tensor.
            # However, the prompt says "Convert each frame to RGB, apply transform".
            # Let's assume the user passes a transform that accepts a PIL Image or Tensor.
            # For this implementation, I will manually implement the resize and to_tensor if transform is None,
            # or apply transform if it exists.
            
            if self.transform is not None:
                frame = self.transform(frame)
            else:
                # Default behavior if no transform provided (though usually one is passed)
                # Resize to 336x336 (default for LLaVA usually) and to Tensor
                frame = cv2.resize(frame, (336, 336))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            processed_frames.append(frame)
            
        video_tensor = torch.stack(processed_frames) # [T, 3, H, W]
        
        return {
            "video": video_tensor,
            "video_id": video_id,
            "meta": {}
        }

    def _load_video(self, path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            # Handle empty video? Create a black frame
            frames = [np.zeros((336, 336, 3), dtype=np.uint8)]
        return frames

def create_dataloader(
    root_dir: str,
    split: str,
    batch_size: int,
    clip_len: int,
    num_workers: int,
    transform: Optional[Callable] = None
) -> DataLoader:
    dataset = DrivingVideoDataset(
        root_dir=root_dir,
        split=split,
        clip_len=clip_len,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
