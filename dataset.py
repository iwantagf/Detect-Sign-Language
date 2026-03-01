from torch.utils.data import Dataset, random_split, WeightedRandomSampler
from torchvision import models
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
import pickle

def read_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if (len(frames) == 0):
        raise ValueError(f"Khong ton tai video trong: {video_path}")
    
    frames = torch.from_numpy(np.stack(frames, axis = 0))
    return frames

def collate_fn(batch):
    frames = torch.stack([item['frames'] for item in batch], dim = 0) # (B, T, C, H, W)
    labels = torch.tensor([item['label_idx'] for item in batch], dtype=torch.long)
    label_name = [item['label'] for item in batch]
    return {'frames': frames, 'labels': labels, 'label_names': label_name}

class Augmentation:
    def __init__(self, crop_scale = (0.85, 1.0), brightness = 0.2, contrast = 0.2, saturation = 0.2, speed_range = (0.9, 1.1)):
        self.crop_scale = crop_scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.speed_range = speed_range

    def __call__(self, frames):
        frames = self.__speed_augmentation(frames)
        frames = self.__random_resized_crop(frames)
        frames = self.__color_jitter(frames)
        
        return frames
    
    # resample video to a new speed
    def __speed_augmentation(self, frames):
        T = frames.shape[0]
        speed = random.uniform(self.speed_range[0], self.speed_range[1])
        
        new_T = int(T / speed)

        if new_T < 4:
            new_T = 4
        
        if new_T == T:
            return frames
        
        # resample frames using linear interpolation
        indices = torch.linspace(0, T - 1, new_T).long()
        indices = torch.clamp(indices, 0, T - 1)
        frames = frames[indices]

        return frames
    
    def __random_resized_crop(self, frames):
        T, H, W, C = frames.shape
        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        new_H = int(H * scale)
        new_W = int(W * scale)

        top = random.randint(0, H - new_H)
        left = random.randint(0, W - new_W)

        frames = frames[:, top:top + new_H, left:left + new_W, :]
        
        frames = frames.permute(0, 3, 1, 2) # (T, C, H, W)
        frames = F.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)
        frames = frames.permute(0, 2, 3, 1) # (T, H, W, C)
        
        return frames.to(torch.uint8)
    
    def __color_jitter(self, frames):
        T, H, W, C = frames.shape

        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        constrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)

        frames = frames.float()

        frames = frames * brightness_factor
        mean = frames.mean(dim=(1, 2), keepdim=True)
        frames = (frames - mean) * constrast_factor + mean

        gray = frames.mean(dim=-1, keepdim=True)
        frames = gray * (1 - saturation_factor) + frames * saturation_factor

        frames = torch.clamp(frames, 0, 255)

        return frames.to(torch.uint8)

NUM_CLASSES = 100
class VideoDataset(Dataset):
    def __init__(self, root_dir: str, label_to_idx_path: str, transform = None, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225],
                 target_fps = 16, training = False):
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.mean, self.std = mean, std
        self.target_fps = target_fps
        self.training = training

        self.augmentation = Augmentation() if training else None
        
        self.instances, self.labels, self.label_to_idx = [], [], []

        with open(label_to_idx_path, 'rb') as f:
            self.label_mapping = pickle.load(f)

        for label_folder in sorted(os.listdir(root_dir))[:NUM_CLASSES]:
            label_path = os.path.join(root_dir, label_folder)

            if not os.path.isdir(label_path):
                continue
            
            for video_file in os.listdir(label_path):
                video_path = os.path.join(label_path, video_file)
                self.instances.append(video_path)
                self.labels.append(label_folder)
                self.label_to_idx.append(self.label_mapping[label_folder])
        
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        video_path = self.instances[index]
        frames = read_video(video_path)

        if self.training and self.augmentation is not None:
            frames = self.augmentation(frames)
        
        frames = self._downsample_frames(frames)
        frames = self._normalize(frames)

        return {
            'frames': frames, # (T, C, H, W)
            'label_idx': self.label_to_idx[index],
            'label': self.labels[index]
        }

    def _downsample_frames(self, frames):
        # Get target_fps from video
        total = frames.shape[0]

        if total >= self.target_fps:
            indices = torch.linspace(0, total - 1, self.target_fps).long()
        else:
            indices = torch.arange(total)
            pad = self.target_fps - total
            indices = torch.cat([indices, indices[:pad]])
        
        frames = frames[indices]

        if frames.shape[1] != 224 or frames.shape[2] != 224:
            frames = frames.permute(0, 3, 1, 2) # (T, C, H, W)
            frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)
            frames = frames.permute(0, 2, 3, 1) # (T, H, W, C)

        return frames
    
    def _normalize(self, frames):
        frames = frames.float() / 255.0
        mean = torch.tensor(self.mean).view(1, 1, 1, 3)
        std = torch.tensor(self.std).view(1, 1, 1, 3)
        frames = (frames - mean) / std
        frames = frames.permute(0, 3, 1, 2) # (T, C, H, W)

        return frames

def create_balanced_sampler(dataset):
    if hasattr(dataset, 'datasets'):
        labels = [dataset.dataset.label_to_idx[label] for label in dataset.indices]
    else:
        labels = dataset.label_to_idx
    
    class_counts = np.bincount(labels)
    weights = [1.0 / class_counts[label] if class_counts[label] > 0 else 1.0 for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    print(f"Balanced sampler created with class counts min: {class_counts.min()}, max: {class_counts.max()}, mean: {class_counts.mean()}")
    return sampler

