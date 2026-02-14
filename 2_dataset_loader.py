"""
DATASET LOADER: Language-Guided Manipulation Dataset (FINAL FIX)
PyTorch Dataset class with h5py encoding fixes

Author: Assignment Submission
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple


class LanguageGuidedDataset(Dataset):
    """
    Dataset for language-conditioned manipulation.
    
    Returns:
        - images: RGB images (T, C, H, W)
        - language_embedding: Text embedding from CLIP
        - actions: Robot actions (T, action_dim)
    """
    
    def __init__(
        self, 
        hdf5_path: str,
        sequence_length: int = 10,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        only_successful: bool = True
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file containing demonstrations
            sequence_length: Number of timesteps in each sequence
            image_size: Resize images to this size
            normalize: Whether to normalize images
            only_successful: Only use successful demonstrations
        """
        self.hdf5_path = hdf5_path
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.only_successful = only_successful
        
        # Image preprocessing
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        
        if normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.image_transform = transforms.Compose(transform_list)
        
        # Load dataset metadata
        self._load_dataset_info()
        
    def _decode_string(self, s):
        """Helper to decode h5py strings (handles both bytes and strings)"""
        if isinstance(s, bytes):
            return s.decode('utf-8')
        return str(s)
        
    def _load_dataset_info(self):
        """Load dataset and create index of valid sequences"""
        with h5py.File(self.hdf5_path, 'r') as f:
            data_grp = f['data']
            
            self.sequences = []
            
            # Iterate through episodes
            for ep_name in sorted(data_grp.keys()):
                if ep_name.startswith('demo_'):
                    ep_grp = data_grp[ep_name]
                    
                    # Check if episode is successful
                    if self.only_successful and not ep_grp.attrs['success']:
                        continue
                    
                    episode_length = ep_grp.attrs['episode_length']
                    # FIX: Decode language instruction properly
                    language_instruction = self._decode_string(ep_grp.attrs['language_instruction'])
                    
                    # Create sequences from this episode
                    # We'll use sliding window with stride
                    stride = max(1, self.sequence_length // 2)
                    
                    for start_idx in range(0, episode_length - self.sequence_length + 1, stride):
                        self.sequences.append({
                            'episode': ep_name,
                            'start_idx': start_idx,
                            'end_idx': start_idx + self.sequence_length,
                            'language': language_instruction
                        })
        
        print(f"Loaded {len(self.sequences)} sequences from {self.hdf5_path}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns a single sequence.
        
        Returns:
            dict with keys:
                - 'images': (T, C, H, W) tensor
                - 'actions': (T, action_dim) tensor
                - 'language': str
        """
        seq_info = self.sequences[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            ep_grp = f['data'][seq_info['episode']]
            
            # Load images
            images = ep_grp['images'][seq_info['start_idx']:seq_info['end_idx']]
            
            # Load actions
            actions = ep_grp['actions'][seq_info['start_idx']:seq_info['end_idx']]
            
            language = seq_info['language']
        
        # Process images
        processed_images = []
        for img in images:
            # Convert to PIL Image
            img_pil = Image.fromarray(img.astype(np.uint8))
            # Apply transforms
            img_tensor = self.image_transform(img_pil)
            processed_images.append(img_tensor)
        
        images_tensor = torch.stack(processed_images)  # (T, C, H, W)
        actions_tensor = torch.FloatTensor(actions)     # (T, action_dim)
        
        return {
            'images': images_tensor,
            'actions': actions_tensor,
            'language': language
        }


def collate_fn(batch):
    """
    Custom collate function for batching.
    """
    images = torch.stack([item['images'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    languages = [item['language'] for item in batch]
    
    return {
        'images': images,      # (B, T, C, H, W)
        'actions': actions,    # (B, T, action_dim)
        'language': languages  # List of strings
    }


def create_dataloader(
    hdf5_path: str,
    batch_size: int = 8,
    sequence_length: int = 10,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
):
    """
    Create a DataLoader for the dataset.
    """
    dataset = LanguageGuidedDataset(
        hdf5_path=hdf5_path,
        sequence_length=sequence_length,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


# Test script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python 2_dataset_loader.py <path_to_hdf5>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    
    print("Testing dataset loader...")
    print("=" * 80)
    
    # Create dataset
    dataset = LanguageGuidedDataset(
        hdf5_path=hdf5_path,
        sequence_length=10,
        only_successful=True
    )
    
    print(f"Dataset size: {len(dataset)} sequences")
    
    # Test loading one sample
    sample = dataset[0]
    
    print("\nSample data:")
    print(f"  Images shape: {sample['images'].shape}")
    print(f"  Actions shape: {sample['actions'].shape}")
    print(f"  Language: '{sample['language']}'")
    
    # Test dataloader
    dataloader = create_dataloader(
        hdf5_path=hdf5_path,
        batch_size=4,
        sequence_length=10,
        num_workers=0
    )
    
    print("\nTesting DataLoader...")
    for batch in dataloader:
        print(f"  Batch images shape: {batch['images'].shape}")
        print(f"  Batch actions shape: {batch['actions'].shape}")
        print(f"  Number of language instructions: {len(batch['language'])}")
        break
    
    print("\nDataset loader test complete!")
