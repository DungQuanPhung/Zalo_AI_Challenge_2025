"""
ReID (Re-Identification) Model Training - AeroEyes Challenge
Cross-domain matching: Ground images (3 reference images) → Drone footage
Optimized for NVIDIA Jetson deployment

Architecture: Lightweight OSNet or ResNet18 with Metric Learning
Loss: TripletLoss with hard mining or CircleLoss
Dataset: Each video_id = one identity, includes both drone crops + 3 reference images
"""

from html import parser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Tuple, Dict
import argparse
import sys

class TripletLoss(nn.Module):
    """
    Triplet Loss với Hard Mining
    """
    def __init__(self, margin: float = 0.3, mining: str = 'hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining  # 'hard', 'semi-hard', 'all'
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
        """
        # Compute pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        batch_size = embeddings.size(0)
        loss = 0.0
        num_triplets = 0
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Positive mask (same identity)
            pos_mask = labels == anchor_label
            pos_mask[i] = False  # Exclude anchor itself
            
            # Negative mask (different identity)
            neg_mask = labels != anchor_label
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            # Hard positive: furthest positive
            if self.mining == 'hard':
                pos_dist = dist_mat[i][pos_mask].max()
                neg_dist = dist_mat[i][neg_mask].min()
            # Semi-hard: positive < negative < positive + margin
            elif self.mining == 'semi-hard':
                pos_dist = dist_mat[i][pos_mask].mean()
                neg_dists = dist_mat[i][neg_mask]
                semi_hard_negatives = neg_dists[(neg_dists > pos_dist) & 
                                                (neg_dists < pos_dist + self.margin)]
                if len(semi_hard_negatives) > 0:
                    neg_dist = semi_hard_negatives.min()
                else:
                    neg_dist = neg_dists.min()
            else:  # All triplets
                pos_dist = dist_mat[i][pos_mask].mean()
                neg_dist = dist_mat[i][neg_mask].mean()
            
            triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
            
            if triplet_loss > 0:
                loss += triplet_loss
                num_triplets += 1
        
        if num_triplets > 0:
            loss = loss / num_triplets
        
        return loss


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Mining: Chọn hard triplets trong batch
    """
    def __init__(self, margin=0.3, mining='hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(self, embeddings, labels):
        """
        Mining strategy:
        - Hard: Hardest positive, hardest negative
        - Semi-hard: Positive < Negative < Positive + margin
        - All: Average of all triplets
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        batch_size = embeddings.size(0)
        loss = 0.0
        num_triplets = 0
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Positive mask
            pos_mask = (labels == anchor_label)
            pos_mask[i] = False
            
            # Negative mask
            neg_mask = (labels != anchor_label)
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            # ✅ HARD MINING
            if self.mining == 'hard':
                # Hardest positive (furthest positive)
                pos_dist = dist_mat[i][pos_mask].max()
                
                # Hardest negative (closest negative)
                neg_dist = dist_mat[i][neg_mask].min()
            
            # ✅ SEMI-HARD MINING
            elif self.mining == 'semi-hard':
                pos_dist = dist_mat[i][pos_mask].mean()
                
                # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
                neg_dists = dist_mat[i][neg_mask]
                semi_hard_mask = (neg_dists > pos_dist) & \
                                (neg_dists < pos_dist + self.margin)
                
                if semi_hard_mask.any():
                    neg_dist = neg_dists[semi_hard_mask].min()
                else:
                    neg_dist = neg_dists.min()
            
            else:  # All triplets
                pos_dist = dist_mat[i][pos_mask].mean()
                neg_dist = dist_mat[i][neg_mask].mean()
            
            # Triplet loss
            triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
            
            if triplet_loss > 0:
                loss += triplet_loss
                num_triplets += 1
        
        if num_triplets > 0:
            loss = loss / num_triplets
        
        return loss


class ReIDDataset(Dataset):
    """
    Dataset for ReID training with support for both Triplet and CrossEntropy loss
    Each identity = one video_id (includes drone crops + 3 reference images)
    """
    def __init__(self, data_dict: Dict, transform=None, mode='triplet'):
        """
        Args:
            data_dict: {identity_id: [list of image paths]}
            transform: torchvision transforms
            mode: 'triplet' for TripletLoss, 'classification' for CrossEntropy
        """
        self.data_dict = data_dict
        self.identities = list(data_dict.keys())
        self.identity_to_label = {identity: idx for idx, identity in enumerate(self.identities)}
        self.transform = transform
        self.mode = mode
        
        # Flatten for indexing
        self.samples = []
        for identity_id, img_paths in data_dict.items():
            for img_path in img_paths:
                self.samples.append((identity_id, img_path))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            - If mode='triplet': (anchor, positive, negative, anchor_label)
            - If mode='classification': (image, label)
        """
        if self.mode == 'classification':
            # For CrossEntropy loss
            identity, img_path = self.samples[idx]
            img = self.load_image(img_path)
            
            if self.transform:
                img = self.transform(img)
            
            label = self.identity_to_label[identity]
            return img, label
        
        else:
            # For Triplet loss (original behavior)
            # Anchor
            anchor_identity, anchor_path = self.samples[idx]
            anchor_img = self.load_image(anchor_path)
            
            # Positive: random image from same identity
            positive_path = random.choice(self.data_dict[anchor_identity])
            positive_img = self.load_image(positive_path)
            
            # Negative: random image from different identity
            negative_identity = random.choice([i for i in self.identities if i != anchor_identity])
            negative_path = random.choice(self.data_dict[negative_identity])
            negative_img = self.load_image(negative_path)
            
            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            
            return anchor_img, positive_img, negative_img, self.identity_to_label[anchor_identity]
    
    def load_image(self, img_path: str):
        """Load and convert image to RGB"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM)
    Better than AvgPool for small/blurry objects - SOTA for ReID
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                           (x.size(-2), x.size(-1))).pow(1./self.p)


class LightweightReIDModel(nn.Module):
    """
    Lightweight ReID model optimized for Jetson
    Based on ResNet18 (pretrained on ImageNet)
    Modified to support both Triplet Loss and CrossEntropy Loss
    
    IMPROVEMENTS:
    - GeM Pooling instead of AvgPool (better for small objects)
    - Option for MaxPool (preserves strong signals from small objects)
    - Optimized for handling blurry/small images
    """
    def __init__(self, embedding_dim: int = 1024,  # ✅ TĂNG: 512 → 1024
                 num_classes: int = None, 
                 pretrained: bool = True, 
                 pooling: str = 'gem',
                 dropout: float = 0.3  # ✅ TĂNG dropout
                 ):
        super(LightweightReIDModel, self).__init__()
        
        # Base: ResNet50 thay vì ResNet18
        base = models.resnet50(pretrained=pretrained)  # ✅ THAY ĐỔI
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        
        # Pooling
        if pooling == 'gem':
            self.global_pool = GeM()
        elif pooling == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # ✅ MỚI: Thêm BatchNorm sau pooling
        self.bn_neck = nn.BatchNorm1d(2048)  # ResNet50 output = 2048
        self.bn_neck.bias.requires_grad_(False)
        
        # Embedding layer với dropout
        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)  # ✅ Thêm dropout
        )
        
        # Classifier (optional)
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x, return_logits=False):
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # ✅ BatchNorm neck
        x = self.bn_neck(x)
        
        # Embedding
        embedding = self.embedding(x)
        
        if return_logits and hasattr(self, 'classifier'):
            logits = self.classifier(embedding)
            return embedding, logits
        
        return embedding


def create_reid_dataset(train_dir: str = 'train', output_dir: str = 'reid_dataset'):
    """
    Create ReID dataset for Few-Shot Object Tracking
    
    KEY CONCEPT:
    - Each video_id (e.g., Backpack_0) = one identity
    - Identity includes:
      * 3 reference images (object_images - ground view)
      * N drone crops (from video - aerial view)
    - Model learns: "reference images match with drone crops of SAME object"
    
    This enables Few-Shot matching:
    - Training: Learn to match reference → drone for known objects
    - Testing: Apply to NEW objects (BlackBox_0) with same pattern
    
    Args:
        train_dir: Directory containing annotations.json and samples/
        output_dir: Output directory for reid dataset
    
    Returns:
        data_dict: {video_id: [reference_images + drone_crops]}
    """
    train_path = Path(train_dir)
    annotations_path = train_path / 'annotations' / 'annotations.json'
    samples_dir = train_path / 'samples'
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"\n{'='*70}")
    print("Creating ReID Dataset for Few-Shot Learning")
    print(f"{'='*70}")
    print(f"Annotations: {annotations_path}")
    print(f"Samples: {samples_dir}")
    print(f"Output: {output_path}")
    print(f"\nStrategy: Each video_id = one identity")
    print(f"         Reference images (ground) + Drone crops (aerial)")
    print(f"         → Learn cross-domain matching pattern")
    print(f"{'='*70}\n")
    
    data_dict = {}
    
    for video_data in tqdm(annotations, desc="Processing videos"):
        video_id = video_data['video_id']
        
        # Find sample directory
        sample_dir = None
        for d in samples_dir.iterdir():
            if d.is_dir() and video_id in d.name:
                sample_dir = d
                break
        
        if sample_dir is None:
            print(f"⚠️  Sample directory not found for {video_id}, skipping...")
            continue
        
        video_path = sample_dir / 'drone_video.mp4'
        object_images_dir = sample_dir / 'object_images'
        
        if not video_path.exists():
            print(f"⚠️  Video not found for {video_id}, skipping...")
            continue
        
        # Initialize this identity
        data_dict[video_id] = []
        
        # 1. Add 3 reference images (object_images - GROUND VIEW)
        ref_count = 0
        if object_images_dir.exists():
            for img_file in sorted(object_images_dir.glob('*.jpg'))[:3]:
                data_dict[video_id].append(str(img_file))
                ref_count += 1
        
        if ref_count < 3:
            print(f"⚠️  {video_id}: Only {ref_count} reference images (expected 3)")
        
        # 2. Extract crops from drone video (AERIAL VIEW)
        cap = cv2.VideoCapture(str(video_path))
        crop_count = 0
        
        for annotation in video_data.get('annotations', []):
            bboxes = annotation.get('bboxes', [])
            
            # Sample frames to avoid too many similar crops
            # Take every 5th frame for diversity
            sampled_bboxes = bboxes[::5] if len(bboxes) > 10 else bboxes
            
            for bbox_data in sampled_bboxes:
                frame_id = bbox_data['frame']
                x1, y1, x2, y2 = bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2']
                
                # Read specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Add padding to bbox
                h, w = frame.shape[:2]
                pad = 10
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(w, x2 + pad)
                y2_pad = min(h, y2 + pad)
                
                # Crop
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                # Validate crop
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                
                # Save crop
                crop_filename = output_path / f"{video_id}_frame{frame_id:06d}.jpg"
                cv2.imwrite(str(crop_filename), crop)
                data_dict[video_id].append(str(crop_filename))
                crop_count += 1
        
        cap.release()
        
        # Validation
        if len(data_dict[video_id]) < 5:
            print(f"⚠️  {video_id}: Too few images ({len(data_dict[video_id])}), removing...")
            del data_dict[video_id]
        else:
            print(f"✓ {video_id}: {ref_count} reference + {crop_count} drone crops = {len(data_dict[video_id])} total")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print(f"{'='*70}")
    print(f"Total identities (video_ids): {len(data_dict)}")
    
    total_images = 0
    for video_id, img_paths in data_dict.items():
        total_images += len(img_paths)
    
    print(f"Total images: {total_images}")
    print(f"Average images per identity: {total_images / len(data_dict) if data_dict else 0:.1f}")
    print(f"{'='*70}\n")
    
    if len(data_dict) == 0:
        raise ValueError("No data created! Check train/ directory structure and annotations.json")
    
    return data_dict


def create_reid_dataset_from_classified(
    train_dir: str = 'observing/train',
    classified_dir: str = 'classified_objects',
    output_dir: str = 'reid_dataset'
):
    """
    Create ReID dataset from CLASSIFIED data
    
    Uses:
    - Reference images: observing/train/samples/{video_id}/object_images/
    - Positive samples: classified_objects/{video_id}/positive_samples/
    
    Args:
        train_dir: Directory containing reference images
        classified_dir: Directory containing classified objects (positive/negative)
        output_dir: Output directory (not used, data already exists)
    
    Returns:
        data_dict: {video_id: [reference_images + positive_samples]}
    """
    train_path = Path(train_dir)
    classified_path = Path(classified_dir)
    samples_dir = train_path / 'samples'
    
    print(f"\n{'='*70}")
    print("Creating ReID Dataset from CLASSIFIED Objects")
    print(f"{'='*70}")
    print(f"Reference images: {samples_dir}")
    print(f"Positive samples: {classified_path}")
    print(f"\nStrategy: Each video_id = one identity")
    print(f"         Reference images (ground) + Positive drone crops (aerial)")
    print(f"         → Learn cross-domain matching pattern")
    print(f"{'='*70}\n")
    
    data_dict = {}
    
    # Iterate through classified folders
    for video_folder in tqdm(sorted(classified_path.iterdir()), desc="Processing videos"):
        if not video_folder.is_dir():
            continue
        
        video_id = video_folder.name  # e.g., "Backpack_0"
        
        # Find corresponding sample directory for reference images
        sample_dir = samples_dir / video_id
        object_images_dir = sample_dir / 'object_images'
        
        if not object_images_dir.exists():
            print(f"⚠️  Reference images not found for {video_id}, skipping...")
            continue
        
        # Initialize this identity
        data_dict[video_id] = []
        
        # 1. Add reference images (object_images - GROUND VIEW)
        ref_count = 0
        for img_file in sorted(object_images_dir.glob('*.jpg'))[:3]:
            data_dict[video_id].append(str(img_file))
            ref_count += 1
        
        if ref_count < 3:
            print(f"⚠️  {video_id}: Only {ref_count} reference images (expected 3)")
        
        # 2. Add positive samples (classified drone crops - AERIAL VIEW)
        positive_dir = video_folder / 'positive_samples'
        crop_count = 0
        
        if positive_dir.exists():
            # Get all .jpg files (not .json)
            for img_file in sorted(positive_dir.glob('*.jpg')):
                data_dict[video_id].append(str(img_file))
                crop_count += 1
        
        # Validation
        total_images = len(data_dict[video_id])
        if total_images < 5:
            print(f"⚠️  {video_id}: Too few images ({total_images}), removing...")
            del data_dict[video_id]
        else:
            print(f"✓ {video_id}: {ref_count} reference + {crop_count} positive crops = {total_images} total")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print(f"{'='*70}")
    print(f"Total identities (video_ids): {len(data_dict)}")
    
    total_images = 0
    for video_id, img_paths in data_dict.items():
        total_images += len(img_paths)
    
    print(f"Total images: {total_images}")
    print(f"Average images per identity: {total_images / len(data_dict) if data_dict else 0:.1f}")
    print(f"{'='*70}\n")
    
    if len(data_dict) == 0:
        raise ValueError("No data created! Check classified_objects/ directory")
    
    return data_dict


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss - Tốt hơn cho ReID
    Paper: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    
    Ưu điểm:
    - Tạo embedding space tốt hơn
    - Tăng inter-class distance
    - Giảm intra-class distance
    """
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Get theta
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # Add angular margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = torch.cos(theta + self.m * one_hot)
        output *= self.s
        
        return F.cross_entropy(output, labels)

class CenterLoss(nn.Module):
    """
    Center Loss: Giảm intra-class variance
    Paper: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """
    def __init__(self, num_classes, embedding_dim, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        
        # Centers của mỗi class
        self.centers = nn.Parameter(
            torch.randn(num_classes, embedding_dim)
        )
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
        """
        batch_size = embeddings.size(0)
        
        # Get centers for batch
        centers_batch = self.centers[labels]  # [batch_size, embedding_dim]
        
        # Compute distances to centers
        loss = (embeddings - centers_batch).pow(2).sum() / batch_size
        
        return loss
class CombinedLoss(nn.Module):
    """
    ArcFace + Center Loss
    Best combination for discrimination
    """
    def __init__(self, embedding_dim, num_classes, 
                 arcface_s=30.0, arcface_m=0.50,
                 center_alpha=0.5, center_weight=0.001):
        super().__init__()
        
        self.arcface = ArcFaceLoss(embedding_dim, num_classes, 
                                   s=arcface_s, m=arcface_m)
        self.center = CenterLoss(num_classes, embedding_dim, 
                                alpha=center_alpha)
        self.center_weight = center_weight
    
    def forward(self, embeddings, labels):
        # ArcFace loss
        arcface_loss = self.arcface(embeddings, labels)
        
        # Center loss
        center_loss = self.center(embeddings, labels)
        
        # Combined
        total_loss = arcface_loss + self.center_weight * center_loss
        
        return total_loss, arcface_loss, center_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing: Giảm overconfidence
    """
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, labels):
        """
        Smooth labels:
        - True class: 1 - epsilon
        - Other classes: epsilon / (num_classes - 1)
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot labels
        targets = torch.zeros_like(log_probs).scatter_(
            1, labels.unsqueeze(1), 1
        )
        
        # Smooth
        targets = (1 - self.epsilon) * targets + \
                  self.epsilon / num_classes
        
        loss = (-targets * log_probs).sum(dim=-1).mean()
        
        return loss


def train_reid_model(
    data_dict: Dict,
    output_model_path: str = 'reid_model_best.pth',
    embedding_dim: int = 256,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 0.0001,
    device: str = 'cuda',
    loss_type: str = 'arcface',
    input_size: Tuple[int, int] = (256, 128),
    pooling: str = 'gem',
    margin: float = 0.50,
    scale: float = 30.0
):
    """Train ReID model with improved loss"""
    H, W = input_size
    aspect = W / H

    # ✅ AUGMENTATION CỰC MẠNH cho Hard Cases
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((H, W)),

        # Random Crop (simulate occlusion nhưng vẫn giữ aspect đúng)
        transforms.RandomResizedCrop(
            (H, W),
            scale=(0.7, 1.0),
            ratio=(0.8 * aspect, 1.2 * aspect)  # tương đương 0.4–0.6 khi H=256, W=128
        ),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),

        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.25
        ),

        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        transforms.RandomErasing(
            p=0.6,
            scale=(0.02, 0.5),
            ratio=(0.3, 3.3)
        )
    ])
    
    # ✅ FIX: Dataset mode based on loss_type
    if loss_type in ['crossentropy', 'arcface', 'combined']:
        dataset_mode = 'classification'
        num_classes = len(data_dict)
    else:
        dataset_mode = 'triplet'
        num_classes = None
    
    # Dataset and DataLoader
    dataset = ReIDDataset(data_dict, transform=train_transforms, mode=dataset_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Model with optimized pooling
    model = LightweightReIDModel(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        pretrained=True,
        pooling=pooling
    )
    model = model.to(device)
    
    # ✅ FIX: Setup loss function correctly
    if loss_type == 'combined':
        criterion = CombinedLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            arcface_s=scale,
            arcface_m=margin,
            center_weight=0.001
        ).to(device)
        
        # ✅ CRITICAL: Separate optimizer for centers
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': lr, 'weight_decay': 5e-4},
            {'params': criterion.arcface.parameters(), 'lr': lr},
            {'params': criterion.center.parameters(), 'lr': lr * 10}
        ])
        
        print(f"✓ Using CombinedLoss (ArcFace + Center)")
        
    elif loss_type == 'arcface':
        criterion = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            s=scale,
            m=margin
        ).to(device)
        
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=lr,
            weight_decay=5e-4
        )
        
        print(f"✓ Using ArcFace Loss (s={scale}, m={margin})")
        
    elif loss_type == 'triplet':
        criterion = OnlineTripletLoss(margin=0.3, mining='hard')
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=5e-4
        )
        
        print("✓ Using Triplet Loss (Hard Mining)")
        
    else:  # crossentropy
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=5e-4
        )
        
        print("✓ Using CrossEntropy Loss")
    
    # Learning rate scheduler
    def warmup_cosine_scheduler(epoch, num_epochs, warmup_epochs=5):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine_scheduler(epoch, num_epochs)
    )
    
    print(f"\n{'='*70}")
    print("Training ReID Model")
    print(f"{'='*70}")
    print(f"Model: LightweightReIDModel (ResNet50)")
    print(f"Pooling: {pooling.upper()}")
    print(f"Input size: {input_size}")
    print(f"Loss: {loss_type.upper()}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Num identities: {len(data_dict)}")
    print(f"{'='*70}\n")
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_arcface_loss = 0.0
        epoch_center_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_data in pbar:
            if dataset_mode == 'classification':
                images, labels = batch_data
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # ✅ FIX: Get embeddings (NOT logits)
                embeddings = model(images)
                
                # ✅ CRITICAL: Validate embedding shape
                assert embeddings.shape == (images.size(0), embedding_dim), \
                    f"Expected embedding shape [{images.size(0)}, {embedding_dim}], got {embeddings.shape}"
                
                # ✅ FIX: Different loss handling
                if loss_type == 'combined':
                    total_loss, arcface_loss, center_loss = criterion(embeddings, labels)
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    epoch_arcface_loss += arcface_loss.item()
                    epoch_center_loss += center_loss.item()
                    
                    pbar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'arc': f'{arcface_loss.item():.4f}',
                        'ctr': f'{center_loss.item():.4f}'
                    })
                
                elif loss_type == 'arcface':
                    loss = criterion(embeddings, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                else:  # crossentropy
                    # Need to get logits from classifier
                    _, logits = model(images, return_logits=True)
                    loss = criterion(logits, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            else:  # triplet mode
                anchor, positive, negative, labels = batch_data
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Stack all images
                all_images = torch.cat([anchor, positive, negative], dim=0)
                embeddings = model(all_images)
                
                # Split embeddings
                batch_size_triplet = anchor.size(0)
                anchor_emb = embeddings[:batch_size_triplet]
                positive_emb = embeddings[batch_size_triplet:2*batch_size_triplet]
                negative_emb = embeddings[2*batch_size_triplet:]
                
                # Compute triplet loss
                loss = criterion(embeddings, labels.repeat(3))
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        
        # ✅ Detailed logging
        if loss_type == 'combined':
            avg_arcface = epoch_arcface_loss / len(dataloader)
            avg_center = epoch_center_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  ArcFace: {avg_arcface:.4f}")
            print(f"  Center: {avg_center:.4f}")
            print(f"  LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"✓ Best model saved: {output_model_path} (Loss: {best_loss:.4f})")
    
    print(f"\n{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Model saved: {output_model_path}")
    print(f"{'='*70}\n")
    
    return model


def export_to_onnx(
    model_path: str = 'reid_model_best.pth',
    onnx_path: str = 'reid_model.onnx',
    embedding_dim: int = 256,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 128),
    pooling: str = 'gem',
    num_classes: int = None
):
    """
    Export trained ReID model to ONNX format for Jetson deployment
    
    Args:
        num_classes: Number of classes (required if model was trained with CrossEntropy loss)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print("Exporting to ONNX")
    print(f"{'='*70}")
    print(f"Input model: {model_path}")
    print(f"Output ONNX: {onnx_path}")
    print(f"Input size: {input_size}")
    print(f"Num classes: {num_classes}")
    print(f"Pooling: {pooling}")
    
    # ✅ FIX: Load model with correct num_classes
    try:
        model = LightweightReIDModel(
            embedding_dim=embedding_dim, 
            num_classes=num_classes, 
            pretrained=True, 
            pooling=pooling
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device)
        
        print("✓ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\n🔍 Troubleshooting:")
        print(f"   1. Check if num_classes matches training config")
        print(f"   2. Verify model_path exists: {Path(model_path).exists()}")
        raise
    
    # Dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # ✅ FIX: Test model forward pass first
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
        print(f"✓ Model forward pass successful, output shape: {test_output.shape}")
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        raise
    
    # ✅ FIX: Export with proper error handling
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding'],  # Changed from 'output' to 'embedding'
            dynamic_axes={
                'input': {0: 'batch_size'},
                'embedding': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"✓ ONNX model exported successfully: {onnx_path}")
        
        # ✅ Verify ONNX model
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model validation passed")
        
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        print(f"\n🔍 Common issues:")
        print(f"   1. Unsupported operations in model")
        print(f"   2. Dynamic shapes not properly defined")
        print(f"   3. Missing onnx package: pip install onnx")
        raise
    
    print(f"{'='*70}\n")
    
    # ✅ Print ONNX model info
    print("ONNX Model Information:")
    print(f"  Input: {onnx_model.graph.input[0].name} {input_size}")
    print(f"  Output: {onnx_model.graph.output[0].name} [batch_size, {embedding_dim}]")
    print(f"\nNext step: Convert to TensorRT on Jetson")
    print(f"  trtexec --onnx={onnx_path} --saveEngine=reid_model.engine --fp16")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description='Train ReID Model for AeroEyes')
    
    # ✅ UPDATED: Thêm classified-dir argument
    parser.add_argument('--train-dir', default='observing/train', 
                       help='Training data directory (for reference images)')
    parser.add_argument('--classified-dir', default='classified_objects',
                       help='Classified objects directory (positive/negative samples)')
    parser.add_argument('--output-dir', default='reid_dataset', 
                       help='Output dataset directory (not used if using classified data)')
    parser.add_argument('--use-classified', action='store_true', default=False,
                   help='Use pre-classified data instead of extracting from videos')

    
    parser.add_argument('--model-path', default='reid_model_best.pth', 
                       help='Output model path')
    parser.add_argument('--onnx-path', default='reid_model.onnx', 
                       help='Output ONNX path')
    parser.add_argument('--embedding-dim', type=int, default=256, 
                       help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--loss-type', default='crossentropy', 
                       choices=['triplet', 'crossentropy'],
                       help='Loss function type')
    parser.add_argument('--input-height', type=int, default=256,
                       help='Input image height')
    parser.add_argument('--input-width', type=int, default=128,
                       help='Input image width')
    parser.add_argument('--pooling', default='gem', 
                       choices=['gem', 'max', 'avg'],
                       help='Pooling type')
    parser.add_argument('--skip-dataset', action='store_true', 
                       help='Skip dataset creation')
    parser.add_argument('--skip-train', action='store_true', 
                       help='Skip training')
    parser.add_argument('--export-only', action='store_true', 
                       help='Only export to ONNX')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔥 Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Step 1: Create dataset
    if not args.skip_dataset and not args.export_only:
        if args.use_classified:
            print("\n📂 Using pre-classified data from classified_objects/")
            data_dict = create_reid_dataset_from_classified(
                train_dir=args.train_dir,
                classified_dir=args.classified_dir,
                output_dir=args.output_dir
            )
        else:
            print("\n📹 Extracting from videos (slower, may have duplicates)")
            data_dict = create_reid_dataset(args.train_dir, args.output_dir)
        
        # Save data_dict for later use
        import pickle
        with open('reid_data_dict.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        # Load existing data_dict
        import pickle
        try:
            with open('reid_data_dict.pkl', 'rb') as f:
                data_dict = pickle.load(f)
        except FileNotFoundError:
            print("❌ Error: reid_data_dict.pkl not found!")
            print("   Please run without --skip-dataset first")
            sys.exit(1)
    
    # Step 2: Train model
    if not args.skip_train and not args.export_only:
        train_reid_model(
            data_dict,
            output_model_path=args.model_path,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            loss_type=args.loss_type,
            input_size=(args.input_height, args.input_width),
            pooling=args.pooling
        )
    
    # Step 3: Export to ONNX
    # ✅ FIX: Calculate num_classes correctly
    try:
        import pickle
        with open('reid_data_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        num_classes = len(data_dict) if args.loss_type == 'crossentropy' else None
    except:
        num_classes = None
        print("⚠️  Warning: Could not load data_dict, num_classes set to None")
    
    export_to_onnx(
        model_path=args.model_path,
        onnx_path=args.onnx_path,
        embedding_dim=args.embedding_dim,
        input_size=(1, 3, args.input_height, args.input_width),
        pooling=args.pooling,
        num_classes=num_classes
    )
    
    print("\n✅ All tasks completed!")
    print(f"\nFiles created:")
    print(f"  - reid_data_dict.pkl (dataset mapping)")
    print(f"  - {args.model_path} (PyTorch model)")
    print(f"  - {args.onnx_path} (ONNX model)")
    print(f"\nNext steps:")
    print(f"  1. Validate model: python validate_reid.py")
    print(f"  2. Convert to TensorRT: trtexec --onnx={args.onnx_path} --saveEngine=reid_model.engine --fp16")
    print(f"  3. Run inference: python run_full_pipeline.py")


if __name__ == "__main__":
    main()