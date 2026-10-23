import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Skip albumentations' online version check: on Windows every DataLoader worker re-imports this module
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Fallback block to safely import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# Parallelism comes from DataLoader workers; OpenCV's own thread pool per worker only adds memory and contention
cv2.setNumThreads(0)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class TwinLiteNetDataset(Dataset):
    """
    Custom PyTorch Dataset for TwinLiteNet.
    Reads images, drivable area masks, and lane masks.
    Applies real-time spatial and pixel-level augmentations.
    """
    def __init__(self, data_dir, image_size=(640, 360), is_train=True, transform=None):
        """
        Args:
            data_dir (str): Path to either 'data/processed/train' or 'data/processed/val'
            image_size (tuple): Format (width, height) to resize inputs
            is_train (bool): Active state for training augmentations
            transform (albumentations.Compose): Custom transforms to apply
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        
        # Define paths to files
        self.img_dir = os.path.join(data_dir, "images")
        self.drivable_dir = os.path.join(data_dir, "masks_drivable")
        self.lanes_dir = os.path.join(data_dir, "masks_lanes")
        
        # Load and align all filenames
        self.filenames = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Fallback to default augmentations if none provided
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()

    def _get_default_transforms(self):
        """Defines basic resizing, weather simulation, and tensor conversion."""
        if not ALBUMENTATIONS_AVAILABLE:
            # The manual fallback skips ImageNet normalisation, which infer.py always applies,
            # so training without albumentations would silently mismatch inference.
            raise ImportError("albumentations is required for TwinLiteNetDataset (pip install albumentations)")
            
        height, width = self.image_size[1], self.image_size[0]
        
        if self.is_train:
            # Training augmentations simulating rain, lens flare, and camera noise
            return A.Compose([
                A.Resize(height, width),
                # Environmental/sensor noise simulation
                A.OneOf([
                    A.RandomRain(rain_type="heavy", brightness_coefficient=0.8, p=0.3),
                    # src_radius 200 (default 400 is tuned for larger frames and costs ~140 ms/img at 640x360)
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=200, p=0.3),
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3)
                ], p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                # Standard normalization (ImageNet distribution)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'drivable': 'mask', 'lanes': 'mask'})
        else:
            # Simple resizing and normalization for validation
            return A.Compose([
                A.Resize(height, width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], additional_targets={'drivable': 'mask', 'lanes': 'mask'})

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Construct absolute file paths
        img_path = os.path.join(self.img_dir, filename)
        drivable_path = os.path.join(self.drivable_dir, filename)
        lanes_path = os.path.join(self.lanes_dir, filename)
        
        # Load image as RGB (OpenCV reads BGR by default, swap channels)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load masks in grayscale (0 to 255)
        # Note: In case masks are saved with different extensions, look for matching filename stem
        drivable_mask = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)
        lanes_mask = cv2.imread(lanes_path, cv2.IMREAD_GRAYSCALE)
        
        # Pre-process masks to strict binary formats: 1 where active, 0 where inactive
        _, drivable_mask = cv2.threshold(drivable_mask, 127, 1, cv2.THRESH_BINARY)
        _, lanes_mask = cv2.threshold(lanes_mask, 127, 1, cv2.THRESH_BINARY)
        
        # Determine domain validity flag for selective masked loss:
        # BDD100K frames supervise lanes (1.0), IDD Lite frames skip lane supervision (0.0)
        is_bdd = 0.0 if "idd_lite" in filename.lower() else 1.0
        lane_valid_tensor = torch.tensor(is_bdd, dtype=torch.float32)
        
        # Apply transforms safely
        if self.transform:
            augmented = self.transform(image=image, drivable=drivable_mask, lanes=lanes_mask)
            image_tensor = augmented['image']
            drivable_tensor = augmented['drivable'].long()
            lanes_tensor = augmented['lanes'].long()
        else:
            # Basic fallback if Albumentations is not installed
            # Resize manually
            h, w = self.image_size[1], self.image_size[0]
            image = cv2.resize(image, (w, h))
            drivable_mask = cv2.resize(drivable_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            lanes_mask = cv2.resize(lanes_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Simple conversion to PyTorch Tensors
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            drivable_tensor = torch.from_numpy(drivable_mask).long()
            lanes_tensor = torch.from_numpy(lanes_mask).long()

        return image_tensor, drivable_tensor, lanes_tensor, lane_valid_tensor

def generate_mock_driving_data(target_dir, count=10, image_size=(1280, 720)):
    """
    Programmatically creates mock synthetic driving scenes with road polygons and lane lines.
    Ensures complete project compilation and sanity checks can run offline without raw datasets.
    """
    print(f"\nGenerating {count} mock structured driving frames inside: {target_dir}")
    
    img_dir = os.path.join(target_dir, "images")
    drivable_dir = os.path.join(target_dir, "masks_drivable")
    lanes_dir = os.path.join(target_dir, "masks_lanes")
    
    # Create target paths if they are missing
    for d in [img_dir, drivable_dir, lanes_dir]:
        os.makedirs(d, exist_ok=True)
        
    width, height = image_size
    
    for i in range(count):
        # 1. Create Mock Camera Frame (Gradient representing road surface & sky)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky: Fill top half with light blue gradient
        frame[0:height//2, :] = [210, 180, 140] # BGR sky blueish-grey
        # Ground: Fill bottom half with charcoal gray
        frame[height//2:, :] = [60, 60, 60]
        
        # 2. Build Drivable Area polygon mask (Perspective trapezoid from bottom to center)
        road_mask = np.zeros((height, width), dtype=np.uint8)
        road_pts = np.array([
            [width * 0.1, height],         # Bottom-Left
            [width * 0.45, height * 0.55], # Top-Left Horizon
            [width * 0.55, height * 0.55], # Top-Right Horizon
            [width * 0.9, height]          # Bottom-Right
        ], dtype=np.int32)
        cv2.fillPoly(road_mask, [road_pts], 255)
        
        # Draw the road polygon on the raw frame to simulate visual layout
        cv2.fillPoly(frame, [road_pts], [80, 80, 80])
        
        # 3. Build Lane lines masks (Two diagonal perspective lines)
        lane_mask = np.zeros((height, width), dtype=np.uint8)
        # Left Lane Line
        cv2.line(lane_mask, (int(width * 0.25), height), (int(width * 0.47), int(height * 0.55)), 255, thickness=12)
        # Right Lane Line
        cv2.line(lane_mask, (int(width * 0.75), height), (int(width * 0.53), int(height * 0.55)), 255, thickness=12)
        
        # Overlay lanes visually onto RGB frame in yellow
        cv2.line(frame, (int(width * 0.25), height), (int(width * 0.47), int(height * 0.55)), (0, 255, 255), thickness=10)
        cv2.line(frame, (int(width * 0.75), height), (int(width * 0.53), int(height * 0.55)), (0, 255, 255), thickness=10)
        
        # Add random shapes (Noise to simulate trees/surroundings)
        cv2.circle(frame, (int(width * 0.15), int(height * 0.3)), 40, (30, 120, 30), -1)
        cv2.circle(frame, (int(width * 0.85), int(height * 0.3)), 50, (40, 110, 20), -1)
        
        # Save output images
        filename = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(img_dir, filename), frame)
        cv2.imwrite(os.path.join(drivable_dir, filename), road_mask)
        cv2.imwrite(os.path.join(lanes_dir, filename), lane_mask)

if __name__ == "__main__":
    print("=" * 60)
    print("            TWINLITENET DATALOADER INTEGRITY TEST")
    print("=" * 60)
    
    # Generate mock training and validation datasets programmatically
    # generate_mock_driving_data("data/processed/train", count=10)
    # generate_mock_driving_data("data/processed/val", count=4)
    
    # Initialize PyTorch dataset pipelines
    print("\nInitializing PyTorch datasets...")
    train_dataset = TwinLiteNetDataset(data_dir="data/processed/train", image_size=(640, 360), is_train=True)
    val_dataset = TwinLiteNetDataset(data_dir="data/processed/val", image_size=(640, 360), is_train=False)
    
    print(f"Verified Real Training Samples   : {len(train_dataset)}")
    print(f"Verified Real Validation Samples : {len(val_dataset)}")
    
    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Extract batches until active lane markings are found
    print("\nScanning batches to verify active lane annotations...")
    found_lanes = False
    
    for images, drivable_masks, lane_masks, lane_valids in train_loader:
        if 1 in torch.unique(lane_masks).tolist():
            found_lanes = True
            print("\n" + "-" * 40)
            print("Batch Processing Verification: SUCCESS!")
            print("-" * 40)
            print(f"Image Batch Tensor Shape (Batch, Channels, Height, Width): {images.shape}")
            print(f"Drivable Area Mask Batch Shape (Batch, Height, Width)      : {drivable_masks.shape}")
            print(f"Lane Lines Mask Batch Shape (Batch, Height, Width)         : {lane_masks.shape}")
            print(f"Domain Flags Batch Shape (Batch)                           : {lane_valids.shape}")
            print(f"Images Value Range (Min / Max)                             : {images.min():.4f} / {images.max():.4f}")
            print(f"Drivable Mask Classes Present                              : {torch.unique(drivable_masks).tolist()}")
            print(f"Lane Mask Classes Present                                  : {torch.unique(lane_masks).tolist()}")
            print(f"Domain Flags Present in Batch                              : {torch.unique(lane_valids).tolist()}")
            break
            
    if not found_lanes:
        print("\n[Warning]: Sampled batches contained only unstriped roads.")
        
    print("=" * 60)