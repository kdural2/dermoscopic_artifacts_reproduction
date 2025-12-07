# src/datasets.py
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import scipy.ndimage
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]




def load_image(path: Path) -> np.ndarray:
    """Load RGB image."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask(path: Path) -> np.ndarray:
    """Load binary segmentation mask."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 0).astype(np.uint8)
    return mask


def resize_pair(img, mask, size=224):
    """Resize image (and mask if present)."""
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = None
    return img_resized, mask_resized




def low_pass_filter(img, sigma=1):
    """Low-pass Gaussian filter."""
    return scipy.ndimage.gaussian_filter(img, sigma=sigma)


def high_pass_filter_grayscale(img, sigma=1):
    """
    High-pass filter applied in grayscale then expanded to 3-channel.
    This MATCHES the paper and original repository behavior.
    """
    gray = np.dot(img[..., :3], [0.2989, 0.587, 0.114])
    low = scipy.ndimage.gaussian_filter(gray, sigma=sigma)
    high = gray - low
    high = np.stack([high] * 3, axis=-1)  # Repeat grayscale into RGB channels
    return high




def apply_mode_paper(img: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """
    Implements: whole, lesion, background, bbox, bbox70, bbox90,
                high_whole, low_whole,
                high_lesion, low_lesion,
                high_background, low_background

    EXACT behavior per Jin et al. (CHIL 2025), Sec. 3.1.
    """
    h, w, _ = img.shape

    # Prepare lesion and background components
    if mask is not None:
        lesion = img * mask[:, :, None]
        background = img * (1 - mask[:, :, None])
    else:
        lesion = img.copy()
        background = img.copy()

    if mode == "whole":
        base = img.copy()

    elif mode == "lesion":
        base = lesion

    elif mode == "background":
        base = background


    elif mode in ["bbox", "bbox70", "bbox90"]:
        if mask is None:
            return img.copy()

        ys, xs = np.where(mask == 1)
        if len(xs) == 0:
            return img * 0

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        base = img.copy()

        if mode == "bbox":
            # Tight box blackout
            cv2.rectangle(base, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

        else:
            # bbox70 / bbox90 — expand to cover 70% or 90% of entire image
            expand_ratio = 0.7 if mode == "bbox70" else 0.9
            total_area = h * w
            bbox_area = (y_max - y_min) * (x_max - x_min)
            target_area = expand_ratio * total_area

            scale_factor = np.sqrt(target_area / max(bbox_area, 1))
            new_h = int((y_max - y_min) * scale_factor)
            new_w = int((x_max - x_min) * scale_factor)

            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2

            y1 = max(0, cy - new_h // 2)
            y2 = min(h, cy + new_h // 2)
            x1 = max(0, cx - new_w // 2)
            x2 = min(w, cx + new_w // 2)

            cv2.rectangle(base, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    elif mode.startswith("high_") or mode.startswith("low_"):

        if "whole" in mode:
            base_image = img
        elif "lesion" in mode:
            base_image = lesion
        elif "background" in mode:
            base_image = background
        else:
            raise ValueError(f"Invalid frequency mode: {mode}")

        if mode.startswith("low_"):
            base = low_pass_filter(base_image, sigma=1)

        else:  # high-pass
            base = high_pass_filter_grayscale(base_image, sigma=1)

    else:
        raise ValueError(f"Unknown mode {mode}")

    return base




def get_default_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])




class ISIC2018Dataset(Dataset):
    """
    Implements the ISIC dataset exactly as in the paper.
    """
    def __init__(self, root, mode="whole", fold_indices=None, transform=None):
        root = Path(root)

        df = pd.read_csv(root / "labels.csv")  # REQUIRED: columns ["image_id", "label"]
        if fold_indices is not None:
            df = df.iloc[fold_indices].reset_index(drop=True)

        self.df = df
        self.images_dir = root / "images"
        self.masks_dir = root / "masks"
        self.mode = mode
        self.transform = transform if transform is not None else get_default_transform()
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # --- Return cached item if available ---
        if idx in self.cache:
            return self.cache[idx]

        row = self.df.iloc[idx]
        image_id = row["isic_id"]
        label = int(row["label"])

        # Image path
        img_path = self.images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / f"{image_id}.png"

        mask_path = self.masks_dir / f"{image_id}_segmentation.png"

        img = load_image(img_path)
        mask = load_mask(mask_path)

        img, mask = resize_pair(img, mask, size=224)

        # Apply mode (your paper-matching function)
        img_mode = apply_mode_paper(img, mask, self.mode)

        # Convert to tensor
        img_mode = Image.fromarray(np.clip(img_mode, 0, 255).astype(np.uint8))
        img_mode = self.transform(img_mode)

        # Pack item
        item = (img_mode, torch.tensor(label, dtype=torch.long), image_id)

        # --- Store in cache ---
        self.cache[idx] = item

        return item



class HAM10000Dataset(Dataset):
    """
    HAM10000 has no segmentation masks → ONLY whole/high/low modes.
    """
    def __init__(self, root, mode="whole", transform=None):
        root = Path(root)

        df = pd.read_csv(root / "labels.csv")  # REQUIRED: ["image_id", "label"]
        self.df = df

        self.images_dir = root / "images"
        self.mode = mode
        self.transform = transform if transform is not None else get_default_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])

        img_path = self.images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.images_dir / f"{image_id}.png"

        img = load_image(img_path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Modes HAM supports
        if "high" in self.mode:
            img = high_pass_filter_grayscale(img, sigma=1)
        elif "low" in self.mode:
            img = low_pass_filter(img, sigma=1)

        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), image_id
