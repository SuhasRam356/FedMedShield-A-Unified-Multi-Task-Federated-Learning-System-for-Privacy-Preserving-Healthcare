"""
═══════════════════════════════════════════════════════════════
FedMedShield — Medical Imaging Data Loader (Module 2)
Handles brain tumor MRI, glaucoma retinal, and COVID chest X-ray images.
Generates synthetic image data when real datasets are unavailable.
Supports non-IID splitting across hospital nodes.
═══════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Dict, List, Tuple, Optional, Literal
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

# ═══ Image Transformation Pipelines ═══

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training augmentation pipeline for medical images.
    Includes rotation, flips, color jitter, and normalization to
    improve model generalization on limited medical data.

    Args:
        image_size: Target image dimension (square).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet pretrained means
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Validation/test transform pipeline (no augmentation, only resize + normalize).

    Args:
        image_size: Target image dimension (square).

    Returns:
        torchvision Compose transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ═══ Dataset Classes ═══

class MedicalImageDataset(Dataset):
    """
    Generic medical image dataset that loads images from disk or uses
    pre-generated synthetic tensors. Supports tumor, glaucoma, and
    COVID X-ray classification tasks.
    """

    def __init__(
        self,
        images: List[np.ndarray],
        labels: np.ndarray,
        task: Literal["tumor", "glaucoma", "covid_xray"],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Args:
            images: List of image arrays (H, W, C) in uint8 format.
            labels: Classification labels array.
            task: Which imaging task this dataset serves.
            transform: torchvision transform pipeline to apply.
        """
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.task = task
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self.images[idx]

        # Convert numpy array to PIL Image for torchvision transforms
        if isinstance(image, np.ndarray):
            # Ensure 3-channel RGB
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        else:
            pil_image = image

        if self.transform:
            tensor_image = self.transform(pil_image)
        else:
            tensor_image = transforms.ToTensor()(pil_image)

        return {
            "image": tensor_image,
            "label": self.labels[idx],
            "task": self.task,
        }


class ImageFolderLoader:
    """
    Loads real image datasets from a structured directory:
        data_dir/
          class_0/
            image1.jpg
            image2.png
          class_1/
            ...
    Falls back to synthetic generation if directory is empty or missing.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, data_dir: str, class_names: List[str]) -> None:
        self.data_dir = data_dir
        self.class_names = class_names

    def load(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Attempt to load images from disk.

        Returns:
            Tuple of (image list, label array) or empty lists if not found.
        """
        images: List[np.ndarray] = []
        labels: List[int] = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue

                fpath = os.path.join(class_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    images.append(np.array(img))
                    labels.append(class_idx)
                except Exception as e:
                    logger.warning(f"Failed to load {fpath}: {e}")

        if images:
            logger.info(
                f"Loaded {len(images)} real images from {self.data_dir} "
                f"across {len(self.class_names)} classes"
            )

        return images, np.array(labels, dtype=np.int64) if labels else np.array([])

    def has_data(self) -> bool:
        """Check if real data exists on disk."""
        images, _ = self.load()
        return len(images) > 0


# ═══ Synthetic Image Generators ═══

def generate_synthetic_tumor_images(
    num_samples: int = 1200,
    image_size: int = 128,
    random_seed: int = 42,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate synthetic brain MRI-like images with tumor indicators.
    Creates grayscale brain-shaped images with bright regions for tumors.

    Classes: 0=none, 1=benign, 2=malignant

    Args:
        num_samples: Number of images to generate per class.
        image_size: Output image dimension.
        random_seed: For reproducibility.

    Returns:
        Tuple of (image list, label array).
    """
    rng = np.random.RandomState(random_seed)
    images: List[np.ndarray] = []
    labels: List[int] = []
    samples_per_class = num_samples // 3

    for class_idx in range(3):
        for _ in range(samples_per_class):
            # Create dark background (MRI-like)
            img = rng.normal(loc=30, scale=10, size=(image_size, image_size)).clip(0, 255)

            # Add brain-shaped elliptical region (brighter center)
            y_grid, x_grid = np.ogrid[:image_size, :image_size]
            cx, cy = image_size // 2, image_size // 2
            brain_mask = ((x_grid - cx) ** 2 / (cx * 0.7) ** 2 +
                          (y_grid - cy) ** 2 / (cy * 0.8) ** 2) < 1
            img[brain_mask] += rng.normal(loc=80, scale=15, size=brain_mask.sum()).clip(0, 180)

            if class_idx == 1:  # benign — small, smooth, well-defined bright spot
                tumor_cx = cx + rng.randint(-20, 20)
                tumor_cy = cy + rng.randint(-20, 20)
                radius = rng.randint(8, 15)
                tumor_mask = ((x_grid - tumor_cx) ** 2 + (y_grid - tumor_cy) ** 2) < radius ** 2
                img[tumor_mask] += rng.normal(loc=100, scale=10, size=tumor_mask.sum())

            elif class_idx == 2:  # malignant — larger, irregular, brighter
                tumor_cx = cx + rng.randint(-25, 25)
                tumor_cy = cy + rng.randint(-25, 25)
                radius = rng.randint(15, 30)
                # Irregular shape using noise
                noise = rng.normal(0, radius * 0.3, size=(image_size, image_size))
                tumor_mask = ((x_grid - tumor_cx) ** 2 + (y_grid - tumor_cy) ** 2) < (radius + noise) ** 2
                tumor_mask &= brain_mask
                img[tumor_mask] += rng.normal(loc=140, scale=20, size=tumor_mask.sum())
                # Add ring enhancement (characteristic of malignant tumors)
                ring_mask = (((x_grid - tumor_cx) ** 2 + (y_grid - tumor_cy) ** 2) < (radius + 5) ** 2) & \
                            (((x_grid - tumor_cx) ** 2 + (y_grid - tumor_cy) ** 2) >= (radius - 3) ** 2)
                ring_mask &= brain_mask
                img[ring_mask] += 40

            img = img.clip(0, 255).astype(np.uint8)
            # Convert to 3-channel
            img_rgb = np.stack([img, img, img], axis=-1)
            images.append(img_rgb)
            labels.append(class_idx)

    logger.info(
        f"Generated {len(images)} synthetic tumor images "
        f"({samples_per_class} per class: none/benign/malignant)"
    )
    return images, np.array(labels, dtype=np.int64)


def generate_synthetic_glaucoma_images(
    num_samples: int = 800,
    image_size: int = 128,
    random_seed: int = 43,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate synthetic retinal fundus-like images for glaucoma detection.
    Simulates optic disc/cup ratio variation (key glaucoma indicator).

    Classes: 0=normal, 1=glaucoma

    Args:
        num_samples: Total images.
        image_size: Output dimension.
        random_seed: For reproducibility.

    Returns:
        Tuple of (image list, label array).
    """
    rng = np.random.RandomState(random_seed)
    images: List[np.ndarray] = []
    labels: List[int] = []
    samples_per_class = num_samples // 2

    for class_idx in range(2):
        for _ in range(samples_per_class):
            # Dark red-orange fundus background
            img = np.zeros((image_size, image_size, 3), dtype=np.float64)
            img[:, :, 0] = rng.normal(loc=140, scale=20, size=(image_size, image_size))  # Red
            img[:, :, 1] = rng.normal(loc=60, scale=15, size=(image_size, image_size))   # Green
            img[:, :, 2] = rng.normal(loc=30, scale=10, size=(image_size, image_size))   # Blue

            cx, cy = image_size // 2, image_size // 2
            y_grid, x_grid = np.ogrid[:image_size, :image_size]

            # Optic disc (bright yellowish circle)
            disc_radius = rng.randint(18, 28)
            disc_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < disc_radius ** 2
            img[disc_mask, 0] += 80
            img[disc_mask, 1] += 70
            img[disc_mask, 2] += 30

            # Optic cup (lighter center within disc)
            if class_idx == 0:  # Normal: small cup-to-disc ratio (0.2-0.4)
                cup_ratio = rng.uniform(0.2, 0.4)
            else:  # Glaucoma: large cup-to-disc ratio (0.6-0.9)
                cup_ratio = rng.uniform(0.6, 0.9)

            cup_radius = int(disc_radius * cup_ratio)
            cup_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < cup_radius ** 2
            img[cup_mask, 0] += 40
            img[cup_mask, 1] += 50
            img[cup_mask, 2] += 40

            # Add blood vessel-like lines (radiating from disc)
            num_vessels = rng.randint(4, 8)
            for _ in range(num_vessels):
                angle = rng.uniform(0, 2 * np.pi)
                length = rng.randint(30, image_size // 2)
                thickness = rng.randint(1, 3)
                for t in range(length):
                    vx = int(cx + t * np.cos(angle) + rng.normal(0, 0.5))
                    vy = int(cy + t * np.sin(angle) + rng.normal(0, 0.5))
                    if 0 <= vx < image_size and 0 <= vy < image_size:
                        for dx in range(-thickness, thickness + 1):
                            for dy in range(-thickness, thickness + 1):
                                nx, ny = vx + dx, vy + dy
                                if 0 <= nx < image_size and 0 <= ny < image_size:
                                    img[ny, nx, 0] = max(img[ny, nx, 0] - 40, 0)
                                    img[ny, nx, 1] = max(img[ny, nx, 1] - 20, 0)

            img = img.clip(0, 255).astype(np.uint8)
            images.append(img)
            labels.append(class_idx)

    logger.info(
        f"Generated {len(images)} synthetic glaucoma images "
        f"({samples_per_class} per class: normal/glaucoma)"
    )
    return images, np.array(labels, dtype=np.int64)


def generate_synthetic_covid_xray_images(
    num_samples: int = 1200,
    image_size: int = 128,
    random_seed: int = 44,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Generate synthetic chest X-ray-like images for COVID classification.
    Simulates lung fields with varying opacity patterns.

    Classes: 0=normal, 1=covid, 2=pneumonia

    Args:
        num_samples: Total images.
        image_size: Output dimension.
        random_seed: For reproducibility.

    Returns:
        Tuple of (image list, label array).
    """
    rng = np.random.RandomState(random_seed)
    images: List[np.ndarray] = []
    labels: List[int] = []
    samples_per_class = num_samples // 3

    for class_idx in range(3):
        for _ in range(samples_per_class):
            # Dark X-ray background
            img = rng.normal(loc=20, scale=8, size=(image_size, image_size)).clip(0, 255)

            y_grid, x_grid = np.ogrid[:image_size, :image_size]
            cx, cy = image_size // 2, image_size // 2

            # Left lung field (elliptical)
            left_cx = cx - image_size // 5
            left_mask = ((x_grid - left_cx) ** 2 / (image_size * 0.18) ** 2 +
                         (y_grid - cy) ** 2 / (image_size * 0.35) ** 2) < 1
            img[left_mask] += rng.normal(loc=60, scale=10, size=left_mask.sum())

            # Right lung field (elliptical)
            right_cx = cx + image_size // 5
            right_mask = ((x_grid - right_cx) ** 2 / (image_size * 0.18) ** 2 +
                          (y_grid - cy) ** 2 / (image_size * 0.35) ** 2) < 1
            img[right_mask] += rng.normal(loc=60, scale=10, size=right_mask.sum())

            # Mediastinum (central bright stripe — spine/heart shadow)
            mediastinum_mask = np.abs(x_grid - cx) < image_size * 0.08
            img[mediastinum_mask] += 30

            if class_idx == 1:  # COVID — bilateral ground-glass opacities (peripheral)
                for lung_cx_offset in [left_cx, right_cx]:
                    num_patches = rng.randint(3, 7)
                    for _ in range(num_patches):
                        px = lung_cx_offset + rng.randint(-15, 15)
                        py = cy + rng.randint(-25, 25)
                        pr = rng.randint(5, 12)
                        patch_mask = ((x_grid - px) ** 2 + (y_grid - py) ** 2) < pr ** 2
                        # Ground-glass: subtle, hazy opacity
                        img[patch_mask] += rng.normal(loc=30, scale=8, size=patch_mask.sum())

            elif class_idx == 2:  # Pneumonia — focal consolidation (usually one-sided)
                affected_cx = rng.choice([left_cx, right_cx])
                consolidation_y = cy + rng.randint(5, 20)
                cr = rng.randint(15, 25)
                consol_mask = ((x_grid - affected_cx) ** 2 + (y_grid - consolidation_y) ** 2) < cr ** 2
                # Dense white consolidation
                img[consol_mask] += rng.normal(loc=70, scale=15, size=consol_mask.sum())

            # Add rib-like horizontal lines
            for rib_y in range(cy - 30, cy + 30, 12):
                if 0 <= rib_y < image_size:
                    img[rib_y:rib_y + 2, :] += 15

            img = img.clip(0, 255).astype(np.uint8)
            img_rgb = np.stack([img, img, img], axis=-1)
            images.append(img_rgb)
            labels.append(class_idx)

    logger.info(
        f"Generated {len(images)} synthetic COVID X-ray images "
        f"({samples_per_class} per class: normal/covid/pneumonia)"
    )
    return images, np.array(labels, dtype=np.int64)


# ═══ Non-IID Splitting ═══

def split_images_non_iid(
    images: List[np.ndarray],
    labels: np.ndarray,
    num_hospitals: int = 4,
    alpha: float = 0.5,
    random_seed: int = 42,
) -> Dict[str, Tuple[List[np.ndarray], np.ndarray]]:
    """
    Split image dataset across hospitals using Dirichlet non-IID distribution.

    Args:
        images: List of image arrays.
        labels: Label array.
        num_hospitals: Number of hospital nodes.
        alpha: Dirichlet concentration (lower = more skewed).
        random_seed: Reproducibility seed.

    Returns:
        Dict mapping hospital_id -> (images, labels) tuple.
    """
    rng = np.random.RandomState(random_seed)
    hospital_ids = [f"hospital-{chr(ord('a') + i)}" for i in range(num_hospitals)]
    hospital_indices: Dict[str, List[int]] = {h: [] for h in hospital_ids}

    num_classes = len(np.unique(labels))

    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_hospitals))
        counts = (proportions * len(class_indices)).astype(int)
        counts[-1] = len(class_indices) - counts[:-1].sum()

        start = 0
        for h_idx, count in enumerate(counts):
            hospital_indices[hospital_ids[h_idx]].extend(
                class_indices[start:start + count].tolist()
            )
            start += count

    result = {}
    for h_id in hospital_ids:
        indices = hospital_indices[h_id]
        rng.shuffle(indices)
        h_images = [images[i] for i in indices]
        h_labels = labels[indices]
        result[h_id] = (h_images, h_labels)
        logger.info(
            f"  {h_id}: {len(h_images)} images, "
            f"class dist: {np.bincount(h_labels, minlength=num_classes).tolist()}"
        )

    return result


# ═══ Main Factory ═══

class ImagingDataLoaderFactory:
    """
    Factory for creating PyTorch DataLoaders for all three imaging tasks.
    Handles real data loading or synthetic generation, augmentation,
    non-IID splitting, and train/val/test partitioning.
    """

    TASK_CONFIGS = {
        "tumor": {
            "class_names": ["none", "benign", "malignant"],
            "num_classes": 3,
            "generator": generate_synthetic_tumor_images,
            "default_samples": 1200,
        },
        "glaucoma": {
            "class_names": ["normal", "glaucoma"],
            "num_classes": 2,
            "generator": generate_synthetic_glaucoma_images,
            "default_samples": 800,
        },
        "covid_xray": {
            "class_names": ["normal", "covid", "pneumonia"],
            "num_classes": 3,
            "generator": generate_synthetic_covid_xray_images,
            "default_samples": 1200,
        },
    }

    def __init__(
        self,
        data_dir: str = "data/imaging_data",
        image_size: int = 224,
        batch_size: int = 16,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_hospitals: int = 4,
        non_iid_alpha: float = 0.5,
        random_seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_hospitals = num_hospitals
        self.non_iid_alpha = non_iid_alpha
        self.random_seed = random_seed
        self._cache: Dict[str, Dict[str, Tuple[List[np.ndarray], np.ndarray]]] = {}

    def _load_task_data(
        self, task: str
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load real data or generate synthetic for a given task."""
        config = self.TASK_CONFIGS[task]
        task_dir = os.path.join(self.data_dir, task)

        # Try real data first
        folder_loader = ImageFolderLoader(task_dir, config["class_names"])
        if folder_loader.has_data():
            return folder_loader.load()

        # Fall back to synthetic
        logger.info(f"No real {task} data found at {task_dir}. Generating synthetic...")
        return config["generator"](
            num_samples=config["default_samples"],
            image_size=self.image_size,
            random_seed=self.random_seed,
        )

    def _get_hospital_split(
        self, task: str
    ) -> Dict[str, Tuple[List[np.ndarray], np.ndarray]]:
        """Get non-IID split for a task (cached)."""
        if task not in self._cache:
            images, labels = self._load_task_data(task)
            self._cache[task] = split_images_non_iid(
                images, labels,
                num_hospitals=self.num_hospitals,
                alpha=self.non_iid_alpha,
                random_seed=self.random_seed,
            )
        return self._cache[task]

    def get_hospital_dataloaders(
        self,
        hospital_id: str,
        task: str,
    ) -> Dict[str, DataLoader]:
        """
        Get train/val/test DataLoaders for a specific hospital and imaging task.

        Args:
            hospital_id: e.g. 'hospital-b'
            task: 'tumor', 'glaucoma', or 'covid_xray'

        Returns:
            Dict with 'train', 'val', 'test' DataLoader objects.
        """
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Available: {list(self.TASK_CONFIGS.keys())}")

        hospital_split = self._get_hospital_split(task)
        if hospital_id not in hospital_split:
            raise ValueError(f"Unknown hospital_id '{hospital_id}'.")

        h_images, h_labels = hospital_split[hospital_id]

        # Train/val/test split
        indices = np.arange(len(h_images))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=h_labels,
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=h_labels[temp_idx],
        )

        train_dataset = MedicalImageDataset(
            [h_images[i] for i in train_idx], h_labels[train_idx],
            task=task, transform=get_train_transforms(self.image_size),
        )
        val_dataset = MedicalImageDataset(
            [h_images[i] for i in val_idx], h_labels[val_idx],
            task=task, transform=get_eval_transforms(self.image_size),
        )
        test_dataset = MedicalImageDataset(
            [h_images[i] for i in test_idx], h_labels[test_idx],
            task=task, transform=get_eval_transforms(self.image_size),
        )

        logger.info(
            f"  {hospital_id}/{task} — train: {len(train_dataset)}, "
            f"val: {len(val_dataset)}, test: {len(test_dataset)}"
        )

        return {
            "train": DataLoader(
                train_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=0, drop_last=False,
            ),
            "val": DataLoader(
                val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=0,
            ),
            "test": DataLoader(
                test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=0,
            ),
        }

    def get_centralized_dataloaders(
        self, task: str
    ) -> Dict[str, DataLoader]:
        """Get combined DataLoaders for centralized baseline comparison."""
        images, labels = self._load_task_data(task)
        indices = np.arange(len(images))

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=self.val_ratio + self.test_ratio,
            random_state=self.random_seed,
            stratify=labels,
        )
        relative_test = self.test_ratio / (self.val_ratio + self.test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test,
            random_state=self.random_seed,
            stratify=labels[temp_idx],
        )

        return {
            "train": DataLoader(
                MedicalImageDataset(
                    [images[i] for i in train_idx], labels[train_idx],
                    task=task, transform=get_train_transforms(self.image_size),
                ),
                batch_size=self.batch_size, shuffle=True,
            ),
            "val": DataLoader(
                MedicalImageDataset(
                    [images[i] for i in val_idx], labels[val_idx],
                    task=task, transform=get_eval_transforms(self.image_size),
                ),
                batch_size=self.batch_size, shuffle=False,
            ),
            "test": DataLoader(
                MedicalImageDataset(
                    [images[i] for i in test_idx], labels[test_idx],
                    task=task, transform=get_eval_transforms(self.image_size),
                ),
                batch_size=self.batch_size, shuffle=False,
            ),
        }
