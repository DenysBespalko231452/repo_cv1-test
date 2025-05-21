import os
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from typing import Literal, Optional, TypedDict, List, Dict, Tuple
import shutil
from rich.progress import track
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import typer
import cv2

def setup_dirs() -> None:
    log_dir = Path("./logs")
    data_raw_dir = Path("./data/raw")
    sacred_test_set_dir = Path("./data/sacred_test_set")
    log_dir.mkdir(parents=True, exist_ok=True)
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    sacred_test_set_dir.mkdir(parents=True, exist_ok=True)

class LoggerManager:
    """
    LoggerManager class for creating and managing log files with dynamic naming and directory structure.
    Handles automatic enumeration of log files/directories to avoid overwriting and enforces naming conventions.
    
    Args:
        name (str): Base name for the logger and log files.
        context (Literal["app", "dev", "roalt"]): Contextual identifier determining the base log directory.
        enum_log (bool, optional): Whether to enumerate log files (e.g., name_0.log, name_1.log). Defaults to True.
        log_subdir (Optional[str], optional): Subdirectory name within the context base path. Defaults to None.
        enum_subdir (bool, optional): Whether to enumerate subdirectories (e.g., log_subdir_0, log_subdir_1). 
            Only effective if log_subdir is provided. Defaults to False.
    
    Raises:
        ValueError: If the provided `name` already exists as a subdirectory in the base path.
    
    Attributes:
        name (str): Logger name.
        base_path (str): Full path to the base directory for logs (e.g., "./logs/app/subdir_0").
        log_file (str): Full path to the log file (e.g., "./logs/dev/name_1.log").
        logger (logging.Logger): Configured logger instance with file and console handlers.
    
    Notes:
        - Uses `RichHandler` for console output (requires `rich` library).
        - Disables logger propagation to prevent duplicate logging.
        - Automatically creates directories if they don't exist.
    """
    def __init__(
            self,
            name:str,
            context:Literal["app", "dev", "roalt"],
            enum_log:bool=True,
            log_subdir:Optional[str]=None,
            enum_subdir:bool=False
        ):
        self.name = name
        self.base_path = f"./logs/{context}"
        os.makedirs(self.base_path, exist_ok=True)

        if (os.path.isdir(os.path.join(self.base_path, name)) or os.path.isdir(os.path.join(self.base_path, name, "_0"))):
            raise ValueError("Provided name is already being used as subdirectory name, please use another name.")

        if log_subdir is None and enum_log is False:
            self.log_file = f"./logs/{context}/{name}.log"
        
        if log_subdir is None and enum_log is True:
            existing_num = self._get_existing_num(name)
            self.log_file = f"{self.base_path}/{name}_{existing_num + 1}.log"


        if enum_subdir:
            existing_num_dir = self._get_existing_num(log_subdir)
            self.base_path = f"./logs/{context}/{log_subdir}_{existing_num_dir + 1}"
            os.makedirs(self.base_path, exist_ok=True)
        elif not(enum_subdir):
            self.base_path = f"./logs/{context}/{log_subdir}"
            os.makedirs(self.base_path, exist_ok=True)


        if enum_log:
            existing_num_log = self._get_existing_num(name)
            self.log_file = f"{self.base_path}/{name}_{existing_num_log + 1}.log"
        else:
            self.log_file = f"{self.base_path}/{name}.log"
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self._setup_handler()

    def _get_existing_num(self, name):
        existing_num = -1
        for entry in os.listdir(self.base_path):
            if entry.startswith(f"{name}_"):
                try:
                    log_num = int(entry.split('_')[1])
                    existing_num = log_num
                except (ValueError, IndexError):
                    continue
        return existing_num

    def _setup_handler(self):
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)

        console_handler = RichHandler(console=Console())
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

# Utils for main.py

def find_files(src: Path, exts: List[str]) -> List[Path]:
    """Return a sorted list of files with the given extensions in source directory."""
    files: List[Path] = []
    for ext in exts:
        files.extend(src.rglob(f"*.{ext}"))
    return sorted(files)

def map_image_to_masks(src: Path, image_paths: List[Path], console: Console) -> Dict[Path, List[Path]]:
    """Map each image file to its corresponding mask files."""
    mapping: Dict[Path, List[Path]] = {}
    for img in image_paths:
        masks = list(src.rglob(f"{img.stem}_*_mask.tif"))
        if masks:
            mapping[img] = masks
        else:
            console.print(f"[yellow]⚠ No masks for {img.name}; skipping.[/]")
    return mapping

# Utils for main.py -- Organising data

def prepare_output_dirs(out_root: Path, dataset_name: str) -> Tuple[Path, Path]:
    """Create and return img and mask directories."""
    img_dest = out_root / dataset_name / "img"
    mask_root = out_root / dataset_name / "mask"
    img_dest.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    return img_dest, mask_root

def _extract_class(png: Path, mask: Path) -> str:
    """Extract class identifier from mask filename."""
    stem = mask.stem
    suffix = "_mask"
    prefix = f"{png.stem}_"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix):-len(suffix)]
    return "root"

def organise_data(
    png_to_masks: Dict[Path, List[Path]],
    img_dest_dir: Path,
    mask_root_dir: Path,
    dataset_name: str,
    copy: bool = True
) -> Dict[str, int]:
    
    """Copy or move images and masks into organised structure."""
    stats = {"images": 0, "masks": 0}
    pad = max(5, len(str(len(png_to_masks))))

    for idx, (png, masks) in track(
        enumerate(png_to_masks.items(), 1),
        total=len(png_to_masks),
        description="Organising"
    ):
        num = f"{idx:0{pad}d}"
        new_img_name = f"{dataset_name}_{num}.png"
        action = shutil.copy2 if copy else shutil.move
        action(png, img_dest_dir / new_img_name)
        stats["images"] += 1

        for mask in masks:
            cls = _extract_class(png, mask)
            dest_dir = mask_root_dir / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            new_mask_name = f"{dataset_name}_{num}_{cls}_mask.tif"
            action(mask, dest_dir / new_mask_name)
            stats["masks"] += 1

    return stats

## Utils for main.py -- EDA

### Utils for main.py -- EDA -- Root mask size distribution

def _get_cluster_shares(sizes: List[float], labels: List[int]) -> Dict[str, float]:
    """
    Calculate the percentage of each cluster and label it with its min-max range.

    Args:
        sizes (List[float]): List of root mask size percentages.
        labels (List[int]): Corresponding cluster labels for each size.

    Returns:
        Dict[str, float]: Cluster ranges (min-max) as keys and their percentage share.
    """
    sizes = np.array(sizes)
    result = {}
    total = len(labels)

    for cluster in np.unique(labels):
        cluster_sizes = sizes[labels == cluster]
        min_val, max_val = cluster_sizes.min(), cluster_sizes.max()
        share = (len(cluster_sizes) / total) * 100
        result[f"{min_val:.2f}-{max_val:.2f}%"] = round(share, 1)

    return dict(sorted(result.items(), key=lambda item: float(item[0].split('-')[0])))



def _calculate_mask_coverage(mask: np.ndarray) -> float:
    """
    Calculate the percentage of root pixels (value == 1) in a mask.

    Args:
        mask (np.ndarray): Grayscale mask image.

    Returns:
        float: Percentage of pixels equal to 1.
    """
    return np.sum(mask == 1) / mask.size * 100


def _cluster_root_sizes(sizes: List[float], n_clusters: int = 3) -> List[int]:
    """
    Apply KMeans clustering to root sizes.

    Args:
        sizes (List[float]): Root size percentages.
        n_clusters (int): Number of clusters.

    Returns:
        List[int]: Cluster labels for each root size.
    """
    sizes_np = np.array(sizes).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(sizes_np)


def _plot_clustered_histogram(sizes: List[float], labels: List[int], out_path: Path):
    """
    Plot and save a histogram of root sizes grouped by cluster labels.

    Args:
        sizes (List[float]): List of root size percentages.
        labels (List[int]): Cluster labels.
        out_path (Path): Path to save the output plot.
    """
    colors = ['blue', 'orange', 'red']
    plt.figure(figsize=(10, 5))

    for i in range(3):
        cluster_sizes = [size for size, label in zip(sizes, labels) if label == i]
        plt.hist(cluster_sizes, bins=3, color=colors[i], alpha=0.6, label=f"Cluster {i + 1}")

    plt.title("Root Mask Size Clusters")
    plt.xlabel("Size (% of pixels)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def get_distribution_root_size(
    masks: List[cv2.typing.MatLike],
    out_dir: Path,
    console: Console
) -> Dict[str, float]:
    """
    Analyze root mask sizes, cluster them, and return cluster distribution.

    Args:
        masks (List[cv2.typing.MatLike]): List of grayscale root mask images.
        out_dir (Path): Directory to save the output plot.
        console (Console): Rich console for status reporting.

    Returns:
        Dict[str, float]: Cluster distribution as percentages.
    """
    sizes = []

    for mask in track(masks, description="Calculating root mask size"):
        if mask is not None:
            sizes.append(_calculate_mask_coverage(mask))

    labels = _cluster_root_sizes(sizes)
    _plot_clustered_histogram(sizes, labels, out_dir / "root_mask_size_clusters.pdf")
    console.print(f"[green]✓ Clustered root mask size distribution saved at {out_dir / 'root_mask_size_clusters.pdf'}[/]")

    return _get_cluster_shares(sizes, labels)

### Utils for main.py -- EDA -- Pixel distribution

def _compute_histograms(images: List[cv2.typing.MatLike]) -> List[np.ndarray]:
    return [np.bincount(img.flatten(), minlength=256) for img in track(images, description="Computing histograms") if img is not None]

def _normalize_histogram(hist: np.ndarray) -> List[float]:
    return (hist / np.sum(hist)).tolist()

def _plot_pixel_distribution(hist: List[float], out_path: Path):
    plt.figure(figsize=(10, 5))
    plt.bar(range(256), hist, color='blue', alpha=0.7)
    plt.title("Pixel Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(out_path)
    plt.close()

def get_dataset_distribution(images: List[cv2.typing.MatLike], out_dir: Path, console: Console) -> List[float]:
    """Get the pixel distribution of the dataset."""
    
    histograms = _compute_histograms(images)
    avg_hist = np.mean(histograms, axis=0)
    distribution = _normalize_histogram(avg_hist)

    _plot_pixel_distribution(distribution, out_dir / "image_pixel_distribution.pdf")

    return distribution

def detect_array_value_range(images: List[cv2.typing.MatLike]) -> str:
    min_val, max_val = float('inf'), float('-inf')
    range_flags = set()

    for img in images:
        if img is None:
            continue
        img_min = img.min()
        img_max = img.max()

        min_val = min(min_val, img_min)
        max_val = max(max_val, img_max)

        if img_max <= 1:
            range_flags.add("0-1")
        elif img_max > 1 and img_max <= 255:
            range_flags.add("0-255")
        else:
            range_flags.add("unexpected")

    if "unexpected" in range_flags:
        return "unexpected"
    elif len(range_flags) == 1:
        return next(iter(range_flags))
    else:
        return "mixed"

def get_unique_shapes(images: List[cv2.typing.MatLike]) -> List[Tuple[int, int]]:
    return list({img.shape for img in images if img is not None})
