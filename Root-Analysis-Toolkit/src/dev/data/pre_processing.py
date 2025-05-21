import os
import os
import cv2
import torch
import torch
import numpy as np
from typing import Tuple
from pathlib import Path

def crop_to_dish(img: cv2.typing.MatLike, msk: cv2.typing.MatLike) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """Finds petri-dish by applying thresholding (75, 255), a closing operation and checking connected components.
    Assumes that the connected component with the second biggest area is the petri-dish (Biggest is background).
    After grayscale conversion, dish pixels should be brighter than 75 and the background is darker.
    The returned crops are always rectangular.

    Might need update when it encounters inverted images with brighter background

    Args:
        img: The image with petri-dish to crop around box around dish.
        msk: The matching mask corresponding to the image with petri-dish

    Returns:
        cropped image, cropped mask
        cropped image, cropped mask
    """
    kernel = np.ones((5, 5), dtype="uint8")
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, im = cv2.threshold(image, 75, 255, cv2.THRESH_BINARY)
    im = cv2.dilate(im, kernel, iterations=2)
    im = cv2.erode(im, kernel, iterations=2)
    _, _, stats, _ = cv2.connectedComponentsWithStats(im)
    
    ########### FIX 1: Skip background component ###########
    if len(stats) > 1:
        # Get components sorted by area (descending) excluding background
        sorted_stats = sorted(stats[1:], key=lambda x: x[4], reverse=True)
        x, y, w, h, _ = sorted_stats[0]  # Largest non-background component
    else:
        # Fallback to full image if no components found
        h, w = img.shape[:2]
        x, y = 0, 0
    
    img = img[y:y+h, x:x+w]
    msk = msk[y:y+h, x:x+w]
    return img, msk

def pad_to_square(img: cv2.typing.MatLike, msk: cv2.typing.MatLike, patch_size: int) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """Pad an image + corresponding mask with 0's dependent on the intended patchsize, the image/mask stay centered.

    Args:
        img: The image to be padded.
        msk: The corresponding mask object to be padded.

    Returns:
        img, msk: image, mask that have been padded equally
    """
    ########### FIX 2: Handle grayscale masks properly ###########
    h, w, c = img.shape
    
    rem_h = patch_size - (h % patch_size)
    rem_w = patch_size - (w % patch_size)

    if rem_h != patch_size:
        pad_h = int(rem_h / 2)
        top = pad_h
        bottom = pad_h if rem_h % 2 == 0 else pad_h + 1
    else:
        top = bottom = 0

    if rem_w != patch_size:
        pad_w = int(rem_w / 2)
        left = pad_w
        right = pad_w if rem_w % 2 == 0 else pad_w + 1
    else:
        left = right = 0
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    msk = cv2.copyMakeBorder(msk, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img, msk

def normalize_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """converts uint8 img to normalized 0 - 1 float32
    Args:
        img: The image to be normalized (uint8).
        
    Returns:
        img: image with normalized pixel values 0-1 (float32)
    """
    return (img / 255).astype("float32")

def save_prep_to_dir(img: cv2.typing.MatLike,
                    msk: cv2.typing.MatLike,
                    patch_row_idx:int,
                    patch_col_idx:int,
                    img_path: Path,
                    msk_path: Path,
                    prep_dir: Path | None) -> Tuple[Path, Path]:
    """Saves preprocessed files with absolute paths"""
    ########### FIX 3: Path handling and mask format ###########
    prep_dir = Path(prep_dir).resolve() if prep_dir else None
    
    if prep_dir is None:
        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(msk_path), msk)
        return img_path, msk_path

    # Create directory structure
    sample_id = msk_path.parents[3].name
    mask_category = msk_path.parent.name
    
    img_dir = prep_dir / sample_id / "img"
    msk_dir = prep_dir / sample_id / "mask" / mask_category
    
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    
    # Maintain original filenames with _preprocessed suffix
    prep_img_path = img_dir / f"{msk_path.stem}_{patch_row_idx}_{patch_col_idx}_preprocessed.png"
    prep_msk_path = msk_dir / f"{msk_path.stem}_{patch_row_idx}_{patch_col_idx}_preprocessed.tif"
    
    # Save with proper formats
    cv2.imwrite(str(prep_img_path), img)
    cv2.imwrite(str(prep_msk_path), msk)  # Save as uint8, 
    
    return prep_img_path, prep_msk_path

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, patch_size: int = 256, prep_dir: Path = Path("./data/pre_processed")):
        ########### FIX 4: Path validation and conversion ###########
        self.img_paths = [Path(p).resolve() for p in image_paths]
        self.msk_paths = [Path(p).resolve() for p in mask_paths]
        self.patch_size = patch_size
        self.patch_indices = []
        self.patch_img_paths = []
        self.patch_msk_paths = []
        
        # Validate inputs before processing
        self._validate_inputs()
        
        for img_path, msk_path in zip(self.img_paths, self.msk_paths):
            # Load and validate images
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load and process mask
            msk = cv2.imread(str(msk_path))

            if msk is None:
                raise ValueError(f"Failed to load mask: {msk_path}")
            
            
            # Preprocessing
            if img.dtype == np.uint8:
                img, msk = crop_to_dish(img, msk)
                img, msk = pad_to_square(img, msk, patch_size)
            
                img_h, img_w, _ = img.shape # img_h = Height, img_w = Width
                num_patches_h = img_h // self.patch_size
                num_patches_w = img_w // self.patch_size
            
                for patch_row_idx in range(num_patches_h):
                    for patch_col_idx in range(num_patches_w):
                        img_patch = self._extract_patch(img, patch_col_idx, patch_row_idx)
                        msk_patch = self._extract_patch(msk, patch_col_idx, patch_row_idx)
            
                        # Save processed patch
                        prep_img_patch_path, prep_msk_patch_path = save_prep_to_dir(
                        img_patch, msk_patch, patch_row_idx, patch_col_idx, img_path, msk_path, prep_dir
                        )

                        self.patch_img_paths.append(prep_img_patch_path)
                        self.patch_msk_paths.append(prep_msk_patch_path)


    def _validate_inputs(self):
        """Validate all input paths exist"""
        for p in self.img_paths + self.msk_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing file: {p}")
        if len(self.img_paths) != len(self.msk_paths):
            raise ValueError("Mismatched number of images and masks")

    def __len__(self) -> int:
        return len(self.patch_img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ########### FIX 7: Safe image loading ###########
        img_patch_path = self.patch_img_paths[idx]
        msk_patch_path = self.patch_msk_paths[idx]
        
        # Load and validate image
        img_patch = cv2.imread(str(img_patch_path))
        if img_patch is None:
            raise ValueError(f"Corrupted image: {img_patch_path}")
        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
        img_patch = normalize_img(img_patch)  # now float32 in [0,1]
        
        # Load and validate mask
        msk_patch = cv2.imread(str(msk_patch_path), cv2.IMREAD_GRAYSCALE)
        if msk_patch is None:
            raise ValueError(f"Corrupted mask: {msk_patch_path}")
        
        ########### FIX 8: Ensure mask is float32 in [0,1] ###########
        # The mask files already contain values 0 or 1, so we only need to cast to float32 (no further binarization needed).
        msk_patch = msk_patch.astype(np.float32)
        
        return (
            # Image tensor: shape (3, H, W), values in [0,1]
            torch.tensor(img_patch).permute(2, 0, 1),
            # Mask tensor: shape (1, H, W), values 0.0 or 1.0
            torch.tensor(msk_patch)[None, :, :]
        )




    def _extract_patch(self, img: np.ndarray, y: int, x: int) -> np.ndarray:
        """Safe patch extraction with bounds checking"""
        size = self.patch_size
        y_start = min(y * size, img.shape[0] - size)
        x_start = min(x * size, img.shape[1] - size)
        return img[y_start:y_start+size, x_start:x_start+size]