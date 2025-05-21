import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dev.training.model_architectures import UNetModel
from dev.data.pre_processing import normalize_img
from rich.console import Console

console = Console()

def crop_to_dish_with_coords(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop to petri-dish and return coordinates (x, y, w, h) in original image."""
    kernel = np.ones((5, 5), dtype="uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
    im = cv2.dilate(thresh, kernel, iterations=2)
    im = cv2.erode(im, kernel, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(im)
    
    if num_labels > 1:
        # Largest component excluding background (index 0)
        max_idx = np.argmax(stats[1:, 4]) + 1
        x, y, w, h, _ = stats[max_idx]
    else:
        h, w = img.shape[:2]
        x, y = 0, 0
        
    cropped = img[y:y+h, x:x+w]
    return cropped, (x, y, w, h)

def predict_masks(
    image_dir: Path,
    model_path: Path,
    output_dir: Path,
    patch_size: int,
    model_size: str,
    use_bn: bool
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with explicit parameters
    state_dict = torch.load(model_path, map_location=device)
    model = UNetModel(
        in_channels=3,
        height=patch_size,
        width=patch_size,
        model_size=model_size,
        use_batch_norm=use_bn,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Process images
    for img_path in image_dir.glob("*.*"):
        # Load original image and get dimensions
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = original_img.shape[:2]
        
        # Crop to dish and get coordinates
        cropped_img, (x, y, w, h) = crop_to_dish_with_coords(original_img)
        
        # Pad cropped image to patch_size multiple
        padded_img = pad_to_square(cropped_img, patch_size)
        
        # Split into patches and predict
        patches = split_into_patches(padded_img, patch_size)
        pred_patches = []
        for patch in patches:
            normalized = normalize_img(patch)
            tensor_patch = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor_patch)
            pred_mask = pred.float().squeeze().cpu().numpy()
            # pred_mask = np.rot90(pred_mask, k=1)  # Rotate counter-clockwise to undo training-time rotation 
            pred_patches.append(pred_mask)
        
        # Stitch patches and remove padding
        stitched = stitch_patches(pred_patches, padded_img.shape[:2], patch_size)
        cropped_mask = remove_padding(stitched, cropped_img.shape[:2], patch_size)
        
        # Create full-size mask and insert prediction
        full_mask = np.zeros((orig_h, orig_w), dtype=np.float32)  # Now INSIDE the loop
        full_mask[y:y+h, x:x+w] = cropped_mask
        
        # Save as float32 TIFF
        mask_path = output_dir / f"{img_path.stem}_mask.tif"
        cv2.imwrite(str(mask_path), full_mask.astype(np.float32))
        

def pad_to_square(img: np.ndarray, patch_size: int) -> np.ndarray:
    """Pad image to make dimensions divisible by patch_size."""
    h, w = img.shape[:2]
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    return cv2.copyMakeBorder(
        img, 
        0, pad_h, 0, pad_w, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )

def split_into_patches(image: np.ndarray, patch_size: int) -> List[np.ndarray]:
    h, w, _ = image.shape
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def stitch_patches(patches: List[np.ndarray], target_shape: tuple, patch_size: int) -> np.ndarray:
    stitched = np.zeros(target_shape[:2], dtype=np.float32)
    idx = 0
    for y in range(0, target_shape[0], patch_size):
        for x in range(0, target_shape[1], patch_size):
            if idx >= len(patches):
                break
            stitched[y:y+patch_size, x:x+patch_size] = patches[idx]
            idx += 1
    return stitched

def remove_padding(padded_mask: np.ndarray, orig_shape: tuple, patch_size: int) -> np.ndarray:
    h, w = orig_shape
    return padded_mask[:h, :w]