import torch
from torch.utils.data import DataLoader
from dev.training.model_architectures import UNetModel
from dev.data.pre_processing import SegmentationDataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from rich.console import Console

console = Console()

def dice_score(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate Dice score for binary segmentation"""
    ### Implement IOU metric
    preds = (preds > 0.5).float()
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    return (2. * intersection) / (union + eps)

def train_unet(config):
    batch_size = config.batch_size
    lr = config.lr
    num_epochs = config.num_epochs
    model_size = config.model_size
    use_bn = config.use_bn
    scheduler = config.scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Using device:[/] {device}")
    # Load preprocessed datasets
    train_ds = torch.load("data/pre_processed/train_obj", weights_only=False)
    eval_ds = torch.load("data/pre_processed/eval_obj", weights_only=False)
    
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # Model
    model = UNetModel(
        dataset=train_ds,
        model_size=model_size,
        use_batch_norm=use_bn,
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler_obj = None
    if scheduler == "step":
        scheduler_obj = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler == "cosine":
        scheduler_obj = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")
    for i, (images, masks) in enumerate(train_loader):
        unique_vals = torch.unique(masks)
        console.print(f"Batch {i}: Unique mask values: {unique_vals}")
        if i == 2:  # just check first 3 batches
            break
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        # progress bar for training
        for images, masks in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
            

            # send to device and convert to float
            images = images.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_score(preds, masks).item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in tqdm(eval_loader, desc=f"Val Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device, non_blocking=True).float()
                masks = masks.to(device, non_blocking=True).float()

                preds = model(images)
                loss = criterion(preds, masks)

                val_loss += loss.item()
                val_dice += dice_score(preds, masks).item()

        avg_train_dice = train_dice / len(train_loader)
        avg_val_dice = val_dice / len(eval_loader)

        # Save best model
        avg_val_loss = val_loss / len(eval_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model_1.pth")
            console.print(f"[bold yellow]Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}[/]")

        if scheduler_obj:
            scheduler_obj.step()

        console.print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"[bold green]Train Loss:[/] {train_loss/len(train_loader):.4f} | "
            f"Dice: {avg_train_dice:.4f} || "
            f"[bold blue]Val Loss:[/] {val_loss/len(eval_loader):.4f} | "
            f"Dice: {avg_val_dice:.4f}"
        )
