import os
import shutil
from pathlib import Path
from dev.training.train import train_unet
from dev.evaluation.inference import predict_masks
from typing import Dict, List, Tuple, TypedDict, Optional, Any
from dev.training.train import train_unet
import argparse
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
import typer
from rich.table import Table
from rich.progress import track
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from utils import LoggerManager, setup_dirs, find_files, prepare_output_dirs, organise_data, detect_array_value_range, get_dataset_distribution, map_image_to_masks, get_unique_shapes, get_distribution_root_size
from dev.data.pre_processing import SegmentationDataset

# Rich console instance
console = Console()
app = typer.Typer(rich_markup_mode="rich")
loggers = None

# Before start of run this should be ready
def start_run_logger() -> None:
    run_mgr = LoggerManager(
            name="run",
            context="dev",
            enum_log=False,
            log_subdir="run",
            enum_subdir=True
        )
    run_dir = Path(run_mgr.base_path).name

    global loggers
    loggers = {"run" : run_mgr}
    modules = ["data", "models", "training", "evaluation"]
    for module in modules:
        loggers[module] = LoggerManager(
            name=module,
            context="dev",
            enum_log=False,
            log_subdir=run_dir,
            enum_subdir=False
            )
        loggers[module].logger.info(f"started module {module}")
        return None

def create_dir_map_table(src_dir:Path) -> Tuple[List, Table]:
    """Create directory map table for dataset selection"""
    dirs = []
    for child in src_dir.iterdir():
        if child.is_dir():
            dirs.append(child)
                
    table = Table("Found", "Dataset")
    for idx, dataset in enumerate(dirs):
        table.add_row(str(idx), str(dataset.stem))
    
    return dirs, table


@app.command("organise", help="Organise raw PNG + mask files into img/mask folders.")
def organise(
    src_dir: Path = typer.Option(Path("./data/raw"), "-idir", "--dir_in", exists=True, file_okay=False, help="Path to directory containing unsorted image-mask pairs"),
    out_dir: Path = typer.Option(Path("./data/organised"), "-odir", "--dir_out", help="Path to directory to store dataset folder with sorted image-mask pairs in"),
    dataset_name: str = typer.Option(None, "-n", "--name", help="Name that will be assigned to the dataset directory that stores sorted images and masks"),
    copy: bool = typer.Option(True, "-cp/-mv", "--copy/--move", help="Copy or move the files to the output directory")
):
    dirs = []

    if src_dir == Path("./data/raw"):
        ds_dirs, table = create_dir_map_table(src_dir)
        console.print(table)

        select_ds = typer.prompt("[All] datasets or [Index] for Dataset to be organised")
        if str(select_ds).lower() == "a" or str(select_ds).lower() == "all":
            dirs = [ds_dir for ds_dir in ds_dirs]
        else:
            dirs.append(ds_dirs[int(select_ds)])
    else:
        dirs.append(src_dir)
        
    for dir in dirs:
        if not dataset_name:
            dataset_name = typer.prompt(f"Dataset name for {dir}")
        console.rule(f"[bold cyan]Organising: {dataset_name}")
        
        png_files = find_files(dir, ['png'])
        if not png_files:
            console.print("[red]No .png files found.[/]")
            raise typer.Exit(code=1)

        png_to_masks = map_image_to_masks(dir, png_files, console)
        if not png_to_masks:
            console.print("[red]No images had masks.[/]")
            raise typer.Exit(code=1)
            
        img_dest, mask_root = prepare_output_dirs(out_dir, dataset_name)
        stats = organise_data(png_to_masks, img_dest, mask_root, dataset_name, copy)

        console.rule("[bold green]Organising Dataset complete")
        console.print(
            f"[green]✓ {stats['images']} images & {stats['masks']} masks organised into {out_dir / dataset_name}.[/]"
        )

        dataset_name = None

@app.command("generate-json", help="Generate JSON file with description of dataset")
def generate_json(
    src_dir: Path = typer.Option(Path("./data/organised"), exists=True, file_okay=False, help="Organised dataset mask/png files"),
    link: str = typer.Option("Unknown origin", "-lnk", "--link", help="Link instead of copy"),
    bg_color: str = typer.Option("white", "-bgc", "--bg_color", help="Background color for images"),
    use_mask: bool = typer.Option(False, "-um", "--use_mask", help="Use masks for EDA if available"),
):
    global loggers
    dirs = []
    
    if src_dir == Path("./data/organised"):
        ds_dirs, table = create_dir_map_table(src_dir)
        console.print(table)

        select_ds = typer.prompt("[All] datasets or [Index] for Dataset to be pre-processed")
        if str(select_ds).lower() == "a" or str(select_ds).lower() == "all":
            dirs = [ds_dir for ds_dir in ds_dirs]
        else:
            dirs.append(ds_dirs[int(select_ds)])
    else:
        dirs.append(src_dir)
    
    for dir in dirs:
        console.rule("")
        console.rule(f"[bold cyan]· Generating JSON for {dir}")
        console.rule("")
        loggers["data"].logger.info(f"started generating JSON for dataset {dir}")

        image_paths = find_files(dir, ["png"])
        mask_paths = [p for p in dir.rglob("*root_mask.tif") if "occluded_root_mask" not in p.name]
        out_dir = dir / "info"

        # Check of images exist
        if not image_paths:
            console.print(f"[red]No images found in the source {dir} directory.[/]")
            raise typer.Exit(code=1)
    
        # If masks are present, check if they exist
        if use_mask:
            mask_paths = find_files(dir, ["tif"])
            if not mask_paths:
                console.print(f"[red]No masks found in the source {dir} directory.[/]")
                raise typer.Exit(code=1)

        # Delete the output directory if it exists
        if out_dir.exists():
            shutil.rmtree(out_dir)
            console.print(f"[yellow]Removed old info for the dataset in {out_dir}[/]")

        # Recreate the output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE) for img_path in track(image_paths, description="Loading images")]
        masks = [cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) for mask_path in track(mask_paths, description="Loading masks")] if use_mask else []

        # Get the distribution for the image dataset
        image_pixel_distribution = get_dataset_distribution(images, out_dir, console)
    
        # Create a dictionary to store dataset information
        dataset_info: Dict[str, Any] = {
            "dataset_name": dir.stem,
            "img_dir": os.path.relpath(str(dir / "img")),
            "mask_dir": os.path.relpath(str(dir / "mask")) if use_mask else None,
            "bg_color": bg_color,
            "pixel_2_mm": None,
            "classes": [name for name in os.listdir(dir / "mask")
                if os.path.isdir(os.path.join(dir / "mask", name))] if use_mask else None,
            "num_images": len(images),
            "img_pixel_values_range": detect_array_value_range(images),
            "num_masks": len(masks) if use_mask else None,
            "mask_pixel_values_range": detect_array_value_range(masks) if use_mask else None,
            "shapes": get_unique_shapes(images),
            "link": link,
            "image_pixel_distribution": image_pixel_distribution,
            "root_mask_size_distribution": get_distribution_root_size(masks, out_dir, console) if use_mask else None,
        }

        # Print dataset_info as a rich table
        info_table = Table(title="Dataset Info")
        info_table.add_column("Key", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="magenta")

        for key, value in dataset_info.items():
            if value is None:
                continue

            # Format long or complex values for display
            if isinstance(value, (list, dict)):
                display_value = str(value)
                if len(display_value) > 80:
                    display_value = "..."
            else:
                display_value = str(value)    
            info_table.add_row(str(key), display_value)

        console.print(info_table)

        # Save the dataset information to a JSON file
        json_file_path = out_dir / "info.json" 
        with open(json_file_path, "w") as json_file:
            json.dump(dataset_info, json_file, indent=4)

        console.print(f"[green]✓ JSON file created at {json_file_path}[/]")
    
    console.rule("")
    console.rule("[bold green]JSON generation complete")
    console.rule("")


@app.command("prep", help="Preprocess images + mask files from an organised folder structured like output by 'organise' command")
def prep(
    src_dir: Path = typer.Option(Path("./data/organised"), "-idir", "--dir_in", exists=True, file_okay=False, help="Path to directory containing Organised dataset with image-mask pairs"),
    cls: str = typer.Option("root", "-cls", "--class", help="which class to train the model on"),
    fraction: float = typer.Option(1.0, "-f", "--fraction", help="fraction of data to include in dataset"),
    train_val_split: float = typer.Option(0.8, "-s", "--split", help="split between evaluation dataset and training dataset"),
    patch_size: int = typer.Option(256,  help="patch size that will be used for pre-processing"),
    prep_dir: Path = typer.Option(Path("./data/pre_processed"), "-odir", "--dir_out", help="Path to directory to store dataset folder with pre-processed image-mask pairs in"),
    test_set_dir: Path = typer.Option(Path("./data/sacred_test_set"), help="Path to the **SACRED UNCHANGING** test dataset"),
):
    global loggers
    train_ds = None
    eval_ds = None
    test_ds = None
    dirs = []

    if src_dir == Path("./data/organised"):
        ds_dirs, table = create_dir_map_table(src_dir)
        console.print(table)

        select_ds = typer.prompt("[All] datasets or [Index] for Dataset to be pre-processed")
        if str(select_ds).lower() == "a" or str(select_ds).lower() == "all":
            dirs = [ds_dir for ds_dir in ds_dirs]
        else:
            dirs.append(ds_dirs[int(select_ds)])
    else:
        dirs.append(src_dir)


    for dir in dirs:
        console.rule("")
        console.rule(f"[bold cyan]Dataset {dir.stem} Preparation Started")
        console.rule("")
        loggers["data"].logger.info(f"started pre-processing for dataset {dir.stem}")

        img_root_dir = Path(f"{dir}/img")
        mask_root_dir = Path(f"{dir}/mask/{cls}")

        all_imgs = list(img_root_dir.glob("*.png"))
        all_msks = list(mask_root_dir.glob("*.tif"))

        if not fraction:
            fraction = float(typer.prompt("fraction [0.0 - 1.0] of dataset to be pre-processed"))

        f_imgs = all_imgs[:int(fraction*len(all_imgs))]
        f_msks = all_msks[:int(fraction*len(all_msks))]

        if not train_val_split:
            train_val_split = float(typer.prompt("train - validation split [0.0 - 1.0] of dataset to be pre-processed"))

        train_imgs = f_imgs[:int(train_val_split*len(f_imgs))]
        val_imgs = f_imgs[int(train_val_split*len(f_imgs)):]

        train_msks = f_msks[:int(train_val_split*len(f_msks))]
        val_msks = f_msks[int(train_val_split*len(f_msks)):]

        if not train_ds and not eval_ds:
            train_ds = SegmentationDataset(train_imgs, train_msks, patch_size, prep_dir)
            eval_ds = SegmentationDataset(val_imgs, val_msks, patch_size, prep_dir)
        else:
            train_ds = ConcatDataset([train_ds, SegmentationDataset(train_imgs, train_msks, patch_size, prep_dir)])
            eval_ds = ConcatDataset([eval_ds, SegmentationDataset(val_imgs, val_msks, patch_size, prep_dir)])

        fraction = None
        train_val_split = None

        console.rule("")
        console.rule(f"[bold green]Dataset {dir.stem} Preparation Complete")
        console.rule("")
        loggers["data"].logger.info(f"finished pre-processing for dataset {dir.stem}")

    img_test_dir = Path(f"{test_set_dir}/img")
    mask_test_dir = Path(f"{test_set_dir}/mask/{cls}")

    test_imgs = list(img_test_dir.glob("*.png"))
    test_msks = list(mask_test_dir.glob("*.tif"))

    # On Error make sure that you have the sacred test dataset in src/data/sacred_test_set -> img/ mask/
    test_ds = SegmentationDataset(test_imgs, test_msks, patch_size, prep_dir)
    console.print(
        f"[green]✓ datasets ready for training.[/]"
    )

    loggers["run"].logger.info(f"example log: data pre-processing finished")
    loggers["data"].logger.info(f"example log: model_type={cls}, patchsize={patch_size}")

    torch.save(train_ds, Path(f"{prep_dir}/train_obj"))
    torch.save(eval_ds, Path(f"{prep_dir}/eval_obj"))
    torch.save(test_ds, Path(f"{prep_dir}/test_obj"))

    load_ds = torch.load(Path(f"{prep_dir}/test_obj"), weights_only=False)

    loader = DataLoader(
    load_ds,
    batch_size=1,
    num_workers=1,
    )

    batch = next(iter(loader))
    imgs, msk = batch

    print("Batch image tensor shape:", imgs.shape)
    print("Corresponding mask shape:", msk.shape)



@app.command("train", help="Train the U-Net segmentation model.")
def train(
    batch_size: int = typer.Option(8),
    lr: float = typer.Option(1e-4),
    num_epochs: int = typer.Option(25),
    model_size: str = typer.Option("medium", help="Choose: small, medium, large"),
    use_bn: bool = typer.Option(False, "--use-bn", help="Use Batch Normalization"),
    scheduler: str = typer.Option("none", help="Choose: none, step, cosine")
):
    class Args:
        pass

    args = Args()
    args.batch_size = batch_size
    args.lr = lr
    args.num_epochs = num_epochs
    args.model_size = model_size
    args.use_bn = use_bn
    args.scheduler = scheduler

    train_unet(args)

@app.command("predict", help="Predict masks for images in a folder.")
def predict(
    image_dir: Path = typer.Option(..., "--image-dir", exists=True, file_okay=False),
    model_name: Path = typer.Option(..., "--model-name"),
    patch_size: int = typer.Option(..., help="Must match training input size"),
    model_size: str = typer.Option(..., help="small/medium/large"),
    use_bn: bool = typer.Option(False, "--use-bn"),
    output_dir: Optional[Path] = typer.Option(None),
):
    output_dir = output_dir or image_dir / "masks"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    predict_masks(
        image_dir=image_dir,
        model_path=model_name,
        output_dir=output_dir,
        patch_size=patch_size,
        model_size=model_size,
        use_bn=use_bn
    )

#def load_and_crop(rec):
    # rec has img_path, mask_path, and x,y,w,h
    #img = cv2.imread(str(rec.img_path))
    # if you have multiple masks per image you might store them as a list in df,
    # or just handle one here
    #mask = cv2.imread(str(rec.mask_path), cv2.IMREAD_UNCHANGED)

    # crop to the dish
    #img_cropped  = img[rec.y : rec.y+rec.h, rec.x : rec.x+rec.w]
    #mask_cropped = mask[rec.y : rec.y+rec.h, rec.x : rec.x+rec.w]

    #return img_cropped, mask_cropped



#@app.command()
#def main():

    # 1- Prep dataset (specify splits for current datasets in folder)
    # 2- Choose preprocessing functions
    # 3- Choose model architecture
    # 4a- Choose training
    # 4b- Specify hyperparameters
    # 5- Choose model evaluation
    
    # Later on:
    # post-processing
    # package evaluation
#    return None


if __name__ == "__main__":
    setup_dirs()
    start_run_logger()
    app()