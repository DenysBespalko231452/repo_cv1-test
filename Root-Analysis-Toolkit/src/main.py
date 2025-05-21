import pandas as pd

import shutil
from pathlib import Path
from typing import Dict, List
from utils import LoggerManager
from dev.data.pre_processing import crop_2_dish

import os
import typer
from rich.progress import track
from rich.console import Console

# Rich console instance
console = Console()
app = typer.Typer(rich_markup_mode=True)


# Before start of run this should be ready
#    run_mgr = LoggerManager(
#        name="run",
#        context="dev",
#        enum_log=False,
#        log_subdir="run",
#        enum_subdir=True
#    )
#    run_dir = Path(run_mgr.base_path).name
#
#    loggers = {"run" : run_mgr}
#    modules = ["data", "models", "training", "evaluation"]
#    for module in modules:
#        loggers[module] = LoggerManager(
#            name=module,
#            context="dev",
#            enum_log=False,
#            log_subdir=run_dir,
#            enum_subdir=False
#            ).logger.info(f"started module {module}")


@app.command("organise", help="Organise raw PNG + mask files from a mixed folder into separate img/mask directories of a shared dataset folder.")
def organise(
    src: Path = typer.Argument(default="./data/raw", exists=True, file_okay=False, help="Folder with unprocessed mixed mask/png files"),
    out_root: Path = typer.Option(Path("./data/organised"), help="Destination root for organised data"),
    dataset_name = typer.Option(None, "-n", "--name", help="Descriptive name for new dataset folder"),
    copy: bool = typer.Option(True, "-cp/-mv" "--copy/--move", help="Copy instead of move"),
):
    """Organise *src* into the canonical layout and rename files."""
    if not dataset_name:
        dataset_name = typer.prompt("Dataset name")
    console.rule(f"[bold cyan]· Organising: {dataset_name}")

    img_dest = out_root / dataset_name / "img"
    mask_root = out_root / dataset_name / "mask"

    png_files = sorted(src.glob("*.png"))
    if not png_files:
        console.print("[red]No .png files found.[/]")
        raise typer.Exit(code=1)

    png_to_masks: Dict[Path, List[Path]] = {}
    for png in png_files:
        masks = list(src.glob(f"{png.stem}_*_mask.tif"))
        if masks:
            png_to_masks[png] = masks
        else:
            console.print(f"[yellow]⚠ No masks for {png.name}; skipping.[/]")

    if not png_to_masks:
        console.print("[red]No images had masks.[/]")
        raise typer.Exit(code=1)
    
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(mask_root, exist_ok=True)

    pad = max(5, len(str(len(png_to_masks))))
    stats = {"images": 0, "masks": 0}

    for idx, (png, masks) in track(
        enumerate(png_to_masks.items(), 1),
        total=len(png_to_masks),
        description="Organising",
    ):
        num = f"{idx:0{pad}d}"
        new_img = f"{dataset_name}_{num}.png"
        (shutil.copy2 if copy else shutil.move)(png, img_dest / new_img)
        stats["images"] += 1

        for mask in masks:
            stem = mask.stem
            suffix="_mask"
            if not stem.startswith(png.stem + "_") or not stem.endswith(suffix):
                cls = "root"
            else:
                cls = stem[len(png.stem) + 1 : -len(suffix)]
            mdir = mask_root / cls
            os.makedirs(mdir, exist_ok=True)
            new_mask = f"{dataset_name}_{num}_{cls}_mask.tif"
            (shutil.copy2 if copy else shutil.move)(mask, mdir / new_mask)
            stats["masks"] += 1

    console.rule("[bold green]Organising Dataset complete")
    console.print(
        f"[green]✓ {stats['images']} images & {stats['masks']} masks organised into {out_root / dataset_name}.[/]"
    )


@app.command("prep", help="Preprocess images + mask files from an organised folder structured like output by 'organise' command")
def prep(
    src: Path = typer.Argument("./data/organised", exists=True, file_okay=False, help="Organised dataset mask/png files"),
    crop2dish: bool = typer.Option(True, "-c2d", "--crop2dish", help="Crop images and masks to petri-dish (square)"),
    out_root: Path = typer.Option(Path("./data/pre_processed"), help="Destination root for pre-processed data"),
    copy: bool = typer.Option(True, "-cp/-mv" "--copy/--move", help="Copy instead of move"),
):
    """Pre-process *src* files according to input parameters."""
    console.rule(f"[bold cyan]· Pre-processing dataset: {src.stem}")

    img_root = Path(f"{src}/img")
    mask_root = Path(f"{src}/mask")

    all_imgs = list(img_root.glob("*.png"))

    records = []
    bboxes = []
    for img_path in track(all_imgs, description="[bold cyan]Scanning images[/]"):
        stem = img_path.stem

        for mask_path in mask_root.rglob(f"{stem}_*_mask.tif"):
            cls = mask_path.parent.name
            records.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "class": cls
            })
        
        if crop2dish:
            x,y,w,h = crop_2_dish(img_path)
            bboxes.append({"img_path": img_path, "x":x, "y":y, "w":w, "h":h})

    df = pd.DataFrame(records)

    if crop2dish:
        bb_df = pd.DataFrame(bboxes)
        df = df.merge(bb_df, on="img_path", how="left")
        img, col= df.shape

    print(df.head())

    console.rule("[bold green]Dataset preparation complete")
    console.print(
        f"[green]✓ {img} images & corresponding masks ready for training.[/]"
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
    app()