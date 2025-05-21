import os
import cv2
import uvicorn
import tempfile
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, Response, UploadFile
from core.logging_config import setup_logging
from middlewares.logging_middleware import LoggingMiddleware
from dev.evaluation.inference import predict_masks

setup_logging()
app = FastAPI()

# Add middleware
app.add_middleware(LoggingMiddleware)

@app.get("/")
def read_root():
    return {"message": "Yes that's definetly a root"}

@app.post("/actions/inference")
def inference(img: UploadFile = File(...), patch_size: int=256):
    try:
        contents = img.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "dish.png")
            success = cv2.imwrite(save_path, img)
            if not success:
                raise RuntimeError(f"Failed to write image to {save_path}")
    
            predict_masks(
                image_dir = Path(tmpdir),
                model_path = Path("./app/best_unet_model.pth"),
                output_dir = Path(tmpdir),
                patch_size = patch_size,
                model_size = "large",
                use_bn = True
            )

            output_path = os.path.join(tmpdir, "dish_mask.tif")
            with open(output_path, "rb") as f:
                data = f.read()

            return Response(content=data, media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)