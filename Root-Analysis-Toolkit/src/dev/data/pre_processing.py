import cv2
import numpy as np
from typing import Tuple
from pathlib import Path

def bbox_2_dish(image_path:Path) -> Tuple[int]:
    """Finds petridish by applying thresholding (75, 255), a closing operation and checking connected components.
    Assumes that the biggest area is the petridish

    Args:
        image_path (Path): The path for the image to retrieve square boundingbox around dish.

    Returns:
        bbox coordinates (Tuple[int]): Coordinates of square bounding box (x,y,w,h)
    """
    kernel = np.ones((5, 5), dtype="uint8")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(image, 75, 255, cv2.THRESH_BINARY)
    im = cv2.dilate(im, kernel, iterations=2)
    im = cv2.erode(im, kernel, iterations=2)
    _, _, stats, _ = cv2.connectedComponentsWithStats(im)

    ind_lrgst_area = 0
    lrgst_area = 0

    for i, component in enumerate(stats):
        if i == 0:
            lrgst_area = component[4]
        
        else:
            if component[4] > lrgst_area:
                ind_lrgst_area = i
                lrgst_area = component[4]
            
            else:
                continue

    x, y, w, h, _ = stats[ind_lrgst_area]
    if w/h != 1:
        if w < h:
            w = h
        else:
            h = w
    
    return x, y, w, h