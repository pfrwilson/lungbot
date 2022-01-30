import numpy as np
import pandas as pd

from typing import Tuple

def iou(rect_1: Tuple[int, int, int, int], 
        rect_2: Tuple[int, int, int, int]):
    """
    Computes intersection over union between rectangles

    Args:
        rect_1 (Tuple[int, int, int, int]),   
        rect_2 (Tuple[int, int, int, int]):
            tuples of the form (x, y, w, h) - 
                x = left
                y = top
                w = width
                h = height
    """
    
    x_1, y_1, w_1, h_1 = rect_1
    x_2, y_2, w_2, h_2 = rect_2 
    
    # length of width intersection 
    x_m, x_M, w_m = (x_1, x_2, w_1) if x_1 <= x_2 else (x_2, x_1, w_2)
    w_i = np.max( [x_m + w_m - x_M, 0] )
    
    # length of height intersection
    y_m, y_M, w_m = (y_1, y_2, h_1) if y_1 <= y_2 else (y_2, y_1, h_2)
    h_i = np.max( [y_m + w_m - y_M, 0] )
    
    # intersection area:
    i = w_i * h_i
    
    # union area (sum minus intersection)
    u = ( w_1 * h_1 ) + ( w_2 * h_2 ) - i
    
    return i / u