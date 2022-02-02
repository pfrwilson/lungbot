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
    i_1 = max( x_1, x_2 )
    i_2 = min( x_1 + w_1, x_2 + w_2 )
    w_i = max( i_2 - i_1, 0 )
    
    # length of height intersection
    i_1 = max( y_1, y_2 )
    i_2 = min( y_1 + h_1, y_2 + h_2 )
    h_i = max( i_2 - i_1, 0 )
    
    # intersection area:
    i = w_i * h_i
    
    # union area (sum minus intersection)
    u = ( w_1 * h_1 ) + ( w_2 * h_2 ) - i
    
    return i / u



