
import pandas as pd
import numpy as np
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


def compute_iou_table(true_boxes, proposed_boxes):
    """Returns a table containing ROI values betweeen true boxes (columns)
        and probosed boxes (rows)

    Args:
        true_boxes (pd.DataFrame), proposed_boxes (pd.DataFrame): 
            dataframes containing columns specifying x, y, width, and height of boxes
    """
    
    iou_table = pd.DataFrame(index=proposed_boxes.index, columns=true_boxes.index)
    
    for i, true_box_spec in enumerate(true_boxes.iloc):
        
        prop_box_specs = proposed_boxes[['x', 'y', 'width', 'height']].values
        true_box_spec = true_box_spec[['x', 'y', 'width', 'height']].values
        
        iou_ = np.apply_along_axis(
            lambda prop_box_spec : iou(prop_box_spec, true_box_spec),
            axis=1, 
            arr=prop_box_specs,
        )
    
        iou_table[i] = iou_
    
    return iou_table


def compute_training_examples(true_boxes, proposed_boxes):
    """
    Return the image together with the processed samples 
    after filtering
    """
    
    iou_table = compute_iou_table(true_boxes, proposed_boxes)
    
    # remove boxes with small IoU
    proposed_boxes = proposed_boxes.loc[iou_table.max(axis='columns') >= 0.1]
    iou_table = iou_table.loc[iou_table.max(axis='columns') >= 0.1]
    
    # label >0.5 iou as positive examples
    proposed_boxes['labels']=pd.Series()
    proposed_boxes.loc[iou_table.max(axis='columns') >= 0.5, 'labels'] = 1
    proposed_boxes.loc[iou_table.max(axis='columns') < 0.5, 'labels'] = 0
    
    matching_true_box_idx=pd.Series(index=proposed_boxes.index)
    
    matching_true_box_idx.loc[proposed_boxes['labels'] == 1] = \
        iou_table.idxmax(axis='columns').loc[proposed_boxes['labels'] == 1]
    
    matching_true_boxes = []
    for idx in proposed_boxes.index:
        if not pd.isna(matching_true_box_idx)[idx]: 
            true_box = true_boxes.loc[matching_true_box_idx[idx], :]
        else: true_box = [np.NAN]*4
        matching_true_boxes.append(true_box)
        
    matching_true_boxes = pd.DataFrame(
        matching_true_boxes, index=proposed_boxes.index, columns=true_boxes.columns
    )
    
    return matching_true_boxes, proposed_boxes
        
        
