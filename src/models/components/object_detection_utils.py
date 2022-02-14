import torch
from torchvision import ops


def match_proposed_boxes_to_true(
        true_boxes: torch.Tensor, 
        proposed_boxes: torch.Tensor, 
        in_format: str = 'xywh', 
    ):
    """Matches proposed bounding boxes to a tensor of ground truth bounding boxes
       and returns a list of iou scores between the proposed boxes and their matching true box.

    Args:
        true_boxes (torch.Tensor): A tensor of boxes of shape (N, 4)
        
        proposed_boxes (torch.Tensor): A tensor of boxes of shape (M, 4)
        
        in_format (str, optional): string specifying the string format - 
        see torchvision ops documentation.Defaults to 'xywh'.

    Returns:
        dict: a dict containing the proposed boxes, their matches, and their iou scores
    """
    assert len(true_boxes.shape) == 2
    assert len(proposed_boxes.shape) == 2
    
    num_true_boxes, _ = true_boxes.shape
    num_proposed_boxes, _ = proposed_boxes.shape

    ious = ops.box_iou(
        ops.box_convert(proposed_boxes, in_fmt=in_format, out_fmt='xyxy'),
        ops.box_convert(true_boxes, in_fmt=in_format, out_fmt='xyxy')
    )
    
    max_ious = torch.max(ious, dim=-1, )
    matching_true_boxes = true_boxes[max_ious.indices]
    
    return {
        'matching_true_boxes': matching_true_boxes, 
        'proposed_boxes': proposed_boxes,
        'iou_scores': max_ious.values
    }
    

def label_by_iou_threshold(
        iou_scores,
        positivity_threshold=0.7,
        negativity_threshold=0.3,
        min_num_positives=0
    ):
    """Creates a list of labels by thresholding the given iou scores. 
    IoU above the positivity threshold are assigned a label of 1. Labels 
    below the negativity threshold are a signed a label of 0. 
    Labels in between are given a label of -1 (inconclusive)
    """
    
    
    labels = (torch.ones_like(iou_scores) * -1).long()
    indices = torch.tensor(range(len(labels))).long()

    negative_indices = indices[iou_scores < negativity_threshold]

    labels[negative_indices] = 0

    positive_indices = indices[iou_scores >= positivity_threshold]
    if len(positive_indices) < min_num_positives:
        positive_indices = torch.sort(iou_scores, dim=-1, descending=True).indices[:min_num_positives]

    labels[positive_indices] = 1
    
    return labels
    
    
def boxreg_transform(regression_scores, anchor_boxes, in_fmt='xywh'):
    """Apply the box regression transform along the last axis of the input.

    Args:
        regression_scores ([type]): a tensor of shape (N, 4), where the last dimension contains t_x, t_y, t_w, t_h
        anchor_boxes ([type]): a tensor of shape (N, 4) specifying the anchor boxes upon which the transofrm is being performed.
        in_fmt (str, optional): The format of the boxes. Defaults to 'xywh'.
    """
    
    # the transform requires 'cxcywh' format:
    anchor_boxes = ops.box_convert(anchor_boxes, in_fmt=in_fmt, out_fmt='cxcywh')
    
    x_a = anchor_boxes[:, 0]
    y_a = anchor_boxes[:, 1]
    w_a = anchor_boxes[:, 2]
    h_a = anchor_boxes[:, 3]
    
    t_x = regression_scores[:, 0]
    t_y = regression_scores[:, 1]
    t_w = regression_scores[:, 2]
    t_h = regression_scores[:, 3]
    
    x = ( t_x * w_a ) + x_a 
    y = ( t_y * h_a ) + y_a
    w = w_a * torch.exp(t_w)
    h = h_a * torch.exp(t_h)
    
    proposed_boxes = torch.stack( [x, y, w, h], dim=-1 )
    proposed_boxes = ops.box_convert(anchor_boxes, in_fmt='cxcywh', out_fmt=in_fmt)
    return proposed_boxes
    

def inverse_boxreg_transform(proposed_boxes, anchor_boxes, in_fmt='xywh'):
    """Obtain the transform parameters that would transform the given anchor boxes to the given 
    output ( proposed boxes )

    Args:
        proposed_boxes (Tensor): A tensor of shape (N, 4)
        anchor_boxes (Tensor): A tensor fo shape (N, 4)
        in_fmt (str, optional): The format of the boxes. Defaults to 'xywh'.
    """
    
    anchor_boxes = ops.box_convert(anchor_boxes, in_fmt=in_fmt, out_fmt='cxcywh')
    proposed_boxes = ops.box_convert(proposed_boxes, in_fmt=in_fmt, out_fmt='cxcywh')
    
    x_a = anchor_boxes[:, 0]
    y_a = anchor_boxes[:, 1]
    w_a = anchor_boxes[:, 2]
    h_a = anchor_boxes[:, 3]
    
    x = proposed_boxes[:, 0]
    y = proposed_boxes[:, 1]
    w = proposed_boxes[:, 2]
    h = proposed_boxes[:, 3]
    
    t_x = ( x - x_a ) / w_a
    t_y = ( y - y_a ) / h_a 
    t_w = torch.log( w / w_a )
    t_h = torch.log( h / h_a)
    
    transform_parameters = torch.stack( [t_x, t_y, t_w, t_h], dim=-1 )
    return transform_parameters
    
    
def select_training_examples(
    proposed_boxes: torch.Tensor, 
    anchor_boxes: torch.Tensor, 
    regression_scores: torch.Tensor, 
    objectness_scores: torch.Tensor, 
    true_boxes: torch.Tensor, 
    num_training_examples,
    min_num_positives, 
    positivity_threshold: float = 0.7, 
    negativity_threshold: float = 0.3, 
):

    box_matching_output = match_proposed_boxes_to_true(
        true_boxes, 
        proposed_boxes, 
    )
    
    labels = label_by_iou_threshold(
        box_matching_output['iou_scores'], 
        positivity_threshold=positivity_threshold, 
        negativity_threshold=negativity_threshold, 
        min_num_positives=min_num_positives
    )

    matching_true_boxes = box_matching_output['matching_true_boxes']
    
    target_regression_scores = inverse_boxreg_transform(
        matching_true_boxes, anchor_boxes
    )
    
    indices_for_loss = torch.tensor(range(len(labels))).long()

    indices_for_loss_positive = indices_for_loss[labels > 0]
    indices_for_loss_negative = indices_for_loss[labels == 0]

    indices_for_loss_positive = indices_for_loss_positive[torch.randperm(len(indices_for_loss_positive))]
    if len(indices_for_loss_positive) >= num_training_examples//2:
        indices_for_loss_positive = indices_for_loss_positive[:num_training_examples//2]

    indices_for_loss_negative = indices_for_loss_negative[torch.randperm(len(indices_for_loss_negative))]
    indices_for_loss_negative = indices_for_loss_negative[:num_training_examples - len(indices_for_loss_positive)]

    indices_for_loss = torch.concat([indices_for_loss_negative, indices_for_loss_positive])
    
    reg_scores_for_loss = regression_scores[indices_for_loss]
    target_reg_scores_for_loss = target_regression_scores[indices_for_loss]

    objectness_scores_for_loss = objectness_scores[indices_for_loss]
    labels_for_loss = labels[indices_for_loss]

    iou_scores = box_matching_output['iou_scores'][indices_for_loss]

    return {
        'regression_scores': reg_scores_for_loss, 
        'target_regression_scores': target_reg_scores_for_loss, 
        'objectness_scores': objectness_scores_for_loss, 
        'labels': labels_for_loss,
        'iou_scores': iou_scores,
    }
    
    
def compute_metrics(
        proposed_boxes, 
        object_probs, 
        true_boxes,
        prob_threshold_for_detection = 0.01,
        iou_threshold_for_detection = 0.5
    ):
    
    proposed_boxes = proposed_boxes[object_probs > prob_threshold_for_detection]
    
    if len(proposed_boxes) == 0:
        return 0, 0
    
    iou_scores = match_proposed_boxes_to_true(
        proposed_boxes, true_boxes
    )['iou_scores']
    
    labels = label_by_iou_threshold(
        iou_scores, 
        positivity_threshold=iou_threshold_for_detection,
        negativity_threshold=iou_threshold_for_detection
    )
    
    assert torch.all(labels != -1)
    
    tp = len(labels[labels == 1])
    fp = len(labels[labels == 0])
    
    return tp, fp, tp+fp


