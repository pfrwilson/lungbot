import pytest

from src.data.processing.iou_filtering import iou

samples_rects = [
    ((0, 0, 2, 2), (2, 2, 2, 2), 0  ),   
    ((0, 0, 2, 2), (1, 1, 2, 2), 1/7), 
    ((0, 0, 2, 2), (0, 1, 2, 2), 2/6 ),
    ((0, 0, 2, 2), (1, 3, 2, 2), 0  ),
    ((0, 0, 1023, 62), (0, 0, 2, 2), 4 / (1023 * 62))
]


@pytest.mark.parametrize("rect1, rect2, expected", samples_rects)
def test_iou(rect1, rect2, expected): 
    
    assert iou(rect1, rect2) == expected
    assert iou(rect2, rect1) == expected


