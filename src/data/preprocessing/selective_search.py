'''
Selective search for R-CNN region proposal.
'''

import selective_search
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import SimpleITK as sitk
import selective_search



boxes = selective_search.selective_search(im2, mode='single', random_sort=False)

boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=60)

# draw rectangles on the original image
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(im2)
for x1, y1, x2, y2 in boxes_filter:
    bbox = mpatches.Rectangle(
        (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(bbox)

plt.axis('off')
plt.show()