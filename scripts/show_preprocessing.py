from matplotlib import pyplot as plt
from skimage import color, exposure, filters
import numpy as np
import os
import selective_search

import matplotlib.patches as mpatches
import SimpleITK as sitk
from skimage.color import rgb2hsv

# To run, requires $ export PYTHONPATH=/path/to/project/root

from src.data.datasets.cxr_dataset import CXRDataset

#root = '/Users/paulwilson/data/node_21/cxr_images/proccessed_data'
root = '/home/ayesha/Desktop/node21/cxr_images/proccessed_data'

base_img = CXRDataset(root)[0][0]
base_img = color.gray2rgb(base_img)

filtered_imgs = {}

filtered_imgs['base_img'] = base_img

filtered_imgs['sobel'] = exposure.equalize_hist(
    filters.sobel(base_img)
)

filtered_imgs['DoG_fine'] = exposure.equalize_hist(filters.difference_of_gaussians(
    base_img, 
    low_sigma=1,
    high_sigma=2,
))


filtered_imgs['DoG_coarse'] = exposure.equalize_hist(filters.difference_of_gaussians(
    base_img, 
    low_sigma=3,
    high_sigma=5,
))

filtered_imgs['hist_eq'] = exposure.equalize_hist(base_img)

filtered_imgs['adaptive_hist_eq'] = exposure.equalize_adapthist(base_img)

mask = np.zeros(base_img.shape, dtype=bool)
mask[400:750, 625:800, :] = True
mask[ 200:515, 515:660, :] = True

filtered_imgs['masked_img'] = mask * base_img

filtered_imgs['hist_eq_masked'] = exposure.equalize_hist(
    base_img, mask=mask
)

filtered_imgs['adapthist + DoG fine'] = (
    filtered_imgs['adaptive_hist_eq'] 
    + filters.gaussian(
            exposure.rescale_intensity(
            filters.difference_of_gaussians(
                base_img, 
                low_sigma=1,
                high_sigma=2,
            )
        ), sigma=0.2
    )
)

fig, axes = plt.subplots(3, 3)

for i, (k, v) in enumerate(filtered_imgs.items()):
    
    ax = axes.flatten()[i] 
    ax.set_axis_off()
    ax.set_title(k)
    ax.imshow(v)
    
plt.tight_layout()
plt.show()

plt.imshow(filtered_imgs['adapthist + DoG fine'])
plt.figure()
plt.imshow(filtered_imgs['base_img'])
plt.show()
plt.close()



# plt.imshow(filtered_imgs['adaptive_hist_eq'])
# plt.savefig('/home/ayesha/Desktop/lungbot/images/00_adaptive_hist_eq')
# plt.close()

print('shape of filtered_imgs[adaptive_hist_eq] is ', filtered_imgs['adaptive_hist_eq'].shape)
array = filtered_imgs['base_img']
array = (array - array.min())/ (array.max()-array.min())
hsv = rgb2hsv(array)

boxes = selective_search.selective_search(hsv, mode='single', random_sort=False)
boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=100)

array1 = filtered_imgs['adaptive_hist_eq']
array1 = (array1 - array1.min())/ (array1.max()-array1.min())
hsv1 = rgb2hsv(array1)

boxes1 = selective_search.selective_search(hsv1, mode='single', random_sort=False)
boxes_filter1 = selective_search.box_filter(boxes1, min_size=20, topN=100)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.imshow(array)
ax2.imshow(array)
ax3.imshow(array1)
ax4.imshow(array1)
for x1, y1, x2, y2 in boxes_filter:
    bbox = mpatches.Rectangle(
        (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='blue', linewidth=1)
    ax2.add_patch(bbox)

for x1, y1, x2, y2 in boxes_filter1:
    bbox = mpatches.Rectangle(
        (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='blue', linewidth=1)
    ax4.add_patch(bbox)

plt.suptitle("Chest X-Rays and Generated Bounding Boxes")
plt.savefig('/home/ayesha/Desktop/lungbot/images/pres-bounding-boxes2')
plt.close()