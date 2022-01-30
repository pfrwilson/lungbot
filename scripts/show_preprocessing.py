from matplotlib import pyplot as plt
from skimage import color, exposure, filters
import numpy as np
import os

# To run, requires $ export PYTHONPATH=/path/to/project/root

from src.data.datasets.cxr_dataset import CXRDataset

root = '/Users/paulwilson/data/node_21/cxr_images/proccessed_data'

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

