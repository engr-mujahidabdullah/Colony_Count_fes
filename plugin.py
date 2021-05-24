from skimage import data, io, filters
import numpy as np
import cv2 as cv2
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import watershed, expand_labels
from skimage.color import label2rgb

def plot_comparison(original, filtered, filter_name):
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

if __name__ == '__main__' :

    image = cv2.imread('C_2.jpg')
        # Make segmentation using edge-detection and watershed.
    edges = sobel(image)

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < 30.0] = background
    markers[image > 150.0] = foreground

    ws = watershed(edges, markers)
    seg1 = label(ws == foreground)

    expanded = expand_labels(seg1, distance=10)

    # Show the segmentations.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
                            sharex=True, sharey=True)

    color1 = label2rgb(seg1, image=image, bg_label=0)
    axes[0].imshow(color1)
    axes[0].set_title('Sobel+Watershed')

    color2 = label2rgb(expanded, image=image, bg_label=0)
    axes[1].imshow(color2)
    axes[1].set_title('Expanded labels')

    for a in axes:
        a.axis('off')
    fig.tight_layout()
    plt.show()



