
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import skimage
import scipy.ndimage as nd

import plotly.express as px
import plotly.graph_objects as go
from scipy import ndimage as ndi
from skimage import data, filters, measure, morphology, feature
from skimage.feature import peak_local_max
from skimage import morphology as morph
from skimage.segmentation import watershed, expand_labels
from skimage.measure import label
from skimage.color import label2rgb
from skimage.feature import peak_local_max

'''
#This method draws simple grid overthe image based on the passed step
#The pxstep controls the size of the grid
'''
def drawBasicGrid(image, pxstep, midX, midY):
    x = pxstep
    y = pxstep
    #Draw all x lines
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=(255, 0, 255), thickness=1)
        x += pxstep
    
    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=(255, 0, 255),thickness=1)
        y += pxstep

def cv_contour(_image_,_image_result_):
    #Find contours in ROI
    contours, hierarchy = cv2.findContours(_image_,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    result = cv2.drawContours(_image_result_, contours, -1, (0,255,0), 3)
    return result

#import Image and copy it as img
image = cv2.imread("C_6.jpeg")
image = imutils.resize(image, width=500, height = 300)
img = image.copy()

#draw Image Grid
drawBasicGrid(img,30,250,250)

#draw Region of intrest
roi = cv2.selectROI(windowName="roi", img=img, showCrosshair=True, fromCenter=False)
x, y, w, h = roi

#Get Circulat Region of Interst
x = round(x + (1/2)*w)
y = round(y + (1/2)*h)
w = round(w/2)
h = round(h/2)

cv2.circle(img,(x, y), w, (0, 0, 0), 1)

#get Mask of Circular Region of Interst
mask = np.zeros(img.shape[:2], dtype="uint8")
cv2.circle(mask,(x, y), w, (255, 255, 255), -1)

masked = cv2.bitwise_or(image, image, mask=mask)

#get Gray scale of ROI
gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)


#applying gaussian blur on ROI with white background
blur = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_REFLECT101)
blur = cv2.bitwise_not(blur)

filter_gau = ndi.gaussian_filter(blur, sigma=0.1)
ret,thresh = cv2.threshold(filter_gau,100,255,cv2.THRESH_BINARY_INV)

cv_cont = cv_contour(thresh, image.copy())


# Find contours at a constant value of 0.8
contours_ = measure.find_contours(thresh, 1)


# DISTANCE and WaterShed
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_FAIR,0)

local_max_location = peak_local_max(dist_transform, min_distance=1, indices=True)
local_max_boolean = peak_local_max(dist_transform, min_distance=1, indices=False)

markers, no = ndi.label(local_max_boolean)
segmented = skimage.segmentation.watershed(thresh, markers, connectivity=1, mask=thresh)

# Make segmentation using edge-detection and watershed.
edges = filters.sobel(thresh)

# Identify some background and foreground pixels from the intensity values.
# These pixels are used as seeds for watershed.
markers = np.zeros_like(thresh)
foreground, background = 1, 2
markers[thresh < 30.0] = background
markers[thresh > 150.0] = foreground

ws = watershed(edges, markers, connectivity=0, mask=thresh)
seg1 = label(ws == foreground)

expanded = expand_labels(seg1, distance=5)

# Show the segmentations.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
                         sharex=True, sharey=True)

color1 = label2rgb(seg1, image=image.copy(), bg_label=0)
axes[0].imshow(color1)
axes[0].set_title('Sobel+Watershed')

color2 = label2rgb(expanded, image=image.copy(), bg_label=0)
axes[1].imshow(color2)
axes[1].set_title('Expanded labels')

for a in axes:
    a.axis('off')
fig.tight_layout()
plt.show()


