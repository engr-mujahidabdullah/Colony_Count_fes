
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

import warnings
warnings.filterwarnings('ignore')

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
    print("Segmented Count:" + str(len(contours)))
    result = cv2.drawContours(_image_result_, contours, -1, (0,255,0), 3)
    return result

#import Image and copy it as img
image = cv2.imread("Colonies.jpg")
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

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(thresh)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=thresh)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, no = ndi.label(mask)

labels = watershed(-distance, markers, mask=thresh)

print("After Water Shed:" + str(no))
fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Overlapping objects')

ax[1].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[1].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()


