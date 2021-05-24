import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology
import matplotlib.pyplot as plt
import cv2
from skimage import feature
import numpy as np

def auto_crop(img,itration = 1):
    # Load image, create blank mask, grayscale, Otsu's threshold
    image = cv2.imread(img)
    original = image.copy()
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours and filter using contour area + contour approximation
    # Determine perfect circle contour then draw onto blank mask
    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) > 4 and area > 10000 and area < 500000:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(mask, (int(x), int(y)), int(r), (255, 255, 255), -1)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 3)

    # Extract ROI
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(mask)
    mask_ROI = mask[y:y+h, x:x+w]
    image_ROI = original[y:y+h, x:x+w]

    # Bitwise-and for result
    result = cv2.bitwise_and(image_ROI, image_ROI, mask=mask_ROI)
    result[mask_ROI==0] = (255,255,255) # Color background white

    return result

img = auto_crop("C_8.jpg")
# Binary image, post-process the binary mask and compute labels
mask = feature.canny(img)
threshold = filters.threshold_otsu(img)
mask = img > threshold
cv2.imshow(mask)
cv2.waitKey(0)

mask = morphology.remove_small_objects(mask, 50)
mask = morphology.remove_small_holes(mask, 50)
mask = morphology.binary_dilation(mask)
mask = morphology.dilation(mask)
mask = morphology.erosion(mask)
labels = measure.label(mask, connectivity=2)

fig = px.imshow(mask, binary_string=True)
fig.update_traces(hoverinfo='skip') # hover is only for label info
print(labels.max()-1)

props = measure.regionprops(labels, img)
properties = ['area', 'eccentricity', 'perimeter', 'mean_intensity']

# For each label, add a filled scatter trace for its contour,
# and display the properties of the label in the hover of this trace.
for index in range(0, labels.max()):
    label = props[index].label
    contour = measure.find_contours(labels == label, 0.5)[0]
    y, x = contour.T
    hoverinfo = ''
    for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))
fig.show()

