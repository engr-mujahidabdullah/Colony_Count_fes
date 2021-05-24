import cv2
import matplotlib.pyplot as plt
from PIL import Image 

image = cv2.imread('C_2.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# applying gaussian blur on the image
blur = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_REFLECT)

# detection of the edges
edge = cv2.Canny(blur,0,150)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, kernel, iterations=1)

contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    if cv2.contourArea(contour) > 10:
        if cv2.contourArea(contour) < 400:
            cv2.drawContours(image, contour, -1, (255, 65, 40), 1)

plt.imshow(image)
plt.show()





