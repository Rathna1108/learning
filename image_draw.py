import cv2 
import numpy as np

image = cv2.imread('textbook_images/page_002.jpg')
# cv2.imshow('image',image)

blank = np.zeros((500,500,3), dtype='uint8')
# cv2.imshow('Blank', blank)

# blank[200:300, 300:400] = 0,255,0
# cv2.imshow('blank', blank)

cv2.rectangle(blank,(200,300), (300,400), (0,0,255),thickness=4)
cv2.imshow('blank', blank)

cv2.waitKey(0)