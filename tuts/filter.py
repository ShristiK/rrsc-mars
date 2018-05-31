import cv2
import numpy as np 

img = cv2.imread('tulip1.jpg', -1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

lower_red = np.array([0,120,50])
upper_red = np.array([10,255,255])
lower_red1 = np.array([155,120,50])
upper_red1 = np.array([180,255,255])

mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red1, upper_red1)
#mask = mask1+mask2
mask = cv2.bitwise_or(mask1, mask2)
res = cv2.bitwise_and(img, img, mask = mask)

mblur = cv2.medianBlur(res, 15)
gblur = cv2.GaussianBlur(res, (15,15), 0)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(mask, kernel, iterations = 1)
dilation = cv2.dilate(mask, kernel, iterations = 1)

cv2.imshow('original', img)
#cv2.imshow('mask', mask)
# cv2.imshow('res', res)
cv2.imshow('mblur', mblur)
cv2.imshow('gblur', gblur)
cv2.imshow('erosion', erosion)
cv2.imshow('dilation', dilation)


cv2.waitKey(0)
cv2.destroyAllWindows()
