import cv2
import numpy as np 
import matplotlib.pyplot as plt 

#img = cv2.imread('index.jpg', cv2.IMREAD_GRAYSCALE)
#IMREAD_GRAYSCALE = 0
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1

# plt.imshow(img, cmap ='gray', interpolation= 'bicubic')
# plt.plot([50,100], [80,100], 'c', linewidth= 2)
# plt.show();

# apple = img[37:111, 107:194]
# img[0:74, 0:87] = apple

img1 = cv2.imread('gold.png',-1)
img2 = cv2.imread('art.jpg',-1)
img3 = cv2.imread('logo.jpg', -1)

#add = img1+img2
#add = cv2.add(img1, img2)

#add = cv2.addWeighted(img1, 0.35, img2, 0.65,0)
r,c,ch = img3.shape
roi = img2[0:r, 0:c]
ret,mask = cv2.threshold(img3, 220, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow('ap', mask)

mask_inv = cv2.bitwise_not(mask)

img2_bck = cv2.bitwise_and(roi, roi, mask = mask_inv)
img3_fg  = cv2.bitwise_and(img3, img3, mask = mask)

img2[0:r, 0:c] = cv2.add(img2_bck, img3_fg)
cv2.imshow('res', img2)
#cv2.imshow('image', add)
cv2.waitKey(0)
cv2.destroyAllWindows()
