import cv2
import numpy as np 

# Applies medianBlur of ksize kernel
def noiseFilter(img, ksize):
    res = cv2.medianBlur(img, ksize)
    return res
    
# Remove Objects with less than 1000px (Incomplete)
def areaFilter(img):
    img = noiseFilter(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , biny = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply actual area filter to remove objects with less than 1000 px
    kernel = np.ones((10,10), np.uint8)
    biny = cv2.morphologyEx(biny, cv2.MORPH_OPEN, kernel)
    # img_mod = cv2.bitwise_and(img, img, mask = biny)
    # img_mod = cv2.GaussianBlur(img_mod, (15,15), 0)
    res = cv2.bitwise_and(img,img, mask = cv2.bitwise_not(biny))

    return res 

# Apply Disc shaped filter to remove non disc elements (Incomplete)
def shapeFilter(img, r):
    size = 2*r+1
    # Disk shaped kernel 
    disk = np.zeros((size,size), np.uint8)
    for i in range(0,size):
        for j in range(0,size):
            d = (i-r)**2 + (j-r)**2
            d = np.sqrt(d)
            if(d<=r):
                disk[i,j] = 1
    res = cv2.filter2D(img, -1, disk)
    return res

# Applies Preprocessing method implemented in the research paper (Incomplete)
def preProc(img):
    res1 = noiseFilter(img, 5)
    res2 = areaFilter(res1)
    res3 = shapeFilter(res2, 5)
    return res3

