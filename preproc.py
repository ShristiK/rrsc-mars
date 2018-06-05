import cv2
import numpy as np 

def noiseFilter(img, ksize):
    res = cv2.medianBlur(img, ksize)
    return res
    
def areaFilter(img):
    img = noiseFilter(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply actual area filter to remove objects with less than 1000 px
        
    return bin

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
    
def preProc(img):
    res1 = noiseFilter(img, 5)
    res2 = areaFilter(res1)
    res3 = shapeFilter(res2, 5)
    return res3

