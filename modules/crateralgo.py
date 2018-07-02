import cv2
import numpy as np 
# import pywt

# Default mode is Haar and level = 1
def waveletTransform(img , mode = 'haar', level = 3):
    ## Datatype conversions
    # Convert image to grayscale
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to float
    img  =  np.float32(img)   
    img /= 255
    # Compute coefficients for transformation
    coeffs = pywt.wavedec2(img, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)  
    coeffs_H[0] *= 0 

    # Reconstruction of Image
    img_H= pywt.waverec2(coeffs_H, mode)
    img_H *= 255
    img_H =  np.uint8(img_H)
    
    # Return resulting image
    return img_H

def watershed(img):
    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
 
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
 
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker Labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers +1

    markers[unknown==255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255,0,0]
    return img

def houghTransform(img):
    img = cv2.medianBlur(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,21,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img, (i[0],i[1]), i[2], (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0],i[1]), 2, (0,0,255), 3)
    
    return img
    