import cv2
import numpy as np
from preproc import *
import pywt

def haar(imArray, mode = 'haar', level = 2):
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_BGR2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs= pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0  

    # reconstruction
    imArray_H= pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)
    #Display result
    return imArray_H

# Main function
def main():
    # original image 
    img = cv2.imread('images/tuts/gold.png')
    img_f = noiseFilter(img, 5)
    img_f = haar(img_f)
    kernel = np.ones((5,5), np.uint8)
    img_f = cv2.erode(img_f, kernel, iterations = 1)
    cv2.imshow('FilterResult', img_f)
    cv2.imshow('Result',haar(img) )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
