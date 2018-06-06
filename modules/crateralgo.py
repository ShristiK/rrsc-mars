import cv2
import numpy as np 
import pywt 

# Default mode is Haar and level = 1
def waveletTransform(img , mode = 'haar', level = 1):
    ## Datatype conversions
    # Convert image to grayscale
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to float
    img  =  np.float32(img )   
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
