import cv2
import numpy as np 
import modules.crateralgo as ca
import modules.preproc as pp

# Main function
def main():
    img1 = cv2.imread('images/sample/Sample_1_ESP_045812_1865.jpg')
    img2 = cv2.imread('images/sample/Sample_1_ESP_022855_1270.jpg')
    res1 = ca.houghTransform(img1)
    res2 = ca.watershed(img2)
    org2 = cv2.imread('images/sample/Sample_1_ESP_022855_1270.jpg')

    cv2.imshow('Original1', img1)
    cv2.imshow('Result1', res1)
    cv2.waitKey(0)
    cv2.imshow('Original2', org2)
    cv2.imshow('Result2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if(__name__ == '__main__'):
    main()
