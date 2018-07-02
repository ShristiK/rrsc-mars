import cv2
import numpy as np
import modules.crateralgo as ca
import modules.preproc as pp

# Main function
def main():
    # Load Images
    img1 = cv2.imread('images/sample/DEM/Sample_1.png')
    img1 = cv2.resize(img1,None,  fx = 0.1, fy=0.1)
    # res1 = ca.houghTransform(img1)
    res2 = ca.watershed(img1)

    # Display Images
    cv2.imshow('Original1', img1)
    # cv2.imshow('Result1', res1)
    cv2.imshow('Result2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if(__name__ == '__main__'):
    main()
