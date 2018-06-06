import cv2
import numpy as np 
import modules.crateralgo as ca
import modules.preproc as pp

# Main function
def main():
    img = cv2.imread('images/sample/Sample_1_ESP_022855_1270.jpg')
    res1 = ca.watershed(img)
    
    # kernel = np.ones((3,3), np.uint8)
    # res1 = cv2.morphologyEx(res1, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Result1', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(__name__ == '__main__'):
    main()
