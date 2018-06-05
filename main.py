import cv2
import numpy as np
from preproc import preProc 

# Main function
def main():
    # original image 
    img = cv2.imread('images/tuts/tulip1.jpg', -1)
    res = preProc(img)
    cv2.imshow('Result', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
main()
