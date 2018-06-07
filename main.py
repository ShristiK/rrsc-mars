import cv2
import numpy as np 
<<<<<<< HEAD
import preproc as pp
=======
import modules.crateralgo as ca
import modules.preproc as pp
>>>>>>> ebb21774f93e35e45988fa4b0bc8e3dd94c86445

# Main function
def main():
    img = cv2.imread('images/sample/Sample_1_ESP_045812_1865.jpg')
    # res1 = ca.watershed(img)
    res1 = ca.houghTransform(img)
    
    # kernel = np.ones((3,3), np.uint8)
    # res1 = cv2.morphologyEx(res1, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Result1', res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

<<<<<<< HEAD
# Main function
def main():
    # original image 
    img = cv2.imread('images/tuts/tulip1.jpg', -1)
    mars1 = cv2.imread('images/mars/ESP_022855_1270.jpg')
    # extractRed(img)
    # extractEdge(img)
    #transform(img)
    img1 = pp.noiseFilter(img, 5)
    cv2.imshow('original', img)
    cv2.imshow('filter', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
=======
if(__name__ == '__main__'):
    main()
>>>>>>> ebb21774f93e35e45988fa4b0bc8e3dd94c86445
