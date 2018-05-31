import cv2
import numpy as np 

def match(img, temp):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h= temp.shape[::-1]

    res = cv2.matchTemplate(img_gray, temp, cv2.TM_CCOEFF_NORMED)
    threshold = 0.72
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)
    
    cv2.imshow('detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    #Load image
    img_bgr = cv2.imread('../images/tuts/template.jpg')
    #Load image to search
    temp = cv2.imread('../images/tuts/match.jpg', 0)
    match(img_bgr, temp)

main()