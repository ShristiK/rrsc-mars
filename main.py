import cv2
import numpy as np 

def transform(img):
    #Removing FalsePositives and FalseNegatives
    kernel = np.ones((5,5), np.uint8)
    mask = extractRed(img) 
    open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #Show Images
    cv2.imshow('open',open)
    cv2.imshow('close',close)
    
    #Escape Windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# custom function to extract red areas of an image
def extractRed(img):
    #Converting img to hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Add Color filter (RED)
    lower_red = np.array([0,200,50])
    upper_red = np.array([10,255,255])
    lower_red1 = np.array([155,200,50])
    upper_red1 = np.array([180,255,255])

    # Generate Color Mask 
    mask = cv2.bitwise_or(cv2.inRange(img_hsv, lower_red, upper_red),cv2.inRange(img_hsv, lower_red1, upper_red1))
    img_red = cv2.bitwise_and(img, img, mask = mask)

    # Apply Blur and smoothen
    img_red = cv2.medianBlur(img_red, 3)
    kernel = np.ones((5,5), np.uint8)
    img_red_d = cv2.dilate(mask, kernel, iterations = 1)

    # Display Images
    # cv2.imshow('original', img)
    # cv2.imshow('img_red', img_red)
    # cv2.imshow('img_red_dilated', img_red_d)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()  
    return mask

# applies canny algo to extract edges and returns mask  
def extractEdge(img):

    # Apply Blur and convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.medianBlur(img, 3)

    # Need to automate threshold values
    otsu, _ = cv2.threshold(img, 0, 225, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(otsu)
    edges = cv2.Canny(img, int(otsu) , int(otsu*1.5))
    
    # img = cv2.resize(img,(0,0), fx=0.1, fy=0.1)
    # edges = cv2.resize(edges, (0,0), fx=0.1, fy=0.1)
    #Display images
    cv2.imshow('original', img)
    cv2.imshow('Edges', edges)
    
    #Press Esc to exit all open windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
def main():
    # original image 
    img = cv2.imread('images/tuts/tulip1.jpg', -1)
    mars1 = cv2.imread('images/mars/ESP_022855_1270.jpg')
    # extractRed(img)
    # extractEdge(img)
    #transform(img)
    extractEdge(mars1[1000:1400, 1800:2000])

main()
