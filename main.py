import cv2
import modules.crateralgo as ca

# Main function
def main():
    img = cv2.imread('images/sample/Sample_1_PSP_009179_2305.jpg')
    res = ca.waveletTransform(img,level=1)
    
    cv2.imshow('Original', img)
    cv2.imshow('Result', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(__name__ == '__main__'):
    main()
