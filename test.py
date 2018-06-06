import cv2

img = cv2.imread('/images/tuts/tulip1.jpg', -1)
print(img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()