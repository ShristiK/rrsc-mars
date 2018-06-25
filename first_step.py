import cv2
import numpy as np
import matplotlib.pyplot as plt
#import modules.crateralgo as ca


img= cv2.imread('images/sample/Sample_1_ESP_045812_1865.jpg',cv2.IMREAD_GRAYSCALE)
#img=cv2.imread('image.jpeg',cv2.IMREAD_GRAYSCALE)
img1=cv2.blur(img,(5,5))
img1=cv2.GaussianBlur(img1,(5,5),0)
img2=cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2 )
#final_img=cv2.Scharr(img2,cv2.CV_64F)
final_img=cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=-1)
abs_sobel64f=np.absolute(final_img)
final_img=np.uint8(abs_sobel64f)
#circles=cv2.HoughCircles(final_img,cv2.HOUGH_GRADIENT,1,21,param1=50,param2=30,minRadius=0,maxRadius=0)
#circles=np.uint16(np.around(circles))
#for i in circles[0,:]:
 #cv2.circle(final_img,(i[0],i[1]),i[2],(0,255,0),2)
 #cv2.circle(final_img,(i[0],i[1]),2,(0,0,255),3)
#final_img=cv2.Laplacian(img2,cv2.CV_64F)
#final_img=cv2.Canny(img2,100,200)
#ret1,img2=cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#plt.subplot(2,2,1), plt.imshow(img,'gray')
#plt.title('original'),plt.xticks([]),plt.yticks([])
#plt.show()
#plt.subplot(1,2,1),\
plt.imshow(img1,'gray')
plt.title('after_blur(g)'),plt.xticks([]),plt.yticks([])
plt.show()
#plt.subplot(1,2,2),\
plt.imshow(img2,'gray')
plt.title('after_thresh'),plt.xticks([]),plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(res,'gray')
#plt.title('after_scharr and circle'),plt.xticks([]),plt.yticks([])
plt.show()




