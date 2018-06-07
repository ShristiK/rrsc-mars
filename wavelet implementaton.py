import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

img= cv2.imread('image.jpeg',cv2.IMREAD_GRAYSCALE)
#print(img.shape)
#print(np.amin(img))
#for i in range(0,350):
 #   for j in range(0,467):
  #      print(img[i,j])
#plt.plot(img)

#print(img)
#plt.hist(img,bins=256, histtype='step',ec='k')
histogram,bins=np.histogram(img,bins=256)

bin_centers=0.5*(bins[1:]+bins[:-1])
pdf=stats.norm.pdf(bin_centers)
plt.plot(bin_centers,histogram,label='sample')
plt.plot(bin_centers,pdf,label='PDF')
plt.legend()
#plt.xlabel('pixel value')
#plt.ylabel('intensity')
plt.show()