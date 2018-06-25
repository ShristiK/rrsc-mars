import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


img= cv2.imread('images/sample/Sample_1_ESP_045812_1865.jpg',cv2.IMREAD_GRAYSCALE)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(img.shape)
#print(np.amin(img))
#for i in range(0,350):
 #   for j in range(0,467):
  #      print(img[i,j])
#plt.plot(img)

#print(img)
plt.hist(img,bins=256,histtype='step',ec='blue')
#sns.distplot(img,hist=True,kde=False,bins=256)
#histogram,bins=np.histogram(img,bins=256)
#first=bins[:-1]+(bins[1]-bins[0])/2
#plt.plot(bin_centers,pdf,label='PDF')

#plt.legend()
#plt.xlabel('pixel value')
#plt.ylabel('intensity')
plt.show()