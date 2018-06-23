import numpy as np 
import matplotlib.pyplot as plt 
import os 
from PIL import Image 
import keras as k 
import sys 
import cv2 
import h5py 
deepmoon_path = os.path.dirname(os.getcwd())
import utils.template_match_target as tmt  

model = k.models.load_model('./model_keras2.h5')
train_imgs = h5py.File('./INP_images.hdf5', 'r')
pred = model.predict(np.array(train_imgs['input_images'][0:1][..., np.newaxis]))


# cv2.imshow('Original', train_imgs['input_images'][10:11][...] )
cv2.imshow('Predicted', pred[0])
cv2.imshow('Original', train_imgs['input_images'][0])
cv2.waitKey(0)
cv2.destroyAllWindows()
# fig = plt.figure(figsize=[16, 16])
# [[ax1, ax2], [ax3, ax4]] = fig.subplots(2,2)
# ax1.imshow(img, origin='upper', cmap='Greys_r', vmin=50, vmax=200)
# ax2.imshow(img, origin='upper', cmap='Greys_r', vmin=0, vmax=1)
# ax3.imshow(pred[0], origin='upper', cmap='Greys_r', vmin=0, vmax=1)
# ax4.imshow(img, origin='upper', cmap="Greys_r")
# ax1.set_title('IMG')
# ax2.set_title('IMG')
# ax3.set_title('CNN Predictions')
# ax4.set_title('Post-CNN Craters')
# plt.show()