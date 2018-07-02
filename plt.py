import matplotlib.pyplot as plt 
import h5py 

input_images = h5py.File('./input_data/input_images.hdf5', 'r')
plt.imshow(input_images['input_images'][0][...], origin='upper', cmap='Greys_r', vmin=120, vmax=200)
plt.show()