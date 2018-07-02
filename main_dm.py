######## Imports ########
import numpy as np 
import keras as k  
import cv2 
import h5py 
import time

######## Script #######

def main(start_time): 
    # Load Model and Images
    model = k.models.load_model('./modules/model_keras2.h5')
    train_imgs = h5py.File('./input_data/input_images.hdf5', 'r')
    pred = model.predict(np.array(train_imgs['input_images'][0:1][..., np.newaxis]))

    # Print out time it takes for prediction 
    elapsed_time = time.time() - start_time
    print("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))
    
    # Display Results
    cv2.imshow('Predicted', pred[0])
    cv2.imshow('Original', train_imgs['input_images'][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


######## To Invoke Main Function #######
if __name__ == '__main__':
    start_time = time.time()
    main(start_time) 