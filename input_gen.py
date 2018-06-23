########## Imports ##########

# Python 2.7 compatibility.
# from __future__ import absolute_import, division, print_function

from PIL import Image
import modules.input_data_gen as igen
import time

########## Global Variables ##########

# Use MPI4py?  Set this to False if it's not supported by the system.
# use_mpi4py = False

# Source image path.
'''We can put the input image in this variable'''
source_image_path = "./images/sample/Crater_Sample.png"

######### Modification ##############
''' These two are not required for us - !!! Need to Modify/Create new functions to remove dependency on these two variable '''
# LROC crater catalog csv path.
lroc_csv_path = "./catalogues/LROCCraters.csv"
# Head et al. catalog csv path.
head_csv_path = "./catalogues/HeadCraters.csv"

# Output filepath and file header.  Eg. if outhead = "./input_data/train",
# files will have extension "./out/train_inputs.hdf5" and
# "./out/train_targets.hdf5"
outhead = "./input_data/input"

# Number of images to make (if using MPI4py, number of image per thread to
# make).
amt = 1

# Range of image widths, in pixels, to crop from source image (input images
# will be scaled down to ilen). For Orthogonal projection, larger images are
# distorted at their edges, so there is some trade-off between ensuring images
# have minimal distortion, and including the largest craters in the image.
rawlen_range = [500, 6500]

# Distribution to sample from rawlen_range - "uniform" for uniform, and "log"
# for loguniform.
rawlen_dist = 'log'

# Size of input images.
ilen = 256

# Size of target images.
tglen = 256

# [Min long, max long, min lat, max lat] dimensions of source image.
source_cdim = [-180., 180., -60., 60.]

# [Min long, max long, min lat, max lat] dimensions of the region of the source
# to use when randomly cropping.  Used to distinguish training from test sets.
sub_cdim = [-180., 180., -60., 60.]

# Minimum pixel diameter of craters to include in in the target.
minpix = 1.

# Radius of the world in km (1737.4 for Moon).
R_km = 1737.4

### Target mask arguments. ###

# If True, truncate mask where image has padding.
truncate = True

# If rings = True, thickness of ring in pixels.
ringwidth = 1

# If True, script prints out the image it's currently working on.
verbose = True

########## Script ##########

def main(start_time):
    
    istart = 0
    # Read source image and crater catalogs.
    img = Image.open(source_image_path).convert("L")
    
    # Generate input images.
    igen.GenDataset1(img, outhead, rawlen_range=rawlen_range,
                    rawlen_dist=rawlen_dist, ilen=ilen, cdim=sub_cdim,
                    arad=R_km, minpix=minpix, tglen=tglen, binary=True,
                    rings=True, ringwidth=ringwidth, truncate=truncate,
                    amt=amt, istart=istart, verbose=verbose)

    elapsed_time = time.time() - start_time
    if verbose:
        print("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))

if __name__ == '__main__':
    start_time = time.time()
    main(start_time)
