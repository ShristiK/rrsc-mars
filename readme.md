# Crater and Sand Dune detection on Mars and Moon

1. **Preprocessing** (Both Optical & DEM)
    1. Noise Filter 
    The Median filter will be used with a 5X5 window size.
    2. Area Filter
    The filter is applied on the Median Filtered and DEM image using OTSU's method.
    Features with less than 1000 px are removed
    3. Shape Filter 
    Disk shape with radius 5px is used.
    The area filtered binary image and DEM is converted back to grayscale using matlab function im2uint16.
    The shape filter is applied on the resulting images.

2. **Crater Detection**
    1. **Optical Image**
        1. Wavelet Transform 
            Wavelet transform has been applied on the preprocessed image (TMC and MiniSAR) to get multiscale 2-D wavelet decomposition of gray scale preprocessed image and the corresponding detailed coefficients (horizontal, vertical and diagonal edges responses).
            The steps: 
            1. Read Original Image
                
            2. Apply Preprocessing 
            3.  Multiscale 2-D Wavelet (Haar) decomposition of gray scale image has been done. The Haar wavelet applies a pair of low pass and high pass filters to image decomposition. Upto five levels (scales) of decomposition have been done to get the detailed edges of craters for TMC image and three levels for MiniSAR images. The inbuilt function ‘wavedec2’ of MATLAB has been applied on the image for all the five levels and it returns corresponding wavelet decomposition structure having the decomposition vector C and the corresponding matrix S for all the five levels. The size of vector C and the size of matrix S depend on the type of analyzed image. In our case, C is having (3n+1) sections and size of S is (n+2) - by – 2, where n is the level of decomposition.
                
            4. After decomposition of the image at five levels, it is reconstructed for corresponding levels. 

            5. The corresponding detailed coefficients (horizontal, vertical and diagonal edge responses) for all the five levels have been achieved by passing the wavelet decomposition structure, obtained in previous step, to a MATLAB inbuilt function ‘detcoef2’ as input parameters. 

            6. Visualize all the detailed coefficients for each level as an image.

            7. Continuous Wavelet Transform using Haar wavelet has been applied on the image with a scale (s) varying from 1 to 32 to get image details in the form of continuous wavelet coefficients using MATLAB inbuilt function ‘cwt’. 

