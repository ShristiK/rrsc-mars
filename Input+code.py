
# coding: utf-8

# In[ ]:


import cartopy.crs as ccrs
import cv2
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import time
import cartopy.img_transform as cimg


# In[ ]:


def create_ring(rrad,rwidth):
    """in this taking radius and ring width as input,then making a mask with 
     crater drawn on it"""
    rtemp=int(np.ceil(rrad + rwidth / 2.)) + 1
    mask =np.zeros([2*rtemp+1,2*rtemp+1])
    plt.subplot(1,2,1)
    plt.imshow(mask)
    ring=cv2.circle(mask,(rtemp,rtemp),int(rtemp),1,thickness=1)
    return(ring.astype(float))


# In[ ]:


def get_coords(center,im_len,ker_len,rad):
    """getting coordinates where to merge the mask"""
    l=center-rad
    r=center+rad+1
    if left < 0:
        img_l = 0
        g_l = -left
    else:
        img_l = left
        g_l = 0
    if right > imglen:
        img_r = imglen
        g_r = ker_shp - (right - imglen)
    else:
        img_r = right
        g_r = ker_shp

    return [img_l, img_r, g_l, g_r]
    


# In[ ]:


def create_mask(catalogue,img,rw=1):
    imgshape=img.shape([:2])
    mask=np.zeros(imgshape)
    cx=catalogue.["x"].values.astype('int')
    cy=catalogue.["y"].values.astype('int')
    radius=catalogue.["Diameter (km)"].values/2.
    
    for i in range(catalogue.shape[0]):
        kernel = create_ring(radius[i],rwidth=rw)
        ker_shape=kernel.shape[0]
        kshape_half =ker_shape\\2
        
        [imxl,imgxr,gxl,gxr]=get_coords(cx[i],imgshape[1],ker_shape,kshape_half)
        [imyl,imgyr,gyl,gyr]=get_coords(cy[i],imgshape[0],ker_shape,kshape_half)
        mask[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]
        mask = (mask > 0).astype(float)
        if img.ndim == 3:
            mask[img[:, :, 0] == 0] = 0
        else:
            mask[img == 0] = 0
        return(mask)
            


# In[ ]:


def Resample(cat,ll_comb):
    crt_x =(cat["long"]>= ll_comb[0]) & (cat["long"] <= ll_comb[1])
    crt_y =(cat["lat"]>= ll_comb[2]) & (cat["lat"] <= ll_comb[3])
    crt_f = cat.loc(crt_x,crt_y ,:).copy()
    ctr_f.reset_index(inplace =True,drop =True)
    return(ctr_f)


# In[ ]:


def conv_pix_to_coord(x_pix,y_pix,i_h,i_l):
    cx=(x_pix/i_h) * 360.0 -180.0
    cy=(y_pix/i_l)*180.0-90.0
    return (cx,cy)

    
def conv_coord_to_pix(cx,cy,i_h,i_l):
    x= i_h * (cx +180.0) / 360.0
    y=i_l*(cy +90.0) / 180.0
    return (x,y)

def km_to_pix(img_h, latext,dc=distortion_coefficient):
    return (180. / np.pi) * img_h * dc / latext / 1737.4


# In[ ]:


def regrid_shape_aspect(r_shape,ext):
    if not isinstance(regrid_shape, collections.Sequence):
        target_size = int(regrid_shape)
        x_range, y_range = np.diff(target_extent)[::2]
        desired_aspect = x_range / y_range
        if x_range >= y_range:
            regrid_shape = (target_size * desired_aspect, target_size)
        else:
            regrid_shape = (target_size, target_size / desired_aspect)
    return regrid_shape


# In[ ]:


def Warp_image(img,iso,iex,ortho,ox):
    
    if iproj == oproj:
        raise Warning("Input and output transforms are identical!"
                      "Returing input!")
        return img
    origin=upper
    img=img[::-1]
    regrid_shape = 1.2 * min(img.shape)
    regrid_shape = regrid_shape_aspect(regrid_shape,
                             oextent)
    iextent_nozeros = np.array(iextent)
    iextent_nozeros[iextent_nozeros == 0] = 1e-8
    iextent_nozeros = list(iextent_nozeros)
    
    
    imgout, extent = cimg.warp_array(img,source_proj=iproj,
                                     source_extent=iextent_nozeros,
                                     target_proj=oproj,
                                     target_res=regrid_shape,
                                     target_extent=oextent,
                                     mask_extrapolated=True)
    imgout = imgout[::-1]
    return(imgout)


# In[ ]:


def WarpImagePad(image,iso,iex,ortho,ox):
    assert image.sum() > 0, "Image input to WarpImagePad is blank!"
    bgval=0
    wimg=Warp_image(image, iso, iex, ortho, ox)
    imgw = np.ma.filled(imgw, fill_value=bgval)
    imgw_loh = imgw.shape[1] / imgw.shape[0]
    if imgw_loh >(img.shape[1]/img.shape[0]):
        dim=(img.shape[0],imgw.shape[1]*imgw_loh)
        imgw =cv2.resize(imgw,dim,interpolation=cv2.INTER_AREA)
    else:
        dim=(imgw.shape[0]*imgw_loh,img.shape[1])
        imgw =cv2.resize(imgw,dim,interpolation=cv2.INTER_AREA)
    imgo=np.zeros((image.shape[0],image.shape[1]))
    imgo[0:imgw.shape[1],0:imgw.shape[0]]+=imgw[:,:]
    return (imgo,imgw.shape)
    
    
def WarpCraterLoc(craters, geoproj, oproj, oextent, imgdim, llbd):
    ctr_xlim = ((craters["Long"] >= llbd[0]) &
                    (craters["Long"] <= llbd[1]))
    ctr_ylim = ((craters["Lat"] >= llbd[2]) &
                    (craters["Lat"] <= llbd[3]))
    ctr_wrp = craters.loc[ctr_xlim & ctr_ylim, :].copy()
    if ctr_wrp.shape[0]:
        ilong = ctr_wrp["Long"].as_matrix()
        ilat = ctr_wrp["Lat"].as_matrix()
        res = oproj.transform_points(x=ilong, y=ilat,
                                     src_crs=geoproj)[:, :2]
        ctr_wrp["x"], ctr_wrp["y"] = conv_coord_to_pix(res[:, 0], res[:, 1],
                                                    imgdim[0],imgdim[1])
    else:
        ctr_wrp["x"] = []
        ctr_wrp["y"] = []

    return ctr_wrp


# In[ ]:


def AddPlateCarree_XY(catalogue,img_h,img_l):
    
    x,y=conv_coord_to_pix(catalogue["long"].as_matrix(),catalogue["lat"].as_matrix(),
                         img_h,img_l)
    catalogue["x"]=x
    catalogue["y"]=y
    return(catalogue)

def PlateCarree_to_Ortho(im,latlong,catalogue,iglobe):
    geoproj = ccrs.Geodetic(globe=iglobe)
    iproj = ccrs.PlateCarree(globe=iglobe)
    oproj = ccrs.Orthographic(central_longitude=np.mean(latlong[:2]),
                              central_latitude=np.mean(latlong[2:]),
                              globe=iglobe)
    xll = np.array([latlong[0], np.mean(latlong[:2]), latlong[1]])
    yll = np.array([latlong[2], np.mean(latlong[2:]), latlong[3]])
    xll, yll = np.meshgrid(xll, yll)
    xll = xll.ravel()
    yll = yll.ravel()
    res = iproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    iextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    res = oproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    oextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    oaspect = (oextent[1] - oextent[0]) / (oextent[3] - oextent[2])
    if oaspect < 0.:
        return [None, None]
    imgo, imgwshp = WarpImagePad(img, iproj, iextent, oproj, oextent)
    ctr_xy = WarpCraterLoc(catalogue , geoproj, oproj, oextent, imgwshp,latlong)
    distortion_coefficient = ((res[7, 1] - res[1, 1]) /
                              (oextent[3] - oextent[2]))
    if distortion_coefficient < 0.7:
        raise ValueError("Distortion Coefficient cannot be"
                         " {0:.2f}!".format(distortion_coefficient))
    pixperkm = km_to_pix(imgo.shape[0], llbd[3] - llbd[2],
                          dc=distortion_coefficient)
    ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pixperkm
    centrallonglat = pd.DataFrame({"Long": [xll[4]], "Lat": [yll[4]]})
    centrallonglat_xy = WarpCraterLoc(centrallonglat, geoproj, oproj, oextent,
                                      imgwshp, llbd=latlong)
    return [imgo, ctr_xy, distortion_coefficient, centrallonglat_xy]


# In[ ]:


def gen_dataset(cat,image,amt=100):
    cat=AddPlateCarree_XY(cat,image.shape[0],image.shape[1])
    r_min=np.log10(500)
    r_max=np.log10(6500)
    
    def random_sampler():
        return int(10**np.random.uniform(r_min, r_max))
    
    outhead ="./input_data/train"
    img_h5=h5py.File(outhead+'_images.hdf5','w')
    imgs_h5_inputs = imgs_h5.create_dataset("input_images", (amt,256, 256),
                                            dtype='uint8')
    imgs_h5_inputs.attrs['definition'] = "Input image dataset."
    imgs_h5_tgts = imgs_h5.create_dataset("target_masks", (amt, 256, 256),
                                          dtype='float32')
    imgs_h5_tgts.attrs['definition'] = "Target mask dataset."
    imgs_h5_llbd = imgs_h5.create_group("longlat_bounds")
    imgs_h5_llbd.attrs['definition'] = ("(long min, long max, lat min, "
                                        "lat max) of the cropped image.")
    imgs_h5_box = imgs_h5.create_group("pix_bounds")
    imgs_h5_box.attrs['definition'] = ("Pixel bounds of the Global DEM region"
                                       " that was cropped for the image.")
    imgs_h5_dc = imgs_h5.create_group("pix_distortion_coefficient")
    imgs_h5_dc.attrs['definition'] = ("Distortion coefficient due to "
                                      "projection transformation.")
    imgs_h5_cll = imgs_h5.create_group("cll_xy")
    imgs_h5_cll.attrs['definition'] = ("(x, y) pixel coordinates of the "
                                       "central long / lat.")
    
    craters_h5=pd.HDFStore(outhead + '_craters.hdf5', 'w')
    for i in range(amt):
        img_len =random_sampler()
        xc=np.random.randint(0,image.shape[1]-img_len)
        yc=np.random.randint(0,image.shape[0]-img_len)
        arr=image[xc:xc+img_len,yc:yc+img_len]
        ix=[xc,xc+img_len]
        iy=[yc,yc+img_len]
        box=np.array([xc,yc,xc+img_len,yc+img_len],dtype='int32')
        llong,llat= conv_pix_to_coord(ix,iy,image.shape[0],image.shape[1])
        l_comb=np.r_[llong,llat[::-1]]
        dim=(256,256)
        arr=cv2.resize(arr,dim,interpolation=cv2.INTER_AREA)
        iglobe = ccrs.Globe(semimajor_axis=arad*1000., semiminor_axis=arad*1000.,
                        ellipse=None)
        cat =Resample(cat,l_comb)
        [imgo, ctr_xy, distortion_coefficient, clonglat_xy]=PlateCarree_to_Ortho(arr,
                                                             l_comb,cat,iglobe)                  
        assert imgo.sum() > 0, ("Sum of imgo is zero!  There likely was "
                                    "an error in projecting the cropped "
                                    "image.")
        tgt =cv2.resize(imgo,dim,interpolation=cv2.INTER_AREA)
        mask = create_mask(ctr_xy, tgt,rw=1)
        
        imgs_h5_inputs[i, ...] = imgo
        imgs_h5_tgts[i, ...] = mask
        sds_box = imgs_h5_box.create_dataset(i, (4,), dtype='int32')
        sds_box[...] = box
        sds_llbd = imgs_h5_llbd.create_dataset(i, (4,), dtype='float')
        sds_llbd[...] = l_comb
        sds_dc = imgs_h5_dc.create_dataset(i, (1,), dtype='float')
        sds_dc[...] = np.array([distortion_coefficient])
        sds_cll = imgs_h5_cll.create_dataset(i, (2,), dtype='float')
        sds_cll[...] = clonglat_xy.loc[:, ['x', 'y']].as_matrix().ravel()

        craters_h5[i] = ctr_xy

        imgs_h5.flush()
        craters_h5.flush()
        
    imgs_h5.close()
    craters_h5.close()

