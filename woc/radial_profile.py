from astropy.io import fits
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel

__all__ = ['radial_profile']

def radial_profile(data, mask, center, rmin, rmax,width):
    y,x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[0])**2+(y-center[1])**2)
    sr = r.flat
    sim = data.flat  
    msk = mask.flat
    msk[sim==-np.inf]=0
    msk[sim==np.inf]=0
    rbins=np.arange(rmin,rmax,width)
    nbins=len(rbins)
    array=np.zeros(np.shape(rbins[:-1]))
    
    n=0
    for rr in rbins[:-1]:
        inds=np.where((sr>=rr) & (sr<=rr+width) & (sim==sim) & (msk==1))
        array[n]=np.median(sim[np.squeeze(inds)]) 
        n=n+1
        #print(len(np.squeeze(inds)))
    return rbins[:-1]+0.5*width, array

