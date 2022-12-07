from astropy.io import fits
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
import matplotlib.pyplot as plt
from pywoc.radial_profile import radial_profile

__all__ = ['woc']

# with NaN dealing & signal strength considered weight
def woc(map1,map2,radii,mask=None,centre=None,pixelsize=1, plot=False,savefig=None):

    if(mask==None):
        mask=(map1*0)+1
        
    if(centre==None):
        centre=np.asarray(np.shape(map1))/2
        print("computing centre",centre)

    maxr=np.shape(map1)[0]/2.0
    step=maxr/20.0
    
    r,DMprofile1=radial_profile(map1,mask,centre,0.0,maxr,step)
    
    nlevel=np.shape(radii)[0]
    if(nlevel<=0):
        print("Error: code requires 2 or more contour levels")
        return
        
    DMradii=np.zeros((nlevel,))
    arearadii=np.zeros((nlevel,)) # contour area
    Overlap=np.zeros((nlevel,))
    massradii1=np.zeros((nlevel,)) # mass sum above the level
    massradii2=np.zeros((nlevel,)) # mass sum above the level

    if(plot):
        fig,ax = plt.subplots(1, nlevel,figsize=(7, 8),sharey=True)
        fig.set_size_inches(w=9,h=3.5)
        if(nlevel>1):
            ax = ax.ravel()
        else:
            ax1=[]
            ax1.append(ax)
            ax=ax1
            
    map1[map2!=map2]=-100.
    map2[map2!=map2]=-100.
    
    Totalmass1=float(np.sum(map1[map1>0.0]))
    Totalmass2=float(np.sum(map2[map2>0.0]))
    r100x, r100y=np.where(map2>1E-20)
    Totalarea2=float(np.shape(r100x)[0])
    
    
    for i in range(nlevel):
        DMradii[i]=np.interp(x=radii[i],xp=pixelsize*r,fp=DMprofile1)
        # Area where DM in 100kpc
        r100x, r100y=np.where(map1>DMradii[i])
        arearadii[i]=float(np.shape(r100x)[0])
        massradii1[i]=np.sum(map1[map1>DMradii[i]])
        #print(arearadii[i])
        if arearadii[i]>Totalarea2:
            print('Warning: map2 too peaky')
            print('Overlap calculation failed')
            return -1000        
        
        level=np.log10(np.max(map2))
        while True:
            level=level-0.001
            tmp1,tmp2=np.where(map2>10**level)
            aICL100=np.shape(tmp1)[0]
            if aICL100>=arearadii[i]:
                level100=level
                break
        
        massradii2[i]=np.sum(map2[map2>10**level100])
        [ox,oy]=np.where((map1>DMradii[i]) & (map2>10**level100))
        Overlap[i]=float(np.shape(ox)[0])
        print("Area at radius of "+str(radii[i])+" = "+str(arearadii[i]))
        print("overlap area at radius of "+str(radii[i])+" = "+str(Overlap[i]))
        print("Enclosed mass1 fraction at radius of "+str(radii[i])+" = "+str(massradii1[i]/Totalmass1))
        print("Enclosed mass2 fraction at radius of "+str(radii[i])+" = "+str(massradii2[i]/Totalmass2))
        
        [a,b]=np.where(map2>10**level100)
        if(plot): ax[i].scatter(a,b,marker='.',alpha=0.3)
        [a,b]=np.where(map1>DMradii[i])
        if(plot): ax[i].scatter(a,b,marker='.',alpha=0.3)
        [ox,oy]=np.where((map1>DMradii[i]) & (map2>10**level100))
        if(plot):
            ax[i].scatter(ox,oy,marker='.',alpha=0.3)
            ax[i].set_xlabel("x [pixel]",fontsize=14)
            if (i==0):
                ax[i].set_ylabel("y [pixel]",fontsize=14)
            ax[i].set(xlim=(0,1000),ylim=(0,1000))
            ax[i].set_aspect('equal')
    
    if(plot): 
        if(savefig==None):
            plt.show()
        else: 
            plt.savefig(savefig)
        
    coefficient1=0.0
    coefficient2=0.0
    for i in range(nlevel):
        coefficient1=coefficient1+(Overlap[i]/arearadii[i])*(arearadii[-1]/arearadii[i])*(Totalmass1/massradii1[i])*(Totalmass2/massradii2[i])
        coefficient2=coefficient2+(arearadii[-1]/arearadii[i])*(Totalmass1/massradii1[i])*(Totalmass2/massradii2[i])

    print('woc:',coefficient1/coefficient2)
    return coefficient1/coefficient2
