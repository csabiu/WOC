from astropy.io import fits
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
import matplotlib.pyplot as plt
from pywoc.radial_profile import radial_profile
import time
from numba import jit

__all__ = ['woc']


@jit
def findX(map1,level):
    r100x, r100y=np.where(map1>level)
    aa=float(np.shape(r100x)[0])
    bb=np.sum(map1[map1>level])
    return aa,bb

# with NaN dealing & signal strength considered weight
def woc(map1,map2,radii,mask=None,centre=None,pixelsize=1, plot=False,savefig=None, rbins=20, maxr=None,
        method='median'):
    start = time.time()
    if(mask==None):
        mask=(map1*0)+1
        
    if(centre==None): # use maximum point
        #centre=np.squeeze(np.where(map1 == map1.max()))
        #print("computing centre",centre)
        centre = np.unravel_index(np.argmax(map1), map1.shape)
        print("computing centre",centre)

    if(centre=="mid"): # use geomtric centre
        centre=np.asarray(np.shape(map1))/2
        print("computing centre",centre)
        
    if(maxr==None):
        maxr=np.shape(map1)[0]/2.0
        
    step=maxr/rbins

    if(method=='median'):
        r,DMprofile1=radial_profile(map1,mask,(centre[1],centre[0]),0.0,maxr,step,method='median')
    if(method=='mean'):
        r,DMprofile1=radial_profile(map1,mask,(centre[1],centre[0]),0.0,maxr,step,method='mean')
        
    if(plot):
        plt.plot(r*pixelsize,DMprofile1,'o')
        plt.xlabel('length unit')
        plt.ylabel('density')
        if(savefig==None):
            plt.show()
        else: 
            plt.savefig(savefig+'.rad.jpg')
        plt.close()

    
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
    
    map2radii=np.zeros((nlevel,))

    for i in range(nlevel):
        DMradii[i]=np.interp(x=radii[i],xp=pixelsize*r,fp=DMprofile1)
        # Area where DM in 100kpc
        #r100x, r100y=np.where(map1>DMradii[i])
        #arearadii[i]=float(np.shape(r100x)[0])
        #massradii1[i]=np.sum(map1[map1>DMradii[i]])
        arearadii[i], massradii1[i]=findX(map1,DMradii[i])

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
        
        map2radii[i]=10**level100

        massradii2[i]=np.sum(map2[map2>10**level100])
        [ox,oy]=np.where((map1>DMradii[i]) & (map2>10**level100))
        Overlap[i]=float(np.shape(ox)[0])
        print("Area at radius of "+str(radii[i])+" = "+str(arearadii[i]))
        print("Levels at radius of "+str(radii[i])+" map 1 = "+str(DMradii[i])+" map 2 = "+ str(10**level100))
        print("overlap area at radius of "+str(radii[i])+" = "+str(Overlap[i]))
        print("Enclosed mass1 fraction at radius of "+str(radii[i])+" = "+str(massradii1[i]/Totalmass1))
        print("Enclosed mass2 fraction at radius of "+str(radii[i])+" = "+str(massradii2[i]/Totalmass2))


        [a,b]=np.where(map1>DMradii[i])
        if(plot): ax[i].scatter(a,b,marker='.',alpha=0.3,color='orange',label='map1')
        
        [a,b]=np.where(map2>10**level100)
        if(plot): ax[i].scatter(a,b,marker='.',alpha=0.3,color='blue',label='map2')

        [ox,oy]=np.where((map1>DMradii[i]) & (map2>10**level100))
        if(plot):
            ax[i].scatter(ox,oy,marker='.',alpha=0.3,color='green')
            ax[i].set_xlabel("x [pixel]",fontsize=14)
            if (i==0):
                ax[i].set_ylabel("y [pixel]",fontsize=14)
            ax[i].set(xlim=(0,np.shape(map1)[0]),ylim=(0,np.shape(map1)[0]))
            ax[i].set_aspect('equal')
            ax[i].plot(centre[0],centre[1],'k+')
            
    sum1=np.sum(DMradii)
    sum2=np.sum(map2radii)
    sum3=np.sum(arearadii)
    sum4=0.0
    for i in range(nlevel):            
        sum4=sum4+arearadii[-1]/arearadii[i]

    coefficient1=0.0
    coefficient2=0.0
    for i in range(nlevel):
        #coefficient1=coefficient1+(Overlap[i]/arearadii[i])*(arearadii[-1]/arearadii[i])*(Totalmass1/massradii1[i])*(Totalmass2/massradii2[i])
        #coefficient2=coefficient2+(arearadii[-1]/arearadii[i])*(Totalmass1/massradii1[i])*(Totalmass2/massradii2[i])

        coefficient1=coefficient1+(Overlap[i]/arearadii[i])*(((arearadii[-1]/arearadii[i])/sum4) +(DMradii[i]/sum1)+(map2radii[i]/sum2))
        coefficient2=coefficient2+((arearadii[-1]/arearadii[i])/sum4)+(DMradii[i]/sum1)+(map2radii[i]/sum2)

        #print (Totalmass1/massradii1[i])
        print(Overlap[i]/arearadii[i])
        print(DMradii[i]/sum1)
        print(map2radii[i]/sum2)
        print((arearadii[-1]/arearadii[i])/sum4)

    if(plot):
        plt.legend()
        plt.title('woc: '+str(coefficient1/coefficient2))
        
        if(savefig==None):
            plt.show()
        else: 
            plt.savefig(savefig)
        plt.close()
        
    print('woc: ',coefficient1/coefficient2)
    print('time taken: ',time.time()-start)
    return coefficient1/coefficient2
