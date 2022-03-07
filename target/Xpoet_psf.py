import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from skimage.transform import downscale_local_mean, resize

from scipy.io import loadmat
from scipy.signal import fftconvolve

from photutils.aperture import CircularAperture,aperture_photometry
from photutils.centroids import centroid_com

from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange

psf_dir = 'Kernels/'
# psf_names = ['Pandora_nir_20210602_trefoil.mat']
psf_names = ['POET_PSF_CenterFoV_VNIR.txt']

class ModelPars:
    """Default Model Parameters
    """

    nplanetmax=9 #code is hardwired to have upto 9 transiting planets.
    #default parameters -- these will cause the program to end quickly
    tstart=0.0 #start time (days)
    tend=1.0 #end time (days)
    iframe=2 #number of frames used to generate exposure time (needed for pointing jitter)
    exptime=1 #exposure time (s)
    deadtime=0.0 #dead time (s)
    modelfile='null' #stellar spectrum file name
    nmodeltype=2 #stellar spectrum type. 1=BT-Settl, 2=Atlas-9+NL limbdarkening
    rvstar=0.0 #radial velocity of star (km/s)
    vsini=0.0 #projected rotation of star (km/s)
    pmodelfile=[None]*nplanetmax #file with Rp/Rs values
    pmodeltype=[None]*nplanetmax #Type of planet file
    emisfile=[None]*nplanetmax #file with emission spectrum
    ttvfile=[None]*nplanetmax #file with O-C measurements
    #nplanet is tracked by pmodelfile.
    nplanet=0 #number of planets -- default is no planets - you will get staronly sim.
    sol=np.zeros(nplanetmax*8+1)
    sol[0]=1.0 #mean stellar density [g/cc]
    xout=1024  #x-axis
    xpad=10    #padding to deal with convolution fall-off
    ypad=10    #padding to deal with convolution fall-off
    yout=1024   #y-axis
    noversample=2 #oversampling
    gain=1.6 # electronic gain [e-/adu]
    saturation=65536.0 #saturation
    jitter_dis=1.0 #pointing jitter in dispersion axis [pixels, rms]
    jitter_spa=1.0 #pointing jitter in spatial axis [pixels, rms]
    readnoise=8.0 #readnoise electrons
    gain=6.1 #e-/ADU

def addflux2pix(px,py,pixels,fmod):
    """Usage: pixels=addflux2pix(px,py,pixels,fmod)

    Drizel Flux onto Pixels using a square PSF of pixel size unity
    px,py are the pixel position (integers)
    fmod is the flux calculated for (px,py) pixel
        and it has the same length as px and py
    pixels is the image.
    """

    xmax = pixels.shape[0] #Size of pixel array
    ymax = pixels.shape[1]

    pxmh = px-0.5 #location of reference corner of PSF square
    pymh = py-0.5

    dx = np.floor(px+0.5)-pxmh
    dy = np.floor(py+0.5)-pymh

    # Supposing right-left as x axis and up-down as y axis:
    # Lower left pixel
    npx = int(pxmh) #Numpy arrays start at zero
    npy = int(pymh)

    #print('n',npx,npy)
    
    #if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
    #    pixels[npx,npy]=pixels[npx,npy]+fmod
    
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*dx*dy

    #Same operations are done for the 3 pixels other neighbouring pixels

    # Lower right pixel
    npx = int(pxmh)+1 #Numpy arrays start at zero
    npy = int(pymh)
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*(1.0-dx)*dy

    # Upper left pixel
    npx = int(pxmh) #Numpy arrays start at zero
    npy = int(pymh)+1
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*dx*(1.0-dy)

    # Upper right pixel
    npx = int(pxmh)+1 #Numpy arrays start at zero
    npy = int(pymh)+1
    if (npx >= 0) & (npx < xmax) & (npy >= 0) & (npy < ymax) :
        pixels[npx,npy]=pixels[npx,npy]+fmod*(1.0-dx)*(1.0-dy)
    
    return pixels;

def gauss2d(x,y,sig):
    g=1/(2*np.pi*sig*sig) * np.exp(-(x*x+y*y)/(2*sig*sig))
    return g

def gen_unconv_image(pars,starmodel_flux,xcoo,ycoo):

    xpad=pars.xpad*pars.noversample
    ypad=pars.ypad*pars.noversample
    #array to hold synthetic image
    xmax=pars.xout*pars.noversample+xpad*2
    ymax=pars.yout*pars.noversample+ypad*2

    pixels=np.zeros((xmax,ymax))
    
    i=xcoo*pars.noversample
    j=ycoo*pars.noversample
    
    pixels=addflux2pix(i,j,pixels,starmodel_flux)
    
    return pixels

def readkernels(psf_dir,psf_names):
    pars = ModelPars

    """Reads in PSFs from Matlab file and resamples to match pixel grid of simulation.
    
    Usage: psf=readkernels(psf_dir,psf_names)
    
        Inputs:
            psf_dir - location of PSFs
            psf_names - names of the PSF files to read in.  Order should match 'psf_wv' array
          
        Outputs:
            psf - array of PSFs.
    """
    
    detector_pixscale=18 #detector pixel size (microns)  ***This should be a model parameter***
    # detector_pixscale = 13 # microns/pxl (source: JF)

    psf=[]
    for name in psf_names:

        if psf_names[0][-3:] == 'mat':
            mat_dict=loadmat(psf_dir+psf_names[0]) #read in PSF from Matlab file
            
            psf_native=mat_dict['psf']
            dx_scale=mat_dict['dx'] #scale in micron/pixel of the PSF
            x_scale=int(psf_native.shape[0]*dx_scale/detector_pixscale*pars.noversample) #This gives the PSF size in pixels 
            y_scale=int(psf_native.shape[1]*dx_scale/detector_pixscale*pars.noversample) #This gives the PSF size in pixels 
            #We now resize the PSF from psf.shape to x_scale,yscale
            psf_resize=resize(psf_native,(x_scale,y_scale))
            psf.append(psf_resize)
            
            #plt.imshow(psf_resize,norm=LogNorm())
            #plt.show()
        else:
            hdr, psf_native = [], []
            with open(psf_dir+psf_names[0]) as f:
                for cnt,line in enumerate(f):
                    if cnt < 18:
                        hdr.append( line.strip() )
                        if hdr[-1].startswith('Data spacing'):
                            _l = hdr[-1].split()
                            dx_scale = float(_l[3])
                    else:
                        psf_native.append( line.strip().split() )
            psf_native = np.array(psf_native,dtype=float)

            # x_scale=int(psf_native.shape[0])#*dx_scale/detector_pixscale*pars.noversample) #This gives the PSF size in pixels 
            # y_scale=int(psf_native.shape[1])#*dx_scale/detector_pixscale*pars.noversample) #This gives the PSF size in pixels 

            x_scale=int(psf_native.shape[0]/8*pars.noversample) #This gives the PSF size in pixels 
            y_scale=int(psf_native.shape[1]/8*pars.noversample) #This gives the PSF size in pixels 


            #We now resize the PSF from psf.shape to x_scale,yscale
            psf_resize=resize(psf_native,(x_scale,y_scale))
            psf.append(psf_resize)
        
    return psf



    