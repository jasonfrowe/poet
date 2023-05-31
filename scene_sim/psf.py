import numpy as np
from skimage.transform import resize

#####
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
##########


#####
def gen_unconv_image(pars,starmodel_flux,xcoo,ycoo):

    xpad=pars.xpad*pars.noversample
    ypad=pars.ypad*pars.noversample
    #array to hold synthetic image
    xmax=pars.xout*pars.noversample+xpad*2
    ymax=pars.yout*pars.noversample+ypad*2

    pixels=np.zeros((xmax,ymax))
    
    i = ( xcoo + (pars.xout - pars.ccd_dim[0])/2 ) * pars.noversample
    j = ( ycoo + (pars.yout - pars.ccd_dim[1])/2 ) * pars.noversample
    
    pixels=addflux2pix(i,j,pixels,starmodel_flux)
    
    return pixels
##########


#####
def readkernels(inp_target):
    # Read in POET PSFs

    if inp_target.quiet == False:
        print('Loading PSF...',end='\r')
    psf = []
    for n in range(len(inp_target.psf_name)):
    
        hdr, psf_native = [], []
        with open(inp_target.psf_dir + inp_target.psf_name[n][0]) as f:
            for cnt,line in enumerate(f):
                if cnt < 18:
                    hdr.append( line.strip() )
                    if hdr[-1].startswith('Data area'):
                        _l = hdr[-1].split()
                        data_area = float(_l[3])
                    if hdr[-1].startswith('Pupil grid size'):
                        _l = hdr[-1].split()
                        pupil_grid_size = float(_l[3])
                    if hdr[-1].startswith('Image grid size'):
                        _l = hdr[-1].split()
                        img_grid_size = float(_l[3])
                else:
                    psf_pixscale = data_area / img_grid_size
                    psf_native.append( line.strip().split() )
        psf_native = np.array(psf_native,dtype=float)

        x_scale = int(psf_native.shape[0] \
                        * (psf_pixscale / inp_target.detector_pixscale) \
                        * inp_target.noversample)
        y_scale = int(psf_native.shape[1] \
                        * (psf_pixscale / inp_target.detector_pixscale) \
                        * inp_target.noversample)
        psf_resize = resize(psf_native,(x_scale,y_scale))

        # Normalize
        if inp_target.quiet == False:
            print('psf_native: ', psf_native.shape)
            print('psf_resize: ', psf_resize.shape)
        psf_resize /= np.sum(psf_resize)

        psf.append(psf_resize)

    if inp_target.quiet == False:
        print('Loading PSF... done.')

    return {'name':inp_target.psf_name, 'psf':psf}
##########



