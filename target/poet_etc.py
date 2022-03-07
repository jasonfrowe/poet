from ipywidgets import *
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
import statistics
import os
from astropy.io import fits #To read in FITS files and tables
import matplotlib

c = 2.998e8 #speed of light (m/s)
h = 6.626e-34 #Planck m^2 kg/s
Pi= np.pi*1.0 #define Pi
Rsun=695700000.0  # Radius of Sun (m)
zero_point = 25.6884
tput = 0.18 
aperture = 0.15
aperture2 = 2.4
gain = 6.1 #Photons needed for a single electron event 
grid_dir="ck04models/" #base directory for ATLAS models

def changeunits(flux):
    """
    Changes the units of flux from ergs/s/cm^2/nm to watts/m^2/nm. 
    Additional scale factor is used to match GAIA magnitude values.
    """
    try:
        fluxScaled = flux * 1e-7/(0.0001)*(4*np.pi)*6.898196982008417e-19
        return fluxScaled
    except: 
        return 0
    
    
def changeunitsA(flux, wavelength):
    """
    changes the units of flux from ergs/s/cm^2/A to watts/m^s/nm
    """
    newWavelength = wavelength /10
    fluxScaled = flux *1e-7/(0.0001)*10*6.898196982008417e-19
    
    return fluxScaled, newWavelength
    
    
def numericIntegration(x,y):
    """
    Function used to numerically integrate a function.
    
    Inputs:
        x: array of the x values of the curve
        y: array of the y values of the curve
        
    Output:
        total: Value of the numeric integration.
    """
    total = 0  #Add all of the areas to this to get the final integration 
    for i in range(len(x)-1):
        width = x[i+1]-x[i]
        height = y[i] + (y[i+1]-y[i])/2
        area = width*height
        total += area
    return total

##############################################################################################
###  Code to read in the ATLAS star data used for the ETC
##############################################################################################




def getStarData(temperature, metallicity, logG):
    """
    Reads in star data from atlas directory, according to temperature, metallicity, and log(G) values.
    
    Inputs: 
        temperature: Temperature of the reference star in kelvin. Type: float
        metallicity: Metallicity of the reference star. Accepted inputs. 'p/m00','p/m05','p/m10','p/m15','p/m20'
                     p is plus, m is minus.
        logG: log(G) value for the reference star. Type: g00,g05,g10,g15,g20,g25,g30,g35,g40,g45,g50
    
    Outputs: 
        starFlux: flux of the reference star in units watts/m^2/nm. Type: array
        starWavelength: Wavelength of the reference star in nm. Type: array
    """


    mh=metallicity #metallicity 
    teff=temperature #3500 -- 13000 in steps of 250 K are available

    specfile=grid_dir+'ck'+mh+'/'+'ck'+mh+'_'+str(teff)+'.fits'
    if os.path.isfile(specfile):
        havespectrum=True
        hdul = fits.open(specfile)
        data = hdul[1].data #the first extension has table
        wv=data['WAVELENGTH'] #Units are Angstrom
        flux=data[logG] #Units are erg/s/cm^2/A
    else:
        havespectrum=False
        print('Spectrum not found: ',specfile)
        
    #extra plotting function for the star data.
    #if havespectrum:
    #    w1=2000
    #    w2=10000
    #    plt.plot(wv[(wv>w1)&(wv<w2)],flux[(wv>w1)&(wv<w2)])
    #    plt.xlabel('Wavelength (A)')
    #    plt.ylabel('Flux (erg/s/cm^2/A)')
    #    plt.show() 
    
    #change flux units to watts/m^2/nm and wavelength to nm
    starFlux, starWavelength = changeunitsA(flux, wv)
    
    return starFlux, starWavelength
    
   


def Photon_Count(temp, metallicity, logG, Wmin, Wmax, bandpass, bandpassWave, aperture, GAIA_mag, zero_point):
    """
    Generates noise for a planetary transit based on the orbiting star. Various star spectrums are read in 
    then selected based on what star type is selected. 
    
    Inputs:
        temp: Temperature of the reference star. Type: float
        metallicity: Metallicity of the reference star. Allowed inputs: 'p/m00','p/m05','p/m10','p/m15','p/m20'
                     p is plus, m is minus.
        logG: log(G) value for the reference star. Allowed inputs: g00,g05,g10,g15,g20,g25,g30,g35,g40,g45,g50
        Wmin: minumum wavelength of the bandpass. Type: float ##### This may change with changes to bandpass.
        Wmax: maximum wavelength of the bandpass. Type: float
        Bandpass: Bandpass of the filter being used. Type: float (Plan to incorporate arrays soon)
        gain: Number of photons needed for a single electron in the detector: Type: float
        aperture: Aperture size of the instrument. Type: float
        time: Time corresponding to the transit duration. Type: array
        transit: Transit function. This input will likely change in future iterations,
                 Currently used to create a function to apply noise to. Type: array
        GAIA_mag: GAIA magnitude of the desired star. Type: float
        zero_point: zero point for the GAIA magnitude. Type: float

    Outputs: 
        elecSec: The number of electrons read by the detector per second. Type: array
        countsAvg: Average number of photons read by the detector: type: float
        countsStdev: standard deviation of the counts read. Type: float
        noise: Electrons per second with added noise. Type: array
    
    """
    
    starFlux, starWavelength = getStarData(temp, metallicity, logG)
    starWavelength = np.asfarray(starWavelength, float)
    
    
    bot = np.where(starWavelength > (Wmin - 1))
    top = np.where(starWavelength > (Wmax-1))
    
    starFlux = starFlux[bot[0][0]:top[0][0]]
    starWavelength = starWavelength[bot[0][0]:top[0][0]]
    
   
    scale = 10**((GAIA_mag-zero_point)/-2.5) #Scale the flux using GAIA magnitudes, using VEGA as the model.
    scaled_flux = starFlux
    starflux = [i*scale for i in scaled_flux]#new star flux that has been scaled appropriately using G mag
        
    middle = len(starWavelength)//2
    photonCountRate = [] #create an array of the photon count over every wavelength, to be averaged over
    scaledWavelength = [] #convert wavelength from nm to m
    for i in range(len(bandpassWave)):
        scaledWavelength.append(bandpassWave[i]*1e-9)
        
    BPint = numericIntegration(x = scaledWavelength, y = bandpass)
    for i in range(len(starflux)):
        counts = starflux[i]*np.pi*aperture**2*(starWavelength[middle]*1e-9/(h*c))*BPint
        photonCountRate.append(counts)
    
    return photonCountRate



    