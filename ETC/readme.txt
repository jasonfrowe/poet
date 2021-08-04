Versions 1.0 and 1.5 of the ETC are available for download. Each version has additional required files that are listed in the jupyter notebook for each version. 

Version 1.0:
This version uses the spectrum from VEGA to determine photon count rate for a star using it's GAIA magnitude. Photon count rate is then used to generate
noise in a given band. For this version the band is assumed to be square to ease the integration. Noise generated is applied to a separate transit array.

Version 1.5:
Additional star data has been added using a custom ATLAS-9 directory. This allows the use to define which star spectrum they want to use, by specifying temperature, 
metallicity, and log(G) values for the star. Bandpass data has also been changed to allow any curve to be used. Numeric integration is used on the given bandpass.
