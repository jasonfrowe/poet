import numpy as np
from astropy.io import fits
import os

#####
def const_Rsun():
    return 6.957e8 # [m]
def const_Msun():
    return 1.988409870698051e+30 # [kg]
def const_pc():
    return 3.08567758e+16 # [m]
def const_G():
    return 6.6743e-11 # [m3/kg/s2]
def const_h():
    return 6.62607015e-34 # [J s]
def const_c():
    return 299792458.0 # [m/s]
##########


#####
def getStarData(temperature, metallicity, logg, 
                    model_grid='ATLAS9', # 'ATLAS9', 'BTSettl'
                    ):
    """
    Reads in star data from atlas directory, according to temperature, metallicity, and log(G) values.
    
    Inputs: 
        temperature: Temperature of the reference star in kelvin. Type: float
        metallicity: Metallicity of the reference star. Accepted inputs. 'p/m00','p/m05','p/m10','p/m15','p/m20'
                     p is plus, m is minus.
        logg: log(G) value for the reference star. Type: g00,g05,g10,g15,g20,g25,g30,g35,g40,g45,g50
    
    Outputs: 
        starWavelength: Wavelength of the reference star in nm. Type: array
        starFlux: flux of the reference star in units watts/m^2/nm. Type: array
    """

    # Default to ATLAS9 model
    #     If Teff < 3500 K (i.e., min Teff in ATLAS9 grid, switch to BTSettl grid)
    if ( (model_grid == 'BTSettl') | (temperature < 3500) ):
        grid_dir = 'stellar_models/BTSettl_CIFIST/'

        # Scan directory to inventory model grid
        # Assumes all files starting with 'lte' are model files
        # Only stored solar metallicity models; may need to update for other metallicities
        dir_flist = os.listdir(grid_dir)
        model_flist = []
        grid_param = []
        for n,fname in enumerate(dir_flist):
            if fname.startswith('lte'):
                _teff = float(fname[3:6]) * 100
                if fname[6] == '+':
                    _logg = -1 * float(fname[7:10])
                else:
                    _logg = float(fname[7:10])
                _mh = float(fname[11:14])
                model_flist.append(fname)
                grid_param.append([_teff, _logg, _mh])
        grid_param = np.vstack(grid_param)
        model_flist = np.array(model_flist)

        # Identify closest model prioritizing teff, logg, then metallicity
        mi = np.argmin(np.abs(grid_param[:,0] - temperature))
        ti = np.where(grid_param[:,0] == grid_param[mi,0])[0]
        grid_param = grid_param[ti]
        model_flist = model_flist[ti]

        mi = np.argmin(np.abs(grid_param[:,1] - logg))
        ti = np.where(grid_param[:,1] == grid_param[mi,1])[0]
        grid_param = grid_param[ti]
        model_flist = model_flist[ti]

        mi = np.argmin(np.abs(grid_param[:,2] - metallicity))
        ti = np.where(grid_param[:,2] == grid_param[mi,2])[0]
        grid_param = grid_param[ti]
        model_flist = model_flist[ti]

        # Read in selected model
        d = np.loadtxt(grid_dir + model_flist[0])
        wv = d[:,0] # [Angstrom]
        flux = d[:,1] # [erg/s/cm2/angstrom]

        wv /= 10 # [Angstrom] -> [nm]
        flux *= 0.01 # [erg/s/cm2/angstrom] -> [W/m2/nm]

    elif model_grid == 'ATLAS9':
        grid_dir = 'stellar_models/ATLAS9/ck04models/'
        teff_grid = np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 
                              6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 
                              8500, 8750, 9000, 9250, 9500, 9750, 10000, 10250, 10500, 10750, 
                              11000, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 14000, 
                              15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 
                              25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 
                              35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 
                              45000, 46000, 47000, 48000, 49000, 50000])
        mh_grid = np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.02, 0.05])
        logg_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

        mi = [ np.argmin( np.abs(teff_grid - temperature) ),
               np.argmin( np.abs(mh_grid - metallicity) ),
               np.argmin( np.abs(logg_grid - logg) ) ]

        teff_str = '{:.0f}'.format( teff_grid[mi[0]] )
        if mh_grid[mi[1]] < 0:
            mh_str = 'm{:0>2d}'.format( int(np.abs( mh_grid[mi[1]] ) * 100) )
        else:
            mh_str = 'p{:0>2d}'.format( int(np.abs( mh_grid[mi[1]] ) * 100) )
        logg_str = 'g{:0>2d}'.format( int( logg_grid[mi[2]] * 10) )

        specfile = grid_dir + 'ck' + mh_str + '/' + 'ck' + mh_str + '_' + teff_str + '.fits'
        if os.path.isfile(specfile):
            havespectrum = True
            hdul = fits.open(specfile)
            data = hdul[1].data #the first extension has table
            wv = data['WAVELENGTH'] #Units are Angstrom
            flux = data[logg_str] #Units are erg/s/cm^2/A
        else:
            havespectrum = False
            print('Spectrum not found: ',specfile)

        wv /= 10 # [Angstrom] -> [nm]
        flux *= 0.01 # [erg/s/cm2/angstrom] -> [W/m2/nm]

    return wv, flux
##########





#####
def Photon_Count(temp=5780., metallicity=0.0, logg=4.44, 
                    Gmag=7.0, Gmag_abs=4.635,
                    radius=1.0, 
                    dpc=1.09, # Not currently used. Included for testing
                    aperture=None,
                    stellar_model_grid='ATLAS9',
                    BPwl=None, BPtr=None):
    # Calculate photon count from model for given transmission function.
    #     (1) Models obtained from getStarData yield stellar surface flux [W/m2/nm].
    #         Models are selected using estimated Teff, [M/H], and log(g)
    #     (2) Models are then multiplied by (R/d)^2 to get observed model flux
    #         as a function of wavelength
    #     (3) Integrate model flux with transmission function and aperture
    #         to calculate photons/sec/wavelength at detector
    #     (4) Return summed count rate [photons/sec]
    #
    # Tested by comparing with photon count reported in Table 6 of Walker+2003:
    #     - Use temp=5780., metallicity=0.0, logg=4.44, radius=1.0, and dpc=1.09
    #       corresponding to Solar-type star with V=0.
    #     - Use BPtr=0.2 and BPwl=np.arange(330,700,0.25), aperture=0.15 (~MOST)
    #     - Yields 1.080e8 photons/sec (Walker+2003 calculate 1.623e8)

    wl, fl = getStarData(temp, metallicity, logg, model_grid=stellar_model_grid)

    # Estimate distance using Gmag and estimated Gabs
    dpc = 10**( (Gmag - Gmag_abs)/5. + 1 )

    # Scale surface flux to observed flux using esimated dpc and radius
    fl *= ( ( radius * const_Rsun() ) / ( dpc * const_pc() ) )**2

    # Calculate Gaia bandpass fluxes
    tr_fcn = np.loadtxt('instrument_data/transmission_functions/GaiaDR2_Passbands.dat')
    gaia_flux = []
    for j in [1,3,5]: # Select columns listing transmission values, not the errors
        ti = np.where(tr_fcn[:,j] < 99)[0]
        _int_tr = np.interp(wl, tr_fcn[ti,0], tr_fcn[ti,j], left=0, right=0)
        gaia_flux.append( np.trapz(_int_tr * fl, wl) )
    
    # Interpolate POET bandpass (including QE) onto model grid
    ti = np.where( (wl >= BPwl[0]) & (wl <= BPwl[-1]) )[0]
    wl, fl = wl[ti], fl[ti]
    BPtr_int = np.interp(wl, BPwl, BPtr, left=0.0, right=0.0)

    # Calculate POET flux
    flux = np.trapz( fl * BPtr_int, wl )

    # Calculate photon count rate (partially based on Gaia ZP Eqn. 2 of Evans+2018)
    _y = fl * np.pi * (aperture/2)**2 \
                * (wl * 1.e-9 / (const_h() * const_c()) ) \
                * BPtr_int
    cnt_rate = np.trapz( _y, wl ) # [counts/sec/wavelength]

    if np.isfinite(cnt_rate) == False:
        cnt_rate = 1.

    return cnt_rate, flux, gaia_flux
##########




