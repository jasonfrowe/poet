import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

from tqdm import tqdm

from scipy.signal import fftconvolve
from scipy.ndimage import interpolation
from skimage.transform import downscale_local_mean

import pickle

import os

from . import util
from scene_sim import calc_flux
from scene_sim import psf as poet_psf

class Observation(object):
    #####
    # Initialize Observation object
    # Minimum required input parameters:
    #     (1) ra, dec
    #     (2) gdr2_id
    # Gaia catalog query will run unless the following parameters are specified:
    #     (1) ra, dec, Gmag, source_model; bkg_sources==False
    #         source_model specifies the stellar model to be adopted while Gmag is used
    #         to appropriately scale the flux.
    #     (2) ra, dec, Gmag, Bpmag, Rpmag; bkg_sources==False
    #         Bpmag-Rpmag is used to estimate the source_model based on the 
    #         EEM_dwarf_UBVIJHK_colors_Teff.txt table; Gmag is then used to scale the flux.
    # Background sources in scene cannot be specified. Therefore, if bkg_sources==True,
    # the Gaia catalog query is run.
    # ETC requires a source model, which can be estimated based on the specified Bpmag, Rpmag
    def __init__(self,
                    prefix='poet_obs', # Prefix used for filenames
                    t_id='', # Target name. Used for plots
                    ra=None,dec=None, # Target coordinates [deg]
                    gdr2_id=None, # Not implemented...
                    Gmag=None, 
                    Bpmag=None, 
                    Rpmag=None, 
                    Teff=None, 
                    logg=None, 
                    metallicity=None, 
                    radius=None, 
                    stellar_model_grid='ATLAS9', # Default grid to use: 'ATLAS9', 'BTSettl'
                    pl_model={}, # Model planet used for transit sim
                    srch_Gmax=21., # Max Gaia G magnitude for Gaia catalog query (applies to both the target and the guide stars)
                    srch_rad=None, # Search radius [deg] for Gaia catalog query (applies to both the target and the guide stars)
                    srch_nmax=100, # Max Gaia sources to include
                    bkg_sources=True, # True => Search Gaia catalog for background stars
                    gs_criteria={'SN_pix':10, 'exptime':0.5, 'ccd_aperture':4}, # Criteria for guide star
                    exptime=60, #exposure time [s]
                    nstack=10, # number of exposures to stack (t_tot = exptime * nstack)
                    tstart=0.0, # light curve start time [days]
                    tend=6.0/24., # light curve end time [days]
                    save=True,quiet=False,

                    ccd_dim=[1024,1024], # [pxl,pxl] used for plotting (including converting ra,dec -> x_pxl,y_pxl)
                    fov=1.0, # Full-width FoV [deg] used for converting ra,dec -> x_pxl, y_pxl
                    fov_pa=0., # FoV position angle [deg]
                    noise_sources=['illum', 'shot', 'read', 'dark', 'jitter'], # List of noise sources to include
                    psf_dir='instrument_data/', # PSF directory
                    psf_name=[ ['POET_PSF_CenterFoV_VNIR_64pxBox.txt', (0.5, 0.5)] ], # List of PSFs to be read;
                                                                                      # [ [Filename, (x_frac, y_frac)] ]
                    # gain=6.1, # electronic gain [e-/adu]
                    gain=3.,
                    saturation=65536.0, #saturation
                    readnoise=8.0, #readnoise electrons
                    darknoise=0.4, #dark current (e-/px/sec)
                    xjit=0.0, # standard deviation of jitter sampling [pxl]
                    yjit=0.0, # standard deviation of jitter sampling [pxl]
                    xpad=10, #padding to deal with convolution fall-off
                    ypad=10, #padding to deal with convolution fall-off
                    xout=1024, #x-axis
                    yout=1024, #y-axis
                    noversample=2, # CCD oversampling factor
                    detector_aperture=0.15, # instrument aperture diameter [m] used for ETC photon count
                    detector_pixscale=13., # pixel size [microns]
                    illumfname='instrument_data/illum_pattern.csv', # CCD illuination pattern file
                    ccd_aperture=8, # radius [pxl] for aperture photometry
                    bandpass_id='VNIR', # 'VNIR', 'SWIR', or None
                                        # 'VNIR' or 'SWIR' => set psf_name, BPwl, BPtr below
                                        # None => use default/user-specified psf_name, BPwl, BPtr
                    BPtr=0.2, # float => square bandpass; if array, dimensions must match BPwl
                    BPwl=np.arange(400,900,0.25) # [nm] # ~VNIR
                    # BPwl=np.arange(1000,1700,0.25) # [nm] # ~SWIR
                    # BPwl=np.arange(330,700,0.25) # [nm] # ~MOST
                    ):

        self.prefix = prefix

        self.t_id = t_id
        self.ra = ra
        self.dec = dec
        self.gdr2_id = gdr2_id

        if ( (ra == None) & (dec == None) & (gdr2_id == None) ) \
                | ( (gdr2_id == None) & ( (ra == None) | (dec == None) ) ):
            raise Exception('Specify (ra, dec) [deg] or gdr2_id.')

        self.Gmag = Gmag
        if len(pl_model) > 0:
            self.pl_model = pl_model

        if srch_rad == None:
            srch_rad = (fov / 2.) * np.sqrt(2.)
        self.srch_rad = srch_rad
        self.srch_nmax = srch_nmax

        self.srch_Gmax = srch_Gmax
        if gs_criteria != None:
            self.gs_criteria = gs_criteria

        self.exptime = exptime
        self.nstack = nstack

        self.tstart = tstart
        self.tend = tend

        self.ccd_dim = ccd_dim
        self.fov = fov
        self.fov_pa = fov_pa
        self.noise_sources = noise_sources

        # Set bandpass-specific parameters
        self.psf_dir = psf_dir
        self.bandpass_id = bandpass_id
        if bandpass_id == 'VNIR':
            self.psf_name = [ ['POET_PSF_CenterFoV_VNIR_64pxBox.txt', (0.5, 0.5)] ]
            self.BPtr = 0.2
            self.BPwl = np.arange(400,900,0.25) # [nm]
        elif bandpass_id == 'SWIR':
            self.psf_name = [ ['POET_PSF_CenterFoV_SWIR_64pxBox.txt', (0.5, 0.5)] ]
            self.BPtr = 0.2
            self.BPwl = np.arange(1000,1700,0.25) # [nm]
        else:
            self.psf_name = psf_name
            self.BPtr = BPtr
            self.BPwl = BPwl

        self.gain = gain
        self.saturation = saturation
        self.readnoise = readnoise
        self.darknoise = darknoise
        self.xjit = xjit
        self.yjit = yjit
        self.xpad = xpad
        self.ypad = ypad
        self.xout = xout
        self.yout = yout
        self.noversample = noversample

        self.detector_aperture = detector_aperture
        self.detector_pixscale = detector_pixscale
        self.illumfile = illumfname

        self.ccd_aperture = ccd_aperture

        self.save = save
        self.quiet = quiet

        # Determine whether to search Gaia catalog for target and/or guide stars
        run_gaia_search = True
        if bkg_sources == False:
            self.srch_nmax = 1
        if (Gmag != None) & (Teff != None):
            run_gaia_search = False

        self.stellar_model_grid = stellar_model_grid

        # Search Gaia catalog for target and/or guide stars
        if run_gaia_search:
            self.search_gaia(srch_Gmax=srch_Gmax, srch_rad=srch_rad, nmax=srch_nmax)
        else:
            # Manually create gaia source list
            self.gaia = {'ra':np.array([ra]), 'dec':np.array([dec]),
                         'x':np.zeros(1) + self.xout/2, 'y':np.zeros(1) + self.yout/2,
                         'Gmag':np.array([np.nan]), 
                         'Bpmag':np.array([np.nan]),
                         'Rpmag':np.array([np.nan]),
                         'Gmag_abs':np.array([np.nan]),
                         'Teff':np.array([np.nan]),
                         'logg':np.array([np.nan]),
                         'radius':np.array([np.nan]),
                         'metallicity':np.array([np.nan]),
                         'results':None, # Full results from Gaia query (save for now just in case it saves time later)
                         'wcs':None # wcs.world_to_pixel object used for projection transformation
                        }

        # Replace target parameters with user-specified values
        if Gmag != None:
            self.gaia['Gmag'][0] = Gmag
        if Bpmag != None:
            self.gaia['Bpmag'][0] = Bpmag
        if Rpmag != None:
            self.gaia['Rpmag'][0] = Rpmag
        if Teff != None:
            self.gaia['Teff'][0] = Teff
        if logg != None:
            self.gaia['logg'][0] = logg
        if radius != None:
            self.gaia['radius'][0] = radius
        if metallicity != None:
            self.gaia['metallicity'][0] = metallicity

        # Interpolate EEM_table using Teff or (Bpmag, Rpmag)
        if 'Teff' in self.gaia.keys():
            interp_eem = util.interp_EEM_table(Teff=self.gaia['Teff'], Gmag=self.gaia['Gmag'])
        else:
            interp_eem = util.interp_EEM_table(Gmag=self.gaia['Gmag'], 
                                Bpmag=self.gaia['Bpmag'], Rpmag=self.gaia['Rpmag'])

        # For run_gaia_search, need to add in Teff, logg, radius, metallicity using interpolate EEM_table values
        if run_gaia_search:
            self.gaia['Teff'] = interp_eem['Teff']
            self.gaia['logg'] = interp_eem['logg']
            self.gaia['radius'] = interp_eem['radius']
            self.gaia['metallicity'] = np.zeros_like(self.gaia['Teff'])
            self.gaia['Gmag_abs'] = interp_eem['Gmag_abs']

        # If certain values are missing, replace with interpolated EEM_table values
        for _lbl in ['Bpmag', 'Rpmag', 'logg', 'radius', 'Gmag_abs']:
            ti = np.isnan(self.gaia[_lbl])
            self.gaia[_lbl][ti] = interp_eem[_lbl][ti]

        # Replace missing metallicities with solar value
        ti = np.isnan(self.gaia['metallicity'])
        self.gaia['metallicity'][ti] = 0.

        if self.save:
            self._save()
        ##########


    #####
    def _save(self):
        # Save Observation object
        pickle.dump(self,open(self.prefix+'.pkl','wb'))
    ##########


    #####
    def search_gaia(self, srch_Gmax=None, srch_rad=None, nmax=100):
        # Search Gaia catalog for target and background stars
        # To do: add search using Gaia ID

        if srch_Gmax != None:
            self.srch_Gmax = srch_Gmax
        if srch_rad != None:
            self.srch_rad = srch_rad

        try:
            from astroquery.gaia import Gaia
            srch_str = "SELECT *, DISTANCE(POINT({:.6f},{:.6f}), POINT(ra, dec)) ".format(self.ra,self.dec) \
                        + "AS ang_sep FROM gaiadr2.gaia_source " \
                        + "WHERE 1 = CONTAINS(   POINT({:.6f},{:.6f}),   ".format(self.ra,self.dec) \
                        + "CIRCLE(ra, dec, {:.2f})) ".format(self.srch_rad) \
                        + "AND phot_g_mean_mag <={:.2f} ".format(self.srch_Gmax) \
                        + "AND parallax IS NOT NULL ORDER BY ang_sep ASC"
            job = Gaia.launch_job(srch_str)
            results = job.get_results()

            ra, dec, Gmag, Bpmag, Rpmag = \
                        np.array(results['ra']), np.array(results['dec']), \
                        np.array(results['phot_g_mean_mag']), \
                        np.array(results['phot_bp_mean_mag']), \
                        np.array(results['phot_rp_mean_mag'])

            if self.quiet == False:
                print('{:.0f} Gaia source(s) found.'.format(len(ra)))

            # Identify target within Gaia search (brightest target within sep_max)
            sep_max = 8.0 / 3600. # [deg]
            c_coord = SkyCoord(ra=self.ra, dec=self.dec, unit=(u.degree, u.degree), frame='icrs')
            g_coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
            sep = g_coord.separation(c_coord).degree
            ti = np.where(sep < sep_max)[0]
            if len(ti) > 0:
                ti = ti[ np.argmin(Gmag[ti]) ]
            else:
                ti = np.argmin(sep)
                if self.quiet == False:
                    print('No Gaia sources found within {:.2f} arcsec of specified RA and DEC.'.format(sep_max*3600.))
                    print('     Asigning nearest source to target.')
            gi = np.delete(np.arange(len(ra)),ti) # Indices for non-target sources

            # Place target at top of list, sort remaining sources by brightness
            si = np.argsort(Gmag[gi])
            ra = np.hstack([ ra[ti], ra[gi[si]] ])
            dec = np.hstack([ dec[ti], dec[gi[si]] ])
            Gmag = np.hstack([ Gmag[ti], Gmag[gi[si]] ])
            Bpmag = np.hstack([ Bpmag[ti], Bpmag[gi[si]] ])
            Rpmag = np.hstack([ Rpmag[ti], Rpmag[gi[si]] ])
            sep = np.hstack([ sep[ti], sep[gi[si]] ])

            # Trim dimmest sources
            if len(ra) > nmax:
                ra = ra[:nmax]
                dec = dec[:nmax]
                Gmag = Gmag[:nmax]
                Bpmag = Bpmag[:nmax]
                Rpmag = Rpmag[:nmax]
                sep = sep[:nmax]

            self.gaia = {'ra':ra, 'dec':dec, 
                         'x':np.zeros_like(ra), 'y':np.zeros_like(ra),
                         'Gmag':Gmag, 'Bpmag':Bpmag, 'Rpmag':Rpmag,
                         'results':None, 'wcs':None}

            self.calc_xy() # Get CCD (x,y) positions for all gaia sources

            if self.save:
                self._save()
        except Exception:
            if self.quiet == False:
                print('Gaia search failed.')
    ##########


    #####
    def calc_xy(self,fov_pa=None):
        # Should add (x,y) for oversampled array so that source positions
        # can be centered on intra-pixel positions

        if fov_pa != None:
            self.fov_pa = fov_pa

        # Get projected CCD coordinates (centered on target)
        wcs_input_dict = {
            'CTYPE1': 'RA---TAN', 
            'CUNIT1': 'deg', 
            'CDELT1': self.fov/self.ccd_dim[0], 
            'CRPIX1': int(self.ccd_dim[0]/2), 
            'CRVAL1': self.gaia['ra'][0], 
            'NAXIS1': self.ccd_dim[0],
            'CTYPE2': 'DEC--TAN', 
            'CUNIT2': 'deg', 
            'CDELT2': self.fov/self.ccd_dim[1], 
            'CRPIX2': int(self.ccd_dim[1]/2), 
            'CRVAL2': self.gaia['dec'][0], 
            'NAXIS2': self.ccd_dim[1],
            'CROTA2': self.fov_pa,
        }
        wcs = WCS(wcs_input_dict)

        # Convert Gaia source ra,dec -> x,y
        g_coord = SkyCoord(ra=self.gaia['ra'], dec=self.gaia['dec'], 
                            unit=(u.degree, u.degree), frame='icrs')
        _x, _y = wcs.world_to_pixel(g_coord)

        self.gaia['x'], self.gaia['y'] = _x, _y
        self.gaia['wcs'] = wcs
    ##########



    #####
    def plot_fov(self,pa=None,save_plot=True,
                    plot_guide_stars=True,
                    vmin=None,vmax=None,
                    plot_grid=True,add_scene_sim=True):

        if hasattr(self,'gaia') == False:
            self.search_gaia()

        if pa != None:
            self.fov_pa = pa

        plt.close(1)
        fig = plt.figure(num=1,figsize=(5.5,5),facecolor='w')
        ax = fig.add_subplot(111)

        ax_pos = ax.get_position()
        ax_pos.x0 = 0.175
        ax_pos.x1 = 0.99
        ax_pos.y0 = 0.108
        ax_pos.y1 = 0.99
        ax.set_position(ax_pos)

        Gmag_lim = [np.min(self.gaia['Gmag']), np.max(self.gaia['Gmag'])]
        sym_lim = [0.5, 100]
        if Gmag_lim[1] == Gmag_lim[0]:
            sym_size = np.zeros_like(self.gaia['Gmag']) + 5
        else:
            sym_size = ((Gmag_lim[1] - self.gaia['Gmag']) / (Gmag_lim[1] - Gmag_lim[0] )) \
                            * (sym_lim[1] - sym_lim[0]) + sym_lim[0]

        # Plot target
        ax.scatter(self.gaia['x'][0],self.gaia['y'][0],edgecolor='r',
                        marker='o',s=80,facecolor='None')

        # Plot FoV boundary
        _x = [0, self.ccd_dim[0], self.ccd_dim[0], 0, 0]
        _y = [0, 0, self.ccd_dim[1], self.ccd_dim[1], 0]
        ax.plot(_x,_y,c='r')

        if plot_guide_stars:
            if ( ('gs_i' in self.gaia.keys()) == False ) & hasattr(self,'gs_criteria'):
                self.id_guide_stars()
            if 'gs_i' in self.gaia.keys():
                ax.scatter(self.gaia['x'][ self.gaia['gs_i'] ],
                           self.gaia['y'][ self.gaia['gs_i'] ],
                           marker='s',zorder=1,facecolor='None',edgecolor='r',lw=0.7)

        ax.set_xlabel(r'x [pxl]',fontsize=16)
        ax.set_ylabel(r'y [pxl]',fontsize=16)

        xlim = int(self.ccd_dim[0]/2) + self.xout * 0.7 * np.array([-1.0,1.0])
        ylim = int(self.ccd_dim[1]/2) + self.yout * 0.7 * np.array([-1.0,1.0])

        if add_scene_sim:
            from matplotlib.colors import LogNorm

            if ('scene' in self.gaia.keys()) == False:
                self.scene_sim()

            extent = np.array([ int(self.ccd_dim[0]/2) - self.xout/2,
                                int(self.ccd_dim[0]/2) + self.xout/2,
                                int(self.ccd_dim[1]/2) - self.yout/2 - 2,
                                int(self.ccd_dim[1]/2) + self.yout/2 - 2])

            cmap = plt.get_cmap('cividis')
            _f = self.gaia['scene']-np.min(self.gaia['scene'])+1
            _f = np.flip(_f,axis=0) # Go through code to figure this out...
            ax.imshow(_f,
                    norm=LogNorm(vmin=vmin,vmax=vmax),
                    extent=extent,
                    interpolation=None,cmap=cmap,zorder=-100)

        if plot_grid:
            # Plot grid lines
            grid_ra = np.linspace( np.floor(np.min(self.gaia['ra']) - 0.5), 
                                            np.ceil(np.max(self.gaia['ra'] + 0.5)), 1000 )
            grid_dec = np.linspace( np.floor(np.min(self.gaia['dec']) - 1.), 
                                            np.ceil(np.max(self.gaia['dec']) + 0.5), 1000 )

            ra_lim = [np.floor(grid_ra[0]), np.ceil(grid_ra[-1])]
            for del_ra in [0.5, 1.0, 5.0, 10.]:
                if (ra_lim[1] - ra_lim[0])/del_ra < 10:
                    break

            dec_lim = [np.floor(grid_dec[0]), np.ceil(grid_dec[-1])]
            for del_dec in [0.5, 1.0, 5.0, 10.]:
                if (dec_lim[1] - dec_lim[0])/del_dec < 10:
                    break

            ra_fact = 1.
            if (ra_lim[1] - ra_lim[0])/del_ra > 8:
                ra_fact = 2.
            dec_fact = 1.
            if (dec_lim[1] - dec_lim[0])/del_dec > 8:
                dec_fact = 2.

            # Extend RA grid
            ax_xy = np.array([ [xlim[0], ylim[0]], [xlim[1], ylim[0]], \
                               [xlim[1], ylim[1]], [xlim[0], ylim[1]] ])
            ax_radec = self.gaia['wcs'].pixel_to_world(ax_xy[:,0],ax_xy[:,1])
            if np.floor(np.min(ax_radec.ra.value)) < (grid_ra[0] - del_ra):
                grid_ra = np.hstack([ np.arange( np.floor(np.min(ax_radec.ra.value)), 
                                                    grid_ra[0], del_ra ), grid_ra ])
                ra_lim[0] = grid_ra[0]
            if np.ceil(np.max(ax_radec.ra.value)) < (grid_ra[0] + del_ra):
                grid_ra = np.hstack([ grid_ra, np.arange( grid_ra[-1], 
                                                    np.ceil(np.max(ax_radec.ra.value)), 
                                                        del_ra ) ])
                ra_lim[1] = grid_ra[-1]

            if grid_ra[-1] < (ra_lim[1] + del_ra):
                grid_ra = np.hstack([ grid_ra, np.arange(grid_ra[-1],ra_lim[1]+3*del_ra) ])
            if grid_dec[-1] < (dec_lim[1] + del_dec):
                grid_dec = np.hstack([ grid_dec, np.arange(grid_dec[-1],dec_lim[1]+del_dec) ])

            for _g_ra in np.arange( ra_lim[0], ra_lim[1] + 2*del_ra, del_ra ):
                grid_coord = SkyCoord(ra=_g_ra+np.zeros(len(grid_dec)), dec=grid_dec, 
                                 unit=(u.degree, u.degree), frame='icrs')
                grid_x, grid_y = self.gaia['wcs'].world_to_pixel(grid_coord)
                ax.plot(grid_x,grid_y,c='k',alpha=0.4,lw=1,zorder=-1)

            for _g_dec in np.arange( dec_lim[0], dec_lim[1] + del_dec, del_dec ):
                grid_coord = SkyCoord(ra=grid_ra, dec=_g_dec+np.zeros(len(grid_ra)), 
                                 unit=(u.degree, u.degree), frame='icrs')
                grid_x, grid_y = self.gaia['wcs'].world_to_pixel(grid_coord)
                ax.plot(grid_x,grid_y,c='k',alpha=0.4,lw=1,zorder=-1)

            for _g_ra in np.arange( ra_lim[0], ra_lim[1] + del_ra, ra_fact*del_ra ):
                for _g_dec in np.arange( dec_lim[0], dec_lim[1] + del_dec, dec_fact*del_dec ):
                    grid_coord = SkyCoord(ra=_g_ra, dec=_g_dec, 
                                     unit=(u.degree, u.degree), frame='icrs')
                    grid_x, grid_y = self.gaia['wcs'].world_to_pixel(grid_coord)

                    if (grid_x > xlim[0]) & (grid_x < (xlim[1] - 0.1*(xlim[1] - xlim[0]))) \
                            & (grid_y > (ylim[0] + 0.05*(ylim[1] - ylim[0]))) \
                            & (grid_y < ylim[1]):
                        grid_coord = SkyCoord(ra=_g_ra+del_ra, dec=_g_dec, 
                                         unit=(u.degree, u.degree), frame='icrs')
                        grid_x2, grid_y2 = self.gaia['wcs'].world_to_pixel(grid_coord)
                        coord_str = '('
                        if del_ra < 1:
                            coord_str += '{:.1f}'.format(_g_ra) + ','
                        else:
                            coord_str += '{:.0f}'.format(_g_ra) + ','

                        if del_dec < 1:
                            coord_str += '{:.1f}'.format(_g_dec) + ')'
                        else:
                            coord_str += '{:.0f}'.format(_g_dec) + ')'
                        ax.text(grid_x + 0.007*(xlim[1] - xlim[0]),
                                grid_y + 0.00*(ylim[1] - ylim[0]),
                                    coord_str,fontsize=6,alpha=0.6,zorder=-1,
                                    va='bottom',ha='left',rotation_mode='anchor',
                                    rotation=(180./np.pi) \
                                                * np.arctan( (grid_y2 - grid_y) \
                                                                / (grid_x2 - grid_x) ) )

        ax.annotate(self.t_id.replace('_',' '),(0.035,0.93),xycoords='axes fraction',
                    fontsize=16,zorder=20)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both',labelsize=16)
        ax.tick_params(axis='both',which='major',length=6)
        ax.tick_params(axis='both',which='minor',length=3)
        ax.tick_params(axis='x',which='both',direction='inout')
        ax.tick_params(axis='x',which='both',top=True,direction='in')
        ax.tick_params(axis='y',which='both',direction='inout')
        ax.tick_params(axis='y',which='both',right=True,direction='in')

        plt.ion()
        plt.show()
        plt.pause(1.e-6)
        if save_plot:
            fig.savefig(self.prefix + '_fov.png',format='png',dpi=200)
    ##########



    #####
    def id_guide_stars(self,gs_criteria=None,plot_SN=False):
        # gs_criteria = {'SN_pix':10, 'exptime':0.5, 'ccd_aperture':4}
        #     SN_pix: minimum S/N per pixel associated with each guide star
        #     exptime: assume 2 Hz for guide camera
        #     ccd_aperture: radius [pxl] to evaluate max S/N associated with each source
        #     
        if gs_criteria != None:
            self.gs_criteria = gs_criteria

        # Generate scene simulation using guide star exposure criteria
        _nstack, _exptime, _quiet = self.nstack, self.exptime, self.quiet # Store previous nstack and exptime
        self.nstack, self.exptime, self.quiet = 1, self.gs_criteria['exptime'], True
        SN_pix, _SN_ap = self.scene_sim(return_SN_only=True) # Get guide star S/N per pixel

        # Restore nstack, exptime settings
        self.nstack, self.exptime, self.quiet = _nstack, _exptime, _quiet

        if len(self.gaia['ra'] > 1):
            gs_i = np.arange(1,len(self.gaia['ra']))

            _ti = np.where( (self.gaia['x'][gs_i] > self.gs_criteria['ccd_aperture']) \
                            & (self.gaia['x'][gs_i] < (self.ccd_dim[0] - self.gs_criteria['ccd_aperture']) ) \
                            & (self.gaia['y'][gs_i] > self.gs_criteria['ccd_aperture']) \
                            & (self.gaia['y'][gs_i] < (self.ccd_dim[1] - self.gs_criteria['ccd_aperture']) ) )[0]
            gs_i = gs_i[_ti]
            
            # Get maximum S/N associated with each potential guide star
            _xx, _yy = np.meshgrid( np.arange(self.ccd_dim[0]), np.arange(self.ccd_dim[1]) )
            _xx, _yy = _xx.T, _yy.T
            SN_pix_max = np.zeros(len(gs_i))
            SN_ap = np.zeros(len(gs_i))
            for i in range(len(gs_i)):
                ti = np.sqrt( (_xx - self.gaia['x'][gs_i[i]])**2 \
                            + (_yy - self.gaia['y'][gs_i[i]])**2 ) < self.gs_criteria['ccd_aperture']
                SN_pix_max[i] = np.max( SN_pix[ti] )
                SN_ap[i] = _SN_ap[gs_i[i]]
            ti = np.where( SN_pix_max > self.gs_criteria['SN_pix'] )[0]
            gs_i = gs_i[ ti ]
            SN_pix_max = SN_pix_max[ ti ]
            SN_ap = SN_ap[ ti ]

            if len(gs_i) > 0:
                # Remove guide stars that are too close together (retaining the brightest one)
                ki = []
                for i in range(len(gs_i)):
                    if (i in ki) == False:
                        _sep = np.sqrt( (self.gaia['x'][gs_i] - self.gaia['x'][gs_i[i]])**2 \
                                      + (self.gaia['y'][gs_i] - self.gaia['y'][gs_i[i]])**2 )
                        ti = np.where( _sep < self.gs_criteria['ccd_aperture'] )[0]
                        ki.append( ti[ np.argmin(self.gaia['Gmag'][gs_i[ti]]) ] )
                ki = np.hstack(ki)
                gs_i = gs_i[ki]
                SN_pix_max = SN_pix_max[ki]
                SN_ap = SN_ap[ki]
            else:
                self.gaia['gs_i'] = []
                self.gaia['gs_SN_pix_max'] = []
        else:
            self.gaia['gs_i'] = []
            self.gaia['gs_SN_pix_max'] = []

        if self.quiet == False:
            print('{:.0f} guide star(s) identified.'.format(len(gs_i)))

        self.gaia['gs_i'] = gs_i
        self.gaia['gs_SN_pix_max'] = SN_pix_max

        if plot_SN:
            fig = plt.figure(num=2)
            plt.clf()
            ax = fig.add_subplot(111)

            ax.scatter(self.gaia['Gmag'][self.gaia['gs_i']],self.gaia['gs_SN_pix_max'])

            ax.set_xlabel('Gaia G [mag]',fontsize=16)
            ax.set_ylabel('Max. S/N per pixel',fontsize=16)

            ylim = np.array(ax.get_ylim())
            ax.set_ylim(ylim)

            xlim = np.array(ax.get_xlim())
            ax.set_xlim(np.flip(xlim))

            from matplotlib.ticker import AutoMinorLocator
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both',labelsize=16)
            ax.tick_params(axis='both',which='major',length=6)
            ax.tick_params(axis='both',which='minor',length=3)
            ax.tick_params(axis='x',which='both',direction='inout')
            ax.tick_params(axis='x',which='both',top=True,direction='in')
            ax.tick_params(axis='y',which='both',direction='inout')
            ax.tick_params(axis='y',which='both',right=True,direction='in')

            fig.tight_layout()
    ##########



    #####
    def scene_sim(self,
                    all_sources=True, # True => include all Gaia sources in scene
                                      # False => include only the target
                    BPwl=None, # Bandpass wavelength [nm]; must be array
                    BPtr=None, # Bandpass transmittance
                    reload_kernel=False, # True => reload all PSFs in self.psf_name
                                         # False => only load PSFs if self.psf not found
                    return_scene=False, # True => scene_flux = self.scene_sim()
                    target_flux_fraction=1.0, # Used to reduce target star flux when injecting transit signal
                    update_gaia=True,
                    quiet=None,
                    return_SN_only=False, # Return S/N without updating object (used for guide star identification)
                    ):
        # Generate simulated scene using sources listed in self.gaia
        # See poet_etc.Photon_Count for photon count rate calculation explanation

        if all_sources:
            nsrc = len(self.gaia['ra'])
        else:
            nsrc = 1

        if quiet != None:
            self.quiet = quiet

        ##### Calculate flux using ETC for each Gaia source
        # Define bandpass
        if BPwl == None:
            BPwl = self.BPwl
        if BPtr == None:
            BPtr = self.BPtr
        if type(BPtr) == float:
            BPtr = np.zeros_like(BPwl) + BPtr

        # Calculate photon count for each source
        scene_phot_count = np.zeros(nsrc)
        scene_flux = np.zeros(nsrc)
        gaia_flux = np.zeros( (nsrc, 3) ) # [G, Bp, Rp] flux
        for n in range(nsrc):
            scene_phot_count[n], scene_flux[n], gaia_flux[n] = \
                                    calc_flux.Photon_Count(
                                        Gmag=self.gaia['Gmag'][n],
                                        Gmag_abs=self.gaia['Gmag_abs'][n],
                                        aperture=self.detector_aperture,
                                        temp=self.gaia['Teff'][n], 
                                        radius=self.gaia['radius'][n],
                                        metallicity=0.0, # Currently assumes solar
                                        logg=self.gaia['logg'][n],
                                        stellar_model_grid=self.stellar_model_grid,
                                        BPwl=BPwl, BPtr=BPtr)
            scene_phot_count[n] *= self.exptime

            # Reduce target star flux due to transit (target_flux_fraction=1 by default; i.e., no transit)
            if n == 0:
                scene_phot_count[n] *= target_flux_fraction
        starmodel_flux = np.copy(scene_phot_count) # [counts]
        ##########

        ##### Load PSF kernel
        # To do: implement interpolation of multiple PSFs across CCD
        #     Load PSFs generated for several points across PSF.
        #     Generate interpolated PSF for each source
        #     Convolve each source (probably slow)
        if reload_kernel | (os.path.exists(self.psf_dir + 'psf.pkl') == False):
            # Force reload of psf
            psf = poet_psf.readkernels(self)
            pickle.dump(psf,open(self.psf_dir + 'psf.pkl','wb'))
        elif os.path.exists(self.psf_dir + 'psf.pkl'):
            # Check if stored psf (if exists) matches user-specified psf
            # Reload psf if necessary
            psf = pickle.load(open(self.psf_dir + 'psf.pkl','rb'))
            if psf['name'] == self.psf_name:
                self.psf = psf['psf']
            else:
                psf = poet_psf.readkernels(self)
                pickle.dump(psf,open(self.psf_dir + 'psf.pkl','wb'))
        else:
            # Load saved psf
            psf = pickle.load(open(self.psf_dir + 'psf.pkl','rb'))
        self.psf = psf['psf']
        ##########

        ##### Generate scene for single observation
        if self.quiet == False:
            print('Generating scene...')
            pbar = tqdm(total=self.nstack)
        pixels_final = np.zeros( (self.xout, self.yout) ) # Final simulated CCD image
        for istack in range(self.nstack):
            # Add pointing jitter
            if 'jitter' in self.noise_sources:
                xjit, yjit = self.xjit, self.yjit
            else:
                xjit, yjit = 0.0, 0.0

            xpad = self.xpad * self.noversample
            ypad = self.ypad * self.noversample

            # Add each unconvolved source one by one
            if (istack == 0) | ( (xjit > 1.e-3) | (yjit > 1.e-3) ):
                for isource in range(nsrc):
                    # Get (x,y) coordinates with jitter
                    xcoo = self.gaia['x'][isource] + xjit + xpad/2
                    ycoo = self.gaia['y'][isource] + yjit + ypad/2
                    pixels1 = poet_psf.gen_unconv_image(self,starmodel_flux[isource],xcoo,ycoo)
                    if (isource == 0):
                        pixels = np.copy(pixels1) # Unconvolved simulated CCD image
                    else:
                        pixels += pixels1
                pixels_unconvolved = np.copy(pixels)
            else:
                pixels = np.copy(pixels_unconvolved)

            # Convolve image
            if ((xjit < 1.e-3) & (yjit < 1.e-3)):
                if istack == 0:
                    pixels_conv = fftconvolve(pixels, self.psf[0], mode='same')
            else:
                pixels_conv = fftconvolve(pixels, self.psf[0], mode='same')

            # Remove padding
            pshape = pixels_conv.shape
            xpad = self.xpad * self.noversample
            ypad = self.ypad * self.noversample
            pixels_conv_ras = pixels_conv[ypad:pshape[0]-ypad,xpad:pshape[1]-xpad]

            # Scale to native resolution (remove oversampling.)
            pixels_conv_ras_nav = downscale_local_mean(pixels_conv_ras,(self.noversample,self.noversample))

            # Calculate S/N per pixel
            # Currently only calculated for nstack=1, which is what's used for guide star evaluation
            SN_pix = np.zeros_like(pixels_conv_ras_nav)
            ti = pixels_conv_ras_nav > 0
            SN_pix[ti] = pixels_conv_ras_nav[ti] / np.sqrt( pixels_conv_ras_nav[ti] \
                                + np.sqrt(self.nstack) * self.readnoise + self.darknoise * self.nstack * self.exptime )

            # Calculate target's S/N within ccd_aperture
            if istack == 0:
                from photutils import CircularAperture, aperture_photometry
                x0 = self.gaia['x'] + (self.xout - self.ccd_dim[0])/2 - 0.5
                y0 = self.gaia['y'] + (self.yout - self.ccd_dim[1])/2 + 0.5
                _coord = []
                for ll in range(len(x0)):
                    _coord.append([x0[ll], y0[ll]])
                aperture = CircularAperture(_coord, r=self.ccd_aperture)
                _ap_phot = aperture_photometry(np.abs(pixels_conv_ras_nav), 
                                    aperture, method='center')
                _fl = _ap_phot['aperture_sum']
                n_pix = np.floor(aperture.area)
                SN_ap = np.array( _fl / np.sqrt( _fl + np.sqrt(self.nstack) * self.readnoise + \
                                                self.darknoise * self.nstack * n_pix * self.exptime ) )

            # Add illumination pattern
            if 'illum' in self.noise_sources:
                #add illumination pattern
                #---------------------------
                #read illumination pattern from illumfile
                il = np.genfromtxt(fname=self.illumfile, delimiter=',')
                #difference in size between scene and illum
                z = len(pixels_conv_ras_nav) / len(il)
                #lin interpolate illum pattern to scene size
                il_int = interpolation.zoom(il,z)
                #multiply scene by this pattern
                pixels_conv_ras_nav *= il_int

            # Add additional noise sources
            pixels_conv_ras_nav_noise = np.copy(pixels_conv_ras_nav)

            if 'shot' in self.noise_sources:
                #Shot-noise
                pixels_conv_ras_nav_noise += np.sqrt(np.abs(pixels_conv_ras_nav)) * \
                                                np.random.normal(size=(pixels_conv_ras_nav.shape[0], \
                                                                            pixels_conv_ras_nav.shape[1]) )

            if ('read' in self.noise_sources) | ('dark' in self.noise_sources):
                _std_dev = 0.
                if 'read' in self.noise_sources:
                    _std_dev += self.readnoise * self.exptime / self.gain
                if 'dark' in self.noise_sources:
                    _std_dev += self.darknoise * self.exptime / self.gain

                pixels_conv_ras_nav_noise += _std_dev * np.random.normal(size=(pixels_conv_ras_nav_noise.shape[0],\
                                                                       pixels_conv_ras_nav_noise.shape[1]))


            #Quantize image
            pixels_conv_ras_nav_noise_int = np.floor(pixels_conv_ras_nav_noise)

            # np.floor() was returning -1 for very small, positive numbers
            #    If pixels_conv_ras_nav_noise_int contains -1, set to 0
            ti = pixels_conv_ras_nav_noise_int < 0
            if len( np.where(ti)[0] ) > 0:
                pixels_conv_ras_nav_noise_int[ti] = 0

            # Add for each frame in stack
            pixels_final += pixels_conv_ras_nav_noise_int

            if self.quiet == False:
                pbar.update()

        if return_SN_only:
            return SN_pix, SN_ap

        # Calculate flux error
        pixels_err = np.sqrt( np.abs(pixels_final) / float(self.nstack) )

        if update_gaia:
            self.gaia['scene'] = pixels_final.T
            self.gaia['scene_err'] = pixels_err.T
            self.gaia['SN_pix'] = SN_pix.T
            self.gaia['SN_ap'] = SN_ap
            self.gaia['counts'] = scene_phot_count
            self.gaia['source_flux'] = scene_flux
            self.gaia['gaia_flux'] = gaia_flux

        if self.quiet == False:
            pbar.close()
        ##########

        if self.save:
            self._save()

        if return_scene:
            return pixels_final.T, pixels_err.T
    ##########


    #####
    def lc_sim(self,quiet=False,return_lc=False,cadence=-1,exp_time=-1):
        # cadence: [s] cadence of observations. If < 0, use total exposure time (nstack * iframe * exptime)
        from photutils import CircularAperture, aperture_photometry

        nsrc = len(self.gaia['ra'])

        if cadence < 0:
            cadence = self.nstack * self.exptime
        t_grid = np.arange(self.tstart, self.tend, cadence / (24. * 3600.))
        nt = len(t_grid)

        fl = np.zeros( (nt, nsrc) )
        fl_err = np.zeros_like(fl)

        # Calculate planet transit model
        # Flux decrease due to transit is included in scene_sim via target_flux_fraction parameter
        pl_lc = self.calc_pl_model(t_grid=t_grid,exp_time=exp_time) + 1.

        if quiet == False:
            print('Generating simulated lightcurve...')
            pbar = tqdm(total=nt)
        for n in range(nt):
            # Generate simulated image
            obs, err = self.scene_sim(return_scene=True,quiet=True,update_gaia=False,
                                        target_flux_fraction=pl_lc[n],all_sources=False)

            # Only run for target, not background stars
            for i in range(1):
                x0 = self.gaia['x'][i] + (self.xout - self.ccd_dim[0])/2 - 0.5
                y0 = self.gaia['y'][i] + (self.yout - self.ccd_dim[1])/2 + 0.5
                aperture = CircularAperture((x0, y0), r=self.ccd_aperture)

                _ap_phot = aperture_photometry(np.abs(obs), 
                                aperture, method='exact')
                fl[n,i] = np.sum( _ap_phot['aperture_sum'] )

                _ap_phot = aperture_photometry(np.abs(err), 
                                aperture, method='exact')
                fl_err[n,i] = np.sum( _ap_phot['aperture_sum'] )

            if quiet == False:
                pbar.update()
        if quiet == False:
            pbar.close()

        self.lc_t = t_grid
        self.lc_fl = fl
        self.lc_err = fl_err

        # Normalize to out-of-transit median
        # ti = np.where( np.abs(pl_lc - 1.) < 1.e-8 )[0]
        # if len(ti) > 0:
        #     ti = np.arange(len(pl_lc))
        ti = np.arange(len(pl_lc))
        for i in range(1):
            med_fl = np.median(self.lc_fl[ti,i])
            self.lc_fl[:,i] /= med_fl
            self.lc_err[:,i] /= med_fl

        if self.save:
            self._save()

        if return_lc:
            return t_grid, fl
    ##########


    #####
    def plot_lc(self,save_plot=True,plot_model=True,exp_time=-1,
                    t_from_mid=True,t_unit='d'):
        # Rough plot. Need to update

        t_offset = 0.
        if t_from_mid:
            t_offset = -(self.lc_t[-1] - self.lc_t[0]) / 2.

        t_scale = 1.
        if t_unit == 'hr':
            t_scale = 24.

        y_offset = -1.0
        y_scale = 1.e3

        plt.close(5)

        fig = plt.figure(num=5,figsize=(8,5),facecolor='w')
        ax = fig.add_subplot(111)

        ax.scatter( (self.lc_t + t_offset) * t_scale, (self.lc_fl[:,0] + y_offset) * y_scale,label='Simulated POET')
        ax.errorbar( (self.lc_t + t_offset) * t_scale, (self.lc_fl[:,0] + y_offset) * y_scale, self.lc_err[:,0] * y_scale,ls='None')
        if t_from_mid:
            ax.set_xlabel(r'$t-t_0$ [' + t_unit + ']',fontsize=16)
        else:
            ax.set_xlabel('Time [d]',fontsize=16)
        ax.set_ylabel('Normalized Flux [ppt]',fontsize=16)
        xlim = np.array([self.lc_t[0], self.lc_t[-1]])
        xlim += t_offset
        xlim *= t_scale
        xlim += 0.5 * t_scale * (self.lc_t[1] - self.lc_t[0]) * np.array([-1.0, 1.0])
        ax.set_xlim(xlim)

        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both',labelsize=16)
        ax.tick_params(axis='both',which='major',length=6)
        ax.tick_params(axis='both',which='minor',length=3)
        ax.tick_params(axis='x',which='both',direction='inout')
        ax.tick_params(axis='x',which='both',top=True,direction='in')
        ax.tick_params(axis='y',which='both',direction='inout')
        ax.tick_params(axis='y',which='both',right=True,direction='in')

        if plot_model & hasattr(self,'pl_model'):
            _t_grid = np.linspace(self.lc_t[0],self.lc_t[-1],1000)
            pl_lc = self.calc_pl_model(t_grid=_t_grid,exp_time=exp_time) + 1.
            ax.plot( (_t_grid + t_offset) * t_scale,(pl_lc + y_offset) * y_scale,c='k',label='Transit Model')

            ax.legend(handletextpad=0.3,handlelength=2,
                                    labelspacing=0.2,facecolor='w',framealpha=0.0,
                                    fontsize=10,frameon=True,ncol=1)

        fig.tight_layout(pad=0.2)

        plt.ion()
        plt.show()

        if save_plot:
            # fig.savefig(self.prefix + '_lc.pdf',format='pdf')
            fig.savefig(self.prefix + '_lc.png',format='png',dpi=200)

    ##########


    #####
    def calc_pl_model(self,model='pytransit_QuadraticModel',t_grid=[],exp_time=-1):
        # To do: include additional/alternative transit models
        if len(t_grid) == 0:
            cadence = self.nstack * self.exptime
            t_grid = np.arange(self.tstart, self.tend, cadence / (24. * 3600.))
        nt = len(t_grid)

        const_G = 6.6743e-8 # [cm3 / (g s2)]

        _pl_lc_tot = np.zeros(nt)
        if hasattr(self,'pl_model'):
            _pl_lc = []
            n_pl = len(self.pl_model['RpRs'])

            if model == 'pytransit_QuadraticModel':
                from pytransit import QuadraticModel
                
                tm = QuadraticModel()

                if exp_time < 0:
                    tot_exptime = self.nstack * self.exptime / (24. * 3600.)
                else:
                    tot_exptime = exp_time / (24. * 3600.)
                tm.set_data(time=t_grid, exptimes=tot_exptime, nsamples=10)

                for n in range(n_pl):
                    if 'u' in self.pl_model.keys():
                        ldc = self.pl_model['u']
                    else:
                        ldc = [0.6, 0.3]
                    if 'aRs' in self.pl_model.keys():
                        aRs = self.pl_model['aRs'][n]
                    elif 'rho_s' in self.pl_model.keys():
                        aRs = ( self.pl_model['rho_s'] * const_G * (self.pl_model['P'][n]*24.*3600.)**2 / (3. * np.pi) )**(1./3.)
                    _irad = np.arccos( self.pl_model['b'][n] / self.pl_model['aRs'][n] )
                    if 'e' in self.pl_model.keys():
                        _e, _w = self.pl_model['e'][n], self.pl_model['w'][n]
                    else:
                        _e, _w = 0., 0.
                    _pv = [self.pl_model['RpRs'][n], self.pl_model['t0'][n],
                            self.pl_model['P'][n], self.pl_model['aRs'][n],
                            _irad, _e, _w ]
                    _pl_lc.append( tm.evaluate_ps(_pv[0], ldc, _pv[1], _pv[2], _pv[3],
                                    _pv[4], _pv[5], _pv[6]) - 1. )

                # Add calculation of total model
                _pl_lc_tot = np.hstack(_pl_lc) # Need to update for multi-planet

        return _pl_lc_tot
    ##########




