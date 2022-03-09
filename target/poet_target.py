import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS

from tqdm import tqdm

import pickle

class Target(object):
    #####
    def __init__(self,prefix='poet_target',t_id='',
                    ra=None,dec=None, # Target coordinates [deg]
                    srch_Gmax=14., # Max Gaia G magnitude for Gaia query
                    srch_nmax=1000, # Max number of Gaia sources to return
                    srch_radmax=None, # Max search radius for Gaia query [deg]
                    fov=1.0, # Full-width FoV [deg] used for converting ra,dec -> x_pxl, y_pxl
                    fov_type='square', # FoV shape
                    fov_pa=0., # FoV position angle [deg]
                    ccd_dim=[1024,1024], # [pxl,pxl] used for plotting (including converting ra,dec -> x_pxl,y_pxl)
                    gs_criteria={'Gmag':[-10, 12.]}, # Criteria for guide star
                    pl_model={},
                    save=False,quiet=False,
                    ):
        self.prefix = prefix
        self.t_id = t_id.replace(' ','_')
        if ra == None:
            raise Exception("Input RA [deg].")
        if dec == None:
            raise Exception("Input DEC [deg].")
        self.ra = ra
        self.dec = dec

        self.fov = fov
        self.fov_type = fov_type
        self.fov_pa = fov_pa
        self.ccd_dim = ccd_dim

        self.srch_Gmax = srch_Gmax
        self.srch_nmax = srch_nmax

        if srch_radmax == None:
            if fov_type == 'square':
                srch_radmax = (fov / 2.) * np.sqrt(2.)
            elif fov_type == 'circle':
                srch_radmax = fov / 2.
        self.srch_radmax = srch_radmax

        self.gs_criteria = gs_criteria

        self.save = save
        self.quiet = quiet

        self.fname = ''
        if self.prefix != '':
            self.fname += self.prefix

        if (self.prefix != '') & (self.t_id != ''):
            self.fname += '_'

        if self.t_id != '':
            self.fname += self.t_id
        self.fname += '.pkl'

        if True:
            # Initialize model parameters
            self.psf_dir = '/Users/james/poet/Kernels/'
            self.psf_name = 'POET_PSF_CenterFoV_VNIR.txt' # Filename of PSF kernel file
            self.tstart = 0.0 # start time (days)
            self.tend = 10.0 # end time (days)
            self.exptime = 1 #exposure time (s)
            self.nstack = 30
            self.iframe = 1 # number of frames used to generate exposure time (needed for pointing jitter)
            self.pl_model = pl_model # Contain parameters for transit model (Rp/Rs, t0, P, ...)
            self.gain = 6.1 # electronic gain [e-/adu]
            self.saturation = 65536.0 #saturation
            self.jitter_dis = 0.#1.0 #pointing jitter in dispersion axis [pixels, rms]
            self.jitter_spa = 0.#1.0 #pointing jitter in spatial axis [pixels, rms]
            self.readnoise = 8.0 #readnoise electrons
            self.xpad = 10 #padding to deal with convolution fall-off
            self.ypad = 10 #padding to deal with convolution fall-off
            self.xout = 1024 #x-axis
            self.yout = 1024 #y-axis
            self.noversample = 2
            self.ccd_aperture = 5 # radius [pxl] for integrating flux on CCD to generate LC
            self.detector_aperture = 0.15 # instrument aperture radius [m] used for ETC photon count
            self.detector_pixscale = 13. # pixel size [microns]

        if self.save:
            self._save()
    ##########


    def _save(self):
        pickle.dump(self,open(self.fname,'wb'))


    #####
    def search_gaia(self, srch_Gmax=None, srch_nmax=None, srch_radmax=None):
        from astroquery.gaia import Gaia

        if srch_Gmax != None:
            self.srch_Gmax = srch_Gmax
        if srch_nmax != None:
            self.srch_nmax = srch_nmax
        if srch_radmax != None:
            self.srch_radmax = srch_radmax

        try:
            # srch_str = "SELECT TOP {:.0f}".format(int(self.srch_nmax)) + " * " \
            #     + "FROM gaiadr2.gaia_source " \
            #     + "WHERE CONTAINS(   POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),   " \
            #     + "CIRCLE('ICRS',{:.6f},{:.6f},{:.2f}))=1".format(self.ra,self.dec,self.srch_radmax) \
            #     + "  AND  (gaiadr2.gaia_source.phot_g_mean_mag<={:.2f})".format(self.srch_Gmax) \
            #     + "ORDER BY ang_sep ASC"
            srch_str = "SELECT * " \
                + "FROM gaiadr2.gaia_source " \
                + "WHERE CONTAINS(   POINT('ICRS',{:.6f},{:.6f}),   ".format(self.ra,self.dec) \
                + "CIRCLE('ICRS',ra,dec,{:.2f}))=1".format(self.srch_radmax) \
                + "  AND  (phot_g_mean_mag<={:.2f})".format(self.srch_Gmax) \
                + "ORDER BY phot_g_mean_mag ASC"
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
            sep_max = 5.0 / 3600. # [deg]
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
            self.gaia = {'ra':ra, 'dec':dec, 
                         'x':np.zeros_like(ra), 'y':np.zeros_like(ra),
                         'Gmag':Gmag, 'Bpmag':Bpmag, 'Rpmag':Rpmag,
                         'results':None, 'wcs':None}

            self.calc_gaia_xy()
            self.calc_gaia_mag()

            if self.save:
                pickle.dump(self,open(self.fname,'wb'))
        except Exception:
            if self.quiet == False:
                print('Gaia search failed.')
    ##########


    #####
    def calc_gaia_xy(self,fov_pa=None):
        if fov_pa != None:
            self.fov_pa = fov_pa

        # Get projected CCD coordinates
        _cd = np.array([ [0., 0.], [0., 0.] ])
        wcs_input_dict = {
            'CTYPE1': 'RA---TAN', 
            'CUNIT1': 'deg', 
            'CDELT1': self.fov/self.ccd_dim[0], 
            'CRPIX1': int(self.ccd_dim[0]/2), 
            # 'CRPIX1': int(self.xout/2), 
            'CRVAL1': self.gaia['ra'][0], 
            'NAXIS1': self.ccd_dim[0],
            'NAXIS1': self.xout,
            'CTYPE2': 'DEC--TAN', 
            'CUNIT2': 'deg', 
            'CDELT2': self.fov/self.ccd_dim[1], 
            'CRPIX2': int(self.ccd_dim[1]/2), 
            # 'CRPIX2': int(self.yout/2), 
            'CRVAL2': self.gaia['dec'][0], 
            'NAXIS2': self.ccd_dim[1],
            # 'NAXIS2': self.yout,
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
    def calc_gaia_mag(self,filters=['U', 'I'],
            eem_table_fname='EEM_dwarf_UBVIJHK_colors_Teff.txt'):
        # Convert Gaia bandpass measurements to other filters
        # Only U and I currently included

        # Read in EEM table
        eem_table = []
        with open(eem_table_fname,'r') as file:
            data_block = False
            for line in file:
                # Only read in between lines starting with '$SpT'
                if line.startswith('#SpT'):
                    if (data_block == False):
                        data_block = True
                    else:
                        break
                if data_block:
                    eem_table.append(line.rstrip().split())
        eem_table_hdr = np.array(eem_table[0])
        eem_table = np.array(eem_table[1:],dtype=str)

        ci = np.where(eem_table_hdr == 'Teff')[0]
        eem_teff = eem_table[:,ci]

        ci = np.where(eem_table_hdr == 'B-V')[0]
        eem_B_V = eem_table[:,ci]

        ci = np.where(eem_table_hdr == 'G-V')[0]
        eem_G_V = eem_table[:,ci]

        ci = np.where(eem_table_hdr == 'Bp-Rp')[0]
        eem_Bp_Rp = eem_table[:,ci]

        ci = np.where(eem_table_hdr == 'U-B')[0]
        eem_U_B = eem_table[:,ci]

        ci = np.where(eem_table_hdr == 'V-Ic')[0]
        eem_V_Ic = eem_table[:,ci]

        ti = np.where( (eem_teff != '...') & (eem_B_V != '...') & \
                       (eem_G_V != '...') & (eem_Bp_Rp != '...') & \
                       (eem_U_B != '...') & (eem_V_Ic != '...') )[0]

        eem_teff = np.array(eem_teff[ti],dtype=float).flatten()
        eem_B_V = np.array(eem_B_V[ti],dtype=float).flatten()
        eem_G_V = np.array(eem_G_V[ti],dtype=float).flatten()
        eem_Bp_Rp = np.array(eem_Bp_Rp[ti],dtype=float).flatten()
        eem_U_B = np.array(eem_U_B[ti],dtype=float).flatten()
        eem_V_Ic = np.array(eem_V_Ic[ti],dtype=float).flatten()

        si = np.argsort(eem_Bp_Rp)
        Bp_Rp = self.gaia['Bpmag'] - self.gaia['Rpmag']
        _G_V = np.interp(Bp_Rp,eem_Bp_Rp[si],eem_G_V[si],left=np.nan,right=np.nan)
        _B_V = np.interp(Bp_Rp,eem_Bp_Rp[si],eem_B_V[si],left=np.nan,right=np.nan)
        _U_B = np.interp(Bp_Rp,eem_Bp_Rp[si],eem_U_B[si],left=np.nan,right=np.nan)
        _V_I = np.interp(Bp_Rp,eem_Bp_Rp[si],eem_V_Ic[si],left=np.nan,right=np.nan)
        U = _U_B + _B_V - _G_V + self.gaia['Gmag']
        Ic = self.gaia['Gmag'] - (_G_V + _V_I)

        if 'U' in filters:
            self.gaia['Umag'] = U

        if 'I' in filters:
            self.gaia['Imag'] = Ic
    ##########


    #####
    def plot_fov(self,pa=None,save_plot=False,
                    plot_guide_stars=True,plot_bkg_stars=True,
                    plot_grid=True,add_scene_sim=False):

        if hasattr(self,'gaia') == False:
            self.search_gaia()

        if pa != None:
            self.fov_pa = pa

        plt.close(1)
        fig = plt.figure(num=1,figsize=(5.5,5))
        ax = fig.add_subplot(111)

        ax_pos = ax.get_position()
        ax_pos.x0 = 0.175
        ax_pos.x1 = 0.99
        ax_pos.y0 = 0.108
        ax_pos.y1 = 0.99
        ax.set_position(ax_pos)

        Gmag_lim = [np.min(self.gaia['Gmag']), np.max(self.gaia['Gmag'])]
        sym_lim = [0.5, 100]
        sym_size = ((Gmag_lim[1] - self.gaia['Gmag']) / (Gmag_lim[1] - Gmag_lim[0] )) \
                        * (sym_lim[1] - sym_lim[0]) + sym_lim[0]

        # Plot background sources
        if plot_bkg_stars:
            ax.scatter(self.gaia['x'],self.gaia['y'],
                        facecolor='tab:blue',edgecolor='None',
                        # facecolor='None',edgecolor='tab:blue',
                        zorder=0,s=sym_size,alpha=0.7)

        # Plot target
        ax.scatter(self.gaia['x'][0],self.gaia['y'][0],c='k',marker='+',s=50)

        # Plot FoV boundary
        if self.fov_type == 'circle':
            _x = (self.gaia.ccd_dim[0] / 2.) * np.cos(_theta) + int(self.gaia.ccd_dim[0]/2)
            _y = (self.gaia.ccd_dim[1] / 2.) * np.sin(_theta) + int(self.gaia.ccd_dim[1]/2)
            ax.plot(_x,_y,c='r')
        elif self.fov_type == 'square':
            _x = [0, self.ccd_dim[0], self.ccd_dim[0], 0, 0]
            _y = [0, 0, self.ccd_dim[1], self.ccd_dim[1], 0]
            ax.plot(_x,_y,c='r')

        if plot_guide_stars:
            if ('gs_i' in self.gaia.keys()) == False:
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
                                int(self.ccd_dim[1]/2) + self.yout/2 - 2]) # I don't know where the -2 comes from that's necessary here...

            cmap = plt.get_cmap('cividis')
            _f = self.gaia['scene']-np.min(self.gaia['scene'])+1
            ax.imshow(_f,norm=LogNorm(),
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
            fig.savefig(self.fname[:-4] + '_fov.pdf',format='pdf')
    ##########


    #####
    def id_guide_stars(self,gs_criteria=None):
        if gs_criteria != None:
            self.gs_criteria = gs_criteria

        gs_i = np.arange(1,len(self.gaia['ra']))

        _ti = np.where( (self.gaia['x'][gs_i] > 0) \
                        & (self.gaia['x'][gs_i] < self.ccd_dim[0]) \
                        & (self.gaia['y'][gs_i] > 0) \
                        & (self.gaia['y'][gs_i] < self.ccd_dim[1]) )[0]
        gs_i = gs_i[_ti]
        
        # Apply guide star criteria filter
        # To add: min/max (x,y) or (ra,dec) separation criterion
        if 'Gmag' in self.gs_criteria.keys():
            _ti = np.where( (self.gaia['Gmag'][gs_i] > self.gs_criteria['Gmag'][0]) \
                            &  (self.gaia['Gmag'][gs_i] < self.gs_criteria['Gmag'][1]) )[0]
            gs_i = gs_i[_ti]

        if 'Umag' in self.gs_criteria.keys():
            _ti = np.where( (self.gaia['Umag'][gs_i] > self.gs_criteria['Umag'][0]) \
                            &  (self.gaia['Umag'][gs_i] < self.gs_criteria['Umag'][1]) )[0]
            gs_i = gs_i[_ti]

        if 'Imag' in self.gs_criteria.keys():
            _ti = np.where( (self.gaia['Imag'][gs_i] > self.gs_criteria['Imag'][0]) \
                            &  (self.gaia['Imag'][gs_i] < self.gs_criteria['Imag'][1]) )[0]
            gs_i = gs_i[_ti]

        if self.quiet == False:
            print('{:.0f} guide star(s) identified.'.format(len(gs_i)))

        self.gaia['gs_i'] = gs_i
    ##########


    #####
    def gen_scene(self,psf,starmodel_flux,quiet=False,return_err=True):
        from target import poet_etc, poet_psf
        from scipy.signal import fftconvolve

        from skimage.transform import downscale_local_mean

        nsrc = len(self.gaia['ra']) # Number of sources

        # Generate scene
        if quiet == False:
            print('Generating scene...')
            pbar = tqdm(total=self.nstack*self.iframe)
        pixels_final = np.zeros((self.xout,self.yout))
        for istack in range(self.nstack):
            for icount in range(self.iframe):
                # adding pointing jitter
                xjit = np.random.normal() * self.jitter_dis
                yjit = np.random.normal() * self.jitter_spa
                xpad = self.xpad * self.noversample
                ypad = self.ypad * self.noversample

                # Add each Gaia source
                for isource in range(nsrc):
                    xcoo = self.gaia['x'][isource] + xjit + xpad/2
                    ycoo = self.gaia['y'][isource] + yjit + ypad/2
                    if (icount == 0) & (isource == 0):
                        pixels = poet_psf.gen_unconv_image(self,starmodel_flux[isource],xcoo,ycoo)
                    else:
                        pixels1 = poet_psf.gen_unconv_image(self,starmodel_flux[isource],xcoo,ycoo)
                        pixels += pixels1
                if quiet == False:
                    pbar.update()
            pixels = pixels / self.iframe

            # Create Convolved Image
            pixels_conv = fftconvolve(pixels, psf[0], mode='same')

            # remove padding
            pshape = pixels_conv.shape
            xpad = self.xpad * self.noversample
            ypad = self.ypad * self.noversample
            pixels_conv_ras = pixels_conv[ypad:pshape[0]-ypad,xpad:pshape[1]-xpad]

            # Scale to native resolution (remove oversampling.)
            pixels_conv_ras_nav = downscale_local_mean(pixels_conv_ras,(self.noversample,self.noversample))

            # add Noise.

            #Shot-noise
            pixels_conv_ras_nav_noise = pixels_conv_ras_nav + np.sqrt(np.abs(pixels_conv_ras_nav))*\
                                                          np.random.normal(size=(pixels_conv_ras_nav.shape[0],\
                                                          pixels_conv_ras_nav.shape[1]))
            #Read-noise
            rnoise_ADU = self.readnoise / self.gain
            pixels_conv_ras_nav_noise += rnoise_ADU * np.random.normal(size=(pixels_conv_ras_nav_noise.shape[0],\
                                                                   pixels_conv_ras_nav_noise.shape[1]))

            #Quantize image
            pixels_conv_ras_nav_noise_int = np.floor(pixels_conv_ras_nav_noise)

            # Add for each frame in stack
            pixels_final += pixels_conv_ras_nav_noise_int

        if quiet == False:
            pbar.close()

        if return_err:
            # Calculate flux error
            rnoise_ADU = self.readnoise / self.gain
            pixels_err = np.sqrt( np.abs(pixels_final) ) + rnoise_ADU * np.sqrt(self.nstack)

            return pixels_final, pixels_err
        else:
            return pixels_final
    ##########


    #####
    def scene_sim(self,update_gaia=True,quiet=False,return_scene=False,
                    reload_kernel=False):
        from target import poet_etc, poet_psf
        from scipy.io import loadmat

        import importlib
        importlib.reload(poet_psf)

        # Generate simulated CCD image based on Gaia sources
        if hasattr(self,'gaia') == False:
            self.search_gaia()

        # Calculate flux using ETC for each Gaia source
        nsrc = len(self.gaia['ra']) # Number of sources
        starmodel_flux = np.zeros(nsrc)
        BPwv = np.arange(733,800,0.25).tolist()
        BPwv = np.asfarray(BPwv, float)
        bandpass = [0.2] * BPwv
        zero_point = 25.6884
        for n in range(nsrc):
            _tmp = poet_etc.Photon_Count(temp = 9500, metallicity = 'p00', logG = 'g45', 
                                           Wmin = 500, Wmax = 1000,
                                           bandpass = bandpass,bandpassWave =BPwv, 
                                           aperture = self.detector_aperture, # aperture radius [m]
                                           GAIA_mag = self.gaia['Gmag'][n],
                                           zero_point = zero_point)
            # Get total counts for single exposure
            starmodel_flux[n] = np.sum(_tmp) * self.exptime / 6.

        # Load PSF kernel
        if reload_kernel | (hasattr(self, 'psf') == False):
            psf = poet_psf.readkernels(self)
            self.psf = psf
        else:
            psf = self.psf

        # Get CCD flux array
        pixels_final, pixels_err = self.gen_scene(psf,starmodel_flux, \
                                        quiet=quiet,return_err=True)

        if update_gaia:
            self.gaia['scene'] = pixels_final.T
            self.gaia['scene_err'] = pixels_err.T

            if self.save:
                self._save()

        if return_scene:
            return pixels_final.T, pixels_err.T
    ##########


    #####
    def lc_sim(self,quiet=False,return_lc=False):
        from photutils import CircularAperture, aperture_photometry

        nsrc = len(self.gaia['ra'])

        t_grid = np.arange(self.tstart,self.tend,self.exptime/(24. * 3600.) )
        nt = len(t_grid)

        if True:
            # on/off for testing
            fl = np.zeros( (nt, nsrc) )
            fl_err = np.zeros_like(fl)
            if quiet == False:
                print('Generating simulated lightcurve...')
                pbar = tqdm(total=nt)
            for n in range(nt):
                # Generate simulated image
                obs, err = self.scene_sim(return_scene=True,quiet=True,update_gaia=False)

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

            # Normalize to median
            for i in range(nsrc):
                med_fl = np.median(self.lc_fl[:,i])
                self.lc_fl[:,i] /= med_fl
                self.lc_err[:,i] /= med_fl

        # Add planet model to central target only
        pl_lc = self.calc_pl_model()
        self.lc_fl[:,0] += pl_lc

        if self.save:
            self._save()

        if return_lc:
            return t_grid, fl
    ##########


    #####
    def calc_pl_model(self,model='pytransit_QuadraticModel'):
        t_grid = np.arange(self.tstart,self.tend,self.exptime/(24.*3600.))
        nt = len(t_grid)

        _pl_lc_tot = np.zeros(nt)
        if len(self.pl_model) > 0:
            _pl_lc = []
            n_pl = len(self.pl_model['RpRs'])

            if model == 'pytransit_QuadraticModel':
                from pytransit import QuadraticModel
                
                tm = QuadraticModel()

                tot_exptime = self.nstack * self.iframe * self.exptime / (24. * 3600.)
                tm.set_data(time=t_grid, exptimes=tot_exptime,
                            nsamples=10)

                for n in range(n_pl):
                    ldc = [0.6, 0.3]
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




