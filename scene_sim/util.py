import numpy as np

#####
def interp_EEM_table(Teff=[], Gmag=[], Bpmag=[], Rpmag=[]):
    # Interpolate EEM_table using input Teff
    table_fname = 'stellar_models/EEM_dwarf_UBVIJHK_colors_Teff.txt'

    _logg_sun = 4.438

    # Read in EEM table
    eem = {'Teff':[], 'radius':[], 'mass':[], 'logg':[], 'Gmag_abs':[],
           'B-V':[], 'G-V':[], 'Bp-Rp':[], 'G-Rp':[], 'U-B':[], 'V-I':[]}
    eem_hdr = []
    with open(table_fname,'r') as file:
        data_block = False
        for line in file:
            # Only read in between lines starting with '$SpT'
            if line.startswith('#SpT'):
                if (data_block == False):
                    data_block = True
                else:
                    break
            if data_block:
                _line = line.rstrip().split()

                if len(eem_hdr) == 0:
                    eem_hdr = np.array(_line.copy())
                else:
                    ci = np.where(eem_hdr == 'Teff')[0][0]
                    eem['Teff'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'R_Rsun')[0][0]
                    if _line[ci].startswith('...'):
                        eem['radius'].append( np.nan )
                    else:
                        eem['radius'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'Msun')[0][0]
                    if _line[ci].startswith('...'):
                        eem['mass'].append( np.nan )
                    else:
                        eem['mass'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'M_G')[0][0]
                    if _line[ci].startswith('...'):
                        eem['Gmag_abs'].append( np.nan )
                    elif _line[ci].endswith(':'):
                        eem['Gmag_abs'].append( float(_line[ci][:-1]) )
                    else:
                        eem['Gmag_abs'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'B-V')[0][0]
                    if _line[ci].startswith('...'):
                        eem['B-V'].append( np.nan )
                    else:
                        eem['B-V'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'G-V')[0][0]
                    if _line[ci].startswith('...'):
                        eem['G-V'].append( np.nan )
                    else:
                        eem['G-V'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'Bp-Rp')[0][0]
                    if _line[ci].startswith('...'):
                        eem['Bp-Rp'].append( np.nan )
                    else:
                        eem['Bp-Rp'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'G-Rp')[0][0]
                    if _line[ci].startswith('...'):
                        eem['G-Rp'].append( np.nan )
                    else:
                        eem['G-Rp'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'U-B')[0][0]
                    if _line[ci].startswith('...'):
                        eem['U-B'].append( np.nan )
                    else:
                        eem['U-B'].append( float(_line[ci]) )

                    ci = np.where(eem_hdr == 'V-Ic')[0][0]
                    if _line[ci].startswith('...'):
                        eem['V-I'].append( np.nan )
                    else:
                        eem['V-I'].append( float(_line[ci]) )
    for _lbl in eem.keys():
        eem[_lbl] = np.array(eem[_lbl])

    # Calculate logg
    eem['logg'] = _logg_sun + np.log10( eem['mass'] )  - 2. * np.log10( eem['radius']**2 )

    # Sort by Teff
    si = np.argsort(eem['Teff'])
    for _lbl in eem.keys():
        eem[_lbl] = eem[_lbl][si]

    # If Teff not specified, use Bpmag, Rpmag to get Teff
    if len(Teff) == 0:
        si = np.argsort(eem['Bp-Rp'])
        Teff = np.interp(Bpmag - Rpmag, eem['Bp-Rp'][si], eem['Teff'][si])

    # Interpolate table using input Teff
    # Use grid edges for missing values
    interp_output = {'Teff':Teff}
    for _lbl in eem.keys():
        if _lbl != 'Teff':
            vi = np.isfinite(eem[_lbl])
            interp_output[_lbl] = np.interp(np.log10(Teff), 
                                        np.log10(eem['Teff'][vi]), eem[_lbl][vi])

    # Calculate U, I and add to table (if Gmag provided)
    if len(Gmag) > 0:
        interp_output['Rpmag'] = Gmag - interp_output['G-Rp']
        interp_output['Bpmag'] = interp_output['Bp-Rp'] + interp_output['Rpmag']

        interp_output['Umag'] = interp_output['U-B'] + interp_output['B-V'] - \
                                    interp_output['G-V'] + Gmag
        interp_output['Imag'] = Gmag - ( interp_output['G-V'] + interp_output['V-I'] )

    return interp_output
##########


