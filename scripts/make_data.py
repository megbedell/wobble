import numpy as np
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
import h5py
from astropy.io import fits
import shutil
import glob
import os

def dimensions(instrument):
    if instrument == 'HARPS':
        M = 4096 # pixels per order
        R = 72 # orders
    elif instrument == 'HARPS-N':
        M = 4096 # pixels per order
        R = 69 # orders
    else:
        print("instrument not recognized. valid options are: HARPS, HARPS-N")
        assert False
    return M, R
    
def read_spec_2d(spec_file, blaze=False, flat=False):
    '''Read a HARPS 2D spectrum file from the ESO pipeline

    Parameters
    ----------
    spec_file : string
    name of the fits file with the data (e2ds format)
    blaze : boolean
    if True, then divide out the blaze function from flux
    flat : boolean
    if True, then divide out the flatfield from flux
    
    Returns
    -------
    wave : np.ndarray (shape n_orders x 4096)
    wavelength (in Angstroms)
    flux : np.ndarray (shape n_orders x 4096)
    flux value 
    '''
    path = spec_file[0:str.rfind(spec_file,'/')+1]
    sp = fits.open(spec_file)
    header = sp[0].header
    flux = sp[0].data
    try:
        wave_file = header['HIERARCH ESO DRS CAL TH FILE']
    except KeyError: # HARPS-N
        wave_file = header['HIERARCH TNG DRS CAL TH FILE']
    wave_file = str.replace(wave_file, 'e2ds', 'wave') # just in case of header mistake..
                                                       # ex. HARPS.2013-03-13T09:20:00.346_ccf_M2_A.fits
    try:
        ww = fits.open(path+wave_file)
        wave = ww[0].data
    except:
        print("Wavelength solution file {0} not found!".format(wave_file))
        return
    if blaze:
        blaze_file = header['HIERARCH ESO DRS BLAZE FILE']
        bl = fits.open(path+blaze_file)
        blaze = bl[0].data
        flux /= blaze
    if flat:
        flat_file = header['HIERARCH ESO DRS CAL FLAT FILE']
        fl = fits.open(path+flat_file)
        flat = fl[0].data
        flux /= flat
    return wave, flux
    
def read_snr(filename, instrument='HARPS'):
    '''Parse SNR from header of a HARPS(-S or -N) file from the ESO or TNG pipelines

    Parameters
    ----------
    filename : string
    name of the fits file with the data (can be ccf, e2ds, s1d)

    Returns
    -------
    snr : np.ndarray
    SNR values taken near the center of each order
    '''
    sp = fits.open(filename)
    header = sp[0].header
    
    if instrument=='HARPS':
        n_orders = 72
    elif instrument=='HARPS-N':
        n_orders = 69
    else:
        print("ERROR: instrument {0} not recognized.".format(instrument))
        return
    snr = np.arange(n_orders, dtype=np.float)
    for i in np.nditer(snr, op_flags=['readwrite']):
        if instrument=='HARPS':
            i[...] = header['HIERARCH ESO DRS SPE EXT SN{0}'.format(str(int(i)))]
        elif instrument=='HARPS-N':
            i[...] = header['HIERARCH TNG DRS SPE EXT SN{0}'.format(str(int(i)))]
    return snr

def read_data_from_fits(filelist, instrument='HARPS', e2ds=False):
    '''Parses a list of HARPS CCF files.
    
    Parameters
    ----------
    filelist : list of strings
    list of filenames for HARPS (or HARPS-N) CCF files

    Returns
    -------
    data : list of numpy arrays
    flux values in format [(N_epochs, M_pixels) for r in R_orders].
    note that the echelle orders may be of different pixel lengths, 
    but all epochs must be consistent.
    
    ivars : list of numpy arrays
    Inverse variance errors on data in the same format.
    
    xs : list of numpy arrays
    Wavelength values for each pixel, in the same format as data.
    
    pipeline_rvs : numpy array
    N_epoch length array of RVs estimated by the HARPS pipeline. 
    These RVs are drift-corrected but NOT barycentric corrected.
    
    pipeline_sigmas : numpy array
    N_epoch length array of error estimates on HARPS pipeline RVs.
    
    dates : numpy array
    N_epoch length array of observation times.
    
    bervs : numpy array
    N_epoch length array of barycentric RVs.
    
    airms : numpy array
    N_epoch length array of airmass.
    
    drifts : numpy array
    N_epoch length array of instrumental drifts.    
    '''
    N = len(filelist)  # number of epochs    
    M, R = dimensions(instrument)
    data = [np.zeros((N,M)) for r in range(R)]
    ivars = [np.zeros((N,M)) for r in range(R)]
    xs = [np.zeros((N,M)) for r in range(R)]
    empty = np.array([], dtype=int)
    pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n,f in enumerate(filelist):
        sp = fits.open(f)
        if not e2ds:
            try:
                if instrument == 'HARPS':
                    pipeline_rvs[n] = sp[0].header['HIERARCH ESO DRS CCF RVC'] * 1.e3 # m/s
                    pipeline_sigmas[n] = sp[0].header['HIERARCH ESO DRS CCF NOISE'] * 1.e3 # m/s
                    drifts[n] = sp[0].header['HIERARCH ESO DRS DRIFT SPE RV']  
                elif instrument == 'HARPS-N':
                    pipeline_rvs[n] = sp[0].header['HIERARCH TNG DRS CCF RVC'] * 1.e3 # m/s
                    pipeline_sigmas[n] = sp[0].header['HIERARCH TNG DRS CCF NOISE'] * 1.e3 # m/s
                    drifts[n] = sp[0].header['HIERARCH TNG DRS DRIFT RV USED']
            except KeyError:
                print("WARNING: {0} does not appear to be a stellar CCF file. Skipping this one.".format(f))
                empty = np.append(empty, n)
                continue
        if instrument == 'HARPS':
            dates[n] = sp[0].header['HIERARCH ESO DRS BJD']        
            bervs[n] = sp[0].header['HIERARCH ESO DRS BERV'] * 1.e3 # m/s
            airms[n] = sp[0].header['HIERARCH ESO TEL AIRM START']
        elif instrument == 'HARPS-N':
            dates[n] = sp[0].header['HIERARCH TNG DRS BJD']        
            bervs[n] = sp[0].header['HIERARCH TNG DRS BERV'] * 1.e3 # m/s
            airms[n] = sp[0].header['AIRMASS']            
        
        spec_file = str.replace(f, 'ccf_G2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_M2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_K5', 'e2ds') 
        try:
            wave, spec = read_spec_2d(spec_file)
        except:
            empty = np.append(empty, n)
            continue
        snrs = read_snr(f, instrument=instrument) # HACK
        # save stuff
        for r in range(R):
            data[r][n,:] = spec[r,:]
            ivars[r][n,:] = snrs[r]**2/spec[r,:]/np.nanmean(spec[r,:]) # scaling hack
            xs[r][n,:] = wave[r,:] 
            
    # delete data without wavelength solutions:
    for r in range(R):
        data[r] = np.delete(data[r], empty, axis=0)
        ivars[r] = np.delete(ivars[r], empty, axis=0)
        xs[r] = np.delete(xs[r], empty, axis=0)
    
    pipeline_rvs = np.delete(pipeline_rvs, empty)
    pipeline_sigmas = np.delete(pipeline_sigmas, empty)
    dates = np.delete(dates, empty)
    bervs = np.delete(bervs, empty)
    airms = np.delete(airms, empty)
    drifts = np.delete(drifts, empty)
    
    # re-introduce BERVs to HARPS results:
    pipeline_rvs -= bervs  
    pipeline_rvs -= np.mean(pipeline_rvs)
        
    return data, ivars, xs, pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts
    
def savfile_to_filelist(savfile, destination_dir='../data/'):
    # copies CCF + E2DS files to destination_dir and returns a list of the CCFs
    # MB personal use only - I have a lot of old IDL files!
    s = readsav(savfile)
    filelist = []
    files = [f.decode('utf8') for f in s.files]
    for f in files:
        shutil.copy2(f, destination_dir)
        spec_file = str.replace(f, 'ccf_G2', 'e2ds')
        shutil.copy2(spec_file, destination_dir)
        basename = f[str.rfind(f,'/')+1:]
        filelist = np.append(filelist, destination_dir+basename)
    return filelist
    
def missing_wavelength_files(filelist):
    # loop through files and make sure that their wavelength solutions exist
    # return list of all missing wavelength solution files
    missing_files = []
    for f in filelist:
        path = f[0:str.rfind(f,'/')+1]
        sp = fits.open(f)
        header = sp[0].header
        wave_file = header['HIERARCH ESO DRS CAL TH FILE']
        if os.path.isfile(path+wave_file):
            continue
        else:
            missing_files = np.append(missing_files, wave_file)
            
    return np.unique(missing_files)
    
    
def write_data(data, ivars, xs, pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts, filenames, hdffile):
    '''Write processed HARPS data to HDF5 file. 
    Note that currently all input parameters are required, 
    but the following ones can be populated with zeros if you don't have them:
    pipeline_rvs, pipeline_sigmas, bervs, drifts, filenames
    
    These parameters *are* strictly required:
    data, ivars, xs, dates, airms
    
    And bervs is strongly recommended as they are used to initialize any stellar RVs.

    Parameters
    ----------
    data : list of numpy arrays
    flux values in format [(N_epochs, M_pixels) for r in R_orders].
    note that the echelle orders may be of different pixel lengths, 
    but all epochs must be consistent.
    
    ivars : list of numpy arrays
    Inverse variance errors on data in the same format.
    
    xs : list of numpy arrays
    Wavelength values for each pixel, in the same format as data.
    
    pipeline_rvs : numpy array
    N_epoch length array of RVs estimated by the HARPS pipeline. 
    These RVs are drift-corrected but NOT barycentric corrected.
    
    pipeline_sigmas : numpy array
    N_epoch length array of error estimates on HARPS pipeline RVs.
    
    dates : numpy array
    N_epoch length array of observation times.
    
    bervs : numpy array
    N_epoch length array of barycentric RVs.
    
    airms : numpy array
    N_epoch length array of airmass.
    
    drifts : numpy array
    N_epoch length array of instrumental drifts.   
    
    filenames : list or numpy array
    N_epoch length list of data files.
    
    hdffile : string
    Filename to write to.
    '''
    h = h5py.File(hdffile, 'w')
    dset = h.create_dataset('data', data=data)
    dset = h.create_dataset('ivars', data=ivars)
    dset = h.create_dataset('xs', data=xs)
    dset = h.create_dataset('pipeline_rvs', data=pipeline_rvs)
    dset = h.create_dataset('pipeline_sigmas', data=pipeline_sigmas)
    dset = h.create_dataset('dates', data=dates)
    dset = h.create_dataset('bervs', data=bervs)
    dset = h.create_dataset('airms', data=airms)
    dset = h.create_dataset('drifts', data=drifts)
    filenames = [a.encode('utf8') for a in filenames] # h5py workaround
    dset = h.create_dataset('filelist', data=filenames)
    h.close()
    

if __name__ == "__main__":
        
    if False: #51 Peg
        ccf_filelist = glob.glob('/Users/mbedell/python/wobble/data/51peg/HARPS*ccf_G2_A.fits')
        d = read_data_from_fits(ccf_filelist)
        hdffile = '../data/51peg_e2ds.hdf5'
        write_data(*d, ccf_filelist, hdffile)
        
    if False: #Barnard's Star
        ccf_filelist = glob.glob('/Users/mbedell/python/wobble/data/barnards/HARPS*ccf_M2_A.fits')
        
        if False: # check for missing wavelength files
            missing_files = missing_wavelength_files(ccf_filelist)
            np.savetxt('missing_files.txt', missing_files, fmt='%s')
            print('{0} missing wavelength files for Barnard\'s Star'.format(len(missing_files)))
            
        d = read_data_from_fits(ccf_filelist)
        hdffile = '../data/barnards_e2ds.hdf5'
        write_data(*d, ccf_filelist, hdffile)     
        
    if False: # HD189733
        ccf_filelist = glob.glob('/Users/mbedell/python/wobble/data/HD189733/HARPS*ccf_*_A.fits')
        if False: # check for missing wavelength files
            missing_files = missing_wavelength_files(ccf_filelist)
            np.savetxt('missing_files.txt', missing_files, fmt='%s')
            print('{0} missing wavelength files'.format(len(missing_files)))
        d = read_data_from_fits(ccf_filelist)
        hdffile = '../data/HD189733_e2ds.hdf5'
        write_data(*d, ccf_filelist, hdffile)
        
    if False: # telluric standard
        e2ds_filelist = glob.glob('/Users/mbedell/python/wobble/data/telluric/HARPS*e2ds_A.fits')
        if True: # check for missing wavelength files
            missing_files = missing_wavelength_files(e2ds_filelist)
            np.savetxt('missing_files.txt', missing_files, fmt='%s')
            print('{0} missing wavelength files'.format(len(missing_files)))
        d = read_data_from_fits(e2ds_filelist, e2ds=True)
        hdffile = '../data/telluric_e2ds.hdf5'
        write_data(*d, e2ds_filelist, hdffile)

    if False: # beta hyi
        #ccf_filelist = glob.glob('/mnt/ceph/users/mbedell/wobble/betahyi/HARPS*ccf_*_A.fits')
        ccf_filelist = glob.glob('/Users/mbedell/python/wobble/data/betahyi_20050907/HARPS*ccf_G2_A.fits') 
        if False: # check for missing wavelength files
            missing_files = missing_wavelength_files(ccf_filelist)
            np.savetxt('missing_files.txt', missing_files, fmt='%s')
            print('{0} missing wavelength files'.format(len(missing_files)))
        d = read_data_from_fits(ccf_filelist)
        #hdffile = '/mnt/ceph/users/mbedell/wobble/betahyi_e2ds.hdf5'
        hdffile = '/Users/mbedell/python/wobble/data/betahyi_20050907_e2ds.hdf5'
        write_data(*d, ccf_filelist, hdffile)

    if True: #BP's star
        ccf_filelist = glob.glob('/Users/mbedell/python/video/data/*/HARPN*ccf_K5_A.fits')
        d = read_data_from_fits(ccf_filelist, instrument='HARPS-N')
        hdffile = '/Users/mbedell/python/video/data/video_e2ds.hdf5'
        write_data(*d, ccf_filelist, hdffile)