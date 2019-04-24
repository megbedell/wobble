import numpy as np
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
    elif instrument == 'ESPRESSO':
        M = 9141
        R = 170
    else:
        print("instrument not recognized. valid options are: HARPS, HARPS-N, ESPRESSO")
        assert False
    return M, R
    
def read_spec_2d(spec_file):
    '''Read an ESPRESSO 2D spectrum file from the ESO pipeline

    Parameters
    ----------
    spec_file : string
    name of the fits file with the data (e2ds format)
    
    Returns
    -------
    wave : np.ndarray (shape n_orders x 4096)
    wavelength (in Angstroms)
    flux : np.ndarray (shape n_orders x 4096)
    flux value 
    '''
    sp = fits.open(spec_file)
    flux = sp[1].data
    wave_file = str.replace(spec_file, 'S2D', 'WAVE_MATRIX')
    try:
        ww = fits.open(wave_file)
        wave = ww[1].data
    except:
        print("Wavelength solution file {0} not found!".format(wave_file))
        return
    return wave, flux
    
def read_snr(filename):
    '''Parse SNR from header of an ESPRESSO file

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
    n_orders = dimensions('ESPRESSO')[1]
    snr = np.arange(n_orders, dtype=np.float)
    for i in np.nditer(snr, op_flags=['readwrite']):
        i[...] = header['HIERARCH ESO QC ORDER{0} SNR'.format(str(int(i)+1))]
    return snr

def read_data_from_fits(filelist, e2ds=False):
    '''Parses a list of ESPRESSO CCF files.
    
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
    M, R = dimensions('ESPRESSO')
    data = [np.zeros((N,M)) for r in range(R)]
    ivars = [np.zeros((N,M)) for r in range(R)]
    xs = [np.zeros((N,M)) for r in range(R)]
    empty = np.array([], dtype=int)
    pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n,f in enumerate(filelist):
        sp = fits.open(f)
        if not e2ds:
            try:
                pipeline_rvs[n] = sp[0].header['HIERARCH ESO QC CCF RV'] * 1.e3 # m/s
                pipeline_sigmas[n] = sp[0].header['HIERARCH ESO QC CCF RV ERROR'] * 1.e3 # m/s
                drifts[n] = sp[0].header['HIERARCH ESO QC DRIFT DET0 MEAN']  
            except KeyError:
                print("WARNING: {0} does not appear to be a stellar CCF file. Skipping this one.".format(f))
                empty = np.append(empty, n)
                continue
        dates[n] = sp[0].header['HIERARCH ESO QC BJD']        
        bervs[n] = sp[0].header['HIERARCH ESO QC BERV'] * 1.e3 # m/s
        aa = np.zeros(4) + np.nan
        for t in range(1,5): # loop over telescopes
            try:
                aa[t] = sp[0].header['HIERARCH ESO TEL{0} AIRM START'.format(t)]
            except:
                continue
        airms[n] = np.nanmedian(aa)         
        
        spec_file = str.replace(f, 'CCF', 'S2D') 
        try:
            wave, spec = read_spec_2d(spec_file)
        except:
            empty = np.append(empty, n)
            continue
        snrs = read_snr(f) # HACK
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
    
    
def write_data(data, ivars, xs, pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts, filenames, hdffile):
    '''Write processed ESPRESSO data to HDF5 file. 
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

    if True: # test
        ccf_filelist = glob.glob('/mnt/home/mbedell/python/wobble/data/toi411/*CCF_A.fits')
        d = read_data_from_fits(ccf_filelist)
        hdffile = '/mnt/home/mbedell/python/wobble/data/toi411.hdf5'
        write_data(*d, ccf_filelist, hdffile)