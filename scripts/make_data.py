import numpy as np
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
from harps_hacks import read_harps
import h5py
import math
from astropy.io import fits
import shutil
import glob
import os

def dimensions(instrument):
    if instrument == 'HARPS':
        M = 4096 # pixels per order
        R = 72 # orders
    else:
        print("instrument not recognized. valid options are: HARPS")
        return
    return M, R

def read_data_from_fits(filelist, e2ds=False):
    # input : a list of CCF filenames
    N = len(filelist)  # number of epochs    
    M, R = dimensions('HARPS')
    data = [np.zeros((N,M)) for r in range(R)]
    ivars = [np.zeros((N,M)) for r in range(R)]
    xs = [np.zeros((N,M)) for r in range(R)]
    empty = np.array([], dtype=int)
    pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n,f in enumerate(filelist):
        sp = fits.open(f)
        if not e2ds:
            pipeline_rvs[n] = sp[0].header['HIERARCH ESO DRS CCF RVC'] * 1.e3 # m/s
            pipeline_sigs[n] = sp[0].header['HIERARCH ESO DRS CCF NOISE'] * 1.e3 # m/s
            drifts[n] = sp[0].header['HIERARCH ESO DRS DRIFT SPE RV']  
        dates[n] = sp[0].header['HIERARCH ESO DRS BJD']        
        bervs[n] = sp[0].header['HIERARCH ESO DRS BERV'] * 1.e3 # m/s
        airms[n] = sp[0].header['HIERARCH ESO TEL AIRM START']
        
        spec_file = str.replace(f, 'ccf_G2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_M2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_K5', 'e2ds') 
        try:
            wave, spec = read_harps.read_spec_2d(spec_file)
        except:
            empty = np.append(empty, n)
            continue
        snrs = read_harps.read_snr(f) # HACK
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
    pipeline_sigs = np.delete(pipeline_sigs, empty)
    dates = np.delete(dates, empty)
    bervs = np.delete(bervs, empty)
    airms = np.delete(airms, empty)
    drifts = np.delete(drifts, empty)
    
    # re-introduce BERVs to HARPS results:
    pipeline_rvs -= bervs  
    pipeline_rvs -= np.mean(pipeline_rvs)
        
    return data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts
    
def savfile_to_filelist(savfile, destination_dir='../data/'):
    # copies CCF + E2DS files to destination_dir and returns a list of the CCFs
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
    
    
def write_data(data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts, filenames, hdffile):
    h = h5py.File(hdffile, 'w')
    dset = h.create_dataset('data', data=data)
    dset = h.create_dataset('ivars', data=ivars)
    dset = h.create_dataset('xs', data=xs)
    dset = h.create_dataset('pipeline_rvs', data=pipeline_rvs)
    dset = h.create_dataset('pipeline_sigs', data=pipeline_sigs)
    dset = h.create_dataset('dates', data=dates)
    dset = h.create_dataset('bervs', data=bervs)
    dset = h.create_dataset('airms', data=airms)
    dset = h.create_dataset('drifts', data=drifts)
    filenames = [a.encode('utf8') for a in filenames] # h5py workaround
    dset = h.create_dataset('filelist', data=filenames)
    h.close()
    

if __name__ == "__main__":
        
    if True: #51 Peg
        ccf_filelist = glob.glob('/Users/mbedell/python/wobble/data/51peg/HARPS*ccf_G2_A.fits')
        data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts = read_data_from_fits(ccf_filelist)
        hdffile = '../data/51peg_e2ds.hdf5'
        write_data(data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts, ccf_filelist, hdffile)
        
    if True: #Barnard's Star
        ccf_filelist = glob.glob('/Users/mbedell/python/wobble/data/barnards/HARPS*ccf_M2_A.fits')
        
        if False: # check for missing wavelength files
            missing_files = missing_wavelength_files(ccf_filelist)
            np.savetxt('missing_files.txt', missing_files, fmt='%s')
            print('{0} missing wavelength files for Barnard\'s Star'.format(len(missing_files)))
            
        data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts = read_data_from_fits(ccf_filelist)
        hdffile = '../data/barnards_e2ds.hdf5'
        write_data(data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts, ccf_filelist, hdffile)     
        
    if True: # HD189733
        ccf_filelist = glob.glob('/Users/mbedell/python/wobble/data/HD189733/HARPS*ccf_G2_A.fits')
        if False: # check for missing wavelength files
            missing_files = missing_wavelength_files(ccf_filelist)
            np.savetxt('missing_files.txt', missing_files, fmt='%s')
            print('{0} missing wavelength files'.format(len(missing_files)))
        data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts = read_data_from_fits(ccf_filelist)
        hdffile = '../data/HD189733_e2ds.hdf5'
        write_data(data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts, ccf_filelist, hdffile)
        
    if False: # telluric standard
        e2ds_filelist = glob.glob('/Users/mbedell/python/wobble/data/telluric/HARPS*e2ds_A.fits')
        if True: # check for missing wavelength files
            missing_files = missing_wavelength_files(e2ds_filelist)
            np.savetxt('missing_files.txt', missing_files, fmt='%s')
            print('{0} missing wavelength files'.format(len(missing_files)))
        data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts = read_data_from_fits(e2ds_filelist, e2ds=True)
        hdffile = '../data/telluric_e2ds.hdf5'
        write_data(data, ivars, xs, pipeline_rvs, pipeline_sigs, dates, bervs, airms, drifts, e2ds_filelist, hdffile)
