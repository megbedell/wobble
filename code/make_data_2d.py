import numpy as np
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
from harps_hacks import read_harps, rv_model 
import h5py
import math

def dimensions(instrument):
    if instrument == 'HARPS':
        M = 4096 # pixels per order
        R = 72 # orders
    else:
        print "instrument not recognized. valid options are: HARPS"
        return
    return M, R

def read_data_from_fits(filelist):
    # input : a list of CCF filenames
    N = len(filelist)  # number of epochs    
    M, R = dimensions('HARPS')
    data = [np.zeros((N,M)) for r in range(R)]
    ivars = [np.zeros((N,M)) for r in range(R)]
    xs = [np.zeros((N,M)) for r in range(R)]
    empty = np.array([], dtype=int)
    pipeline_rvs, dates, bervs, airms, drifts = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n,f in enumerate(filelist):
        sp = fits.open(f)
        pipeline_rvs[n] = sp[0].header['HIERARCH ESO DRS CCF RVC'] * 1.e3 # m/s
        dates[n] = sp[0].header['HIERARCH ESO DRS BJD']        
        bervs[n] = sp[0].header['HIERARCH ESO DRS BERV'] * 1.e3 # m/s
        airms[n] = sp[0].header['HIERARCH ESO TEL AIRM START']
        drifts[n] = sp[0].header['HIERARCH ESO DRS DRIFT SPE RV']  
        
        spec_file = str.replace(f, 'ccf_G2', 'e2ds')      
        try:
            wave, spec = read_harps.read_spec_2d(spec_file)
        except:
            empty = np.append(empty, n)
            continue
        snrs = read_harps.read_snr(f) # HACK
        # save stuff
        for r in range(R):
            data[r][n,:] = spec[r,:]
            ivars[r][n,:] = np.zeros(M) + snrs[r]**2
            xs[r][n,:] = wave[r,:]  
            
    for r in range(R):
        data[r] = np.delete(data[r], empty, axis=0)
        ivars[r] = np.delete(ivars[r], empty, axis=0)
        xs[r] = np.delete(xs[r], empty, axis=0)
    
    pipeline_rvs = np.delete(pipeline_rvs, empty)
    bervs = np.delete(bervs, empty)
    dates = np.delete(dates, empty)
    airms = np.delete(airms, empty)
    drifts = np.delete(drifts, empty)
    
    pipeline_rvs += bervs  
    pipeline_rvs -= np.mean(pipeline_rvs)
    
    return data, ivars, xs, pipeline_rvs, dates, bervs, airms, drifts    

def read_data_from_savfile(savfile):
    s = readsav(savfile)
    N = len(s.files)  # number of epochs    
    M, R = dimensions('HARPS')
    data = [np.zeros((N,M)) for r in range(R)]
    ivars = [np.zeros((N,M)) for r in range(R)]
    xs = [np.zeros((N,M)) for r in range(R)]
    empty = np.array([], dtype=int)
    for n,(f,b,snr) in enumerate(zip(s.files, s.berv, s.snr)):
            # read in the spectrum
            spec_file = str.replace(f, 'ccf_G2', 'e2ds')
            try:
                wave, spec = read_harps.read_spec_2d(spec_file)
            except:
                empty = np.append(empty, n)
                continue
            snrs = read_harps.read_snr(f)
            # save stuff
            for r in range(R):
                data[r][n,:] = spec[r,:]
                ivars[r][n,:] = np.zeros(M) + snrs[r]**2
                xs[r][n,:] = wave[r,:]
            
    for r in range(R):
        data[r] = np.delete(data[r], empty, axis=0)
        ivars[r] = np.delete(ivars[r], empty, axis=0)
        xs[r] = np.delete(xs[r], empty, axis=0)
    
    pipeline_rvs = (s.berv + s.rv) * 1.e3  # m/s    
    pipeline_rvs = np.delete(pipeline_rvs, empty)
    pipeline_rvs -= np.mean(pipeline_rvs)
    bervs = np.delete(s.berv, empty) * 1.e3 # m/s
    dates = np.delete(s.date, empty)
    airms = np.delete(s.airm, empty)
    drifts = np.delete(s.drift, empty)
    
    return data, ivars, xs, pipeline_rvs, dates, bervs, airms, drifts
    
def write_data(data, ivars, xs, pipeline_rvs, dates, bervs, airms, drifts, hdffile):
    h = h5py.File(hdffile, 'w')
    dset = h.create_dataset('data', data=data)
    dset = h.create_dataset('ivars', data=ivars)
    dset = h.create_dataset('xs', data=xs)
    dset = h.create_dataset('pipeline_rvs', data=pipeline_rvs)
    dset = h.create_dataset('dates', data=dates)
    dset = h.create_dataset('bervs', data=bervs)
    dset = h.create_dataset('airms', data=airms)
    dset = h.create_dataset('drifts', data=drifts)
    h.close()
    

if __name__ == "__main__":
   
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    savfile = data_dir+'HIP54287_result.dat'
    data, ivars, xs, pipeline_rvs, dates, bervs, airms, drifts = read_data_from_savfile(savfile)

    hdffile = '../data/hip54287_e2ds.hdf5'
    write_data(data, ivars, xs, pipeline_rvs, dates, bervs, airms, drifts, hdffile)

