import numpy as np
from scipy.interpolate import interp1d
import h5py
import math
import glob
from astropy.io import fits

from make_data import write_data


def read_data(filelist):
    N = len(filelist)  # number of epochs 
    M = 4021 # pixels per order
    R = 23 # orders
    data = [np.zeros((N,M)) for r in range(R)]
    ivars = [np.zeros((N,M)) for r in range(R)]
    xs = [np.zeros((N,M)) for r in range(R)]
    dates, airm = np.zeros(N), np.zeros(N)
    for n,f in enumerate(filelist):
        sp = fits.open(f)
        airm[n] = sp[0].header['AIRMASS']
        dates[n] = sp[0].header['MJD']
        spec = np.copy(sp[0].data)
        errs = np.copy(sp[1].data)
        waves = np.copy(sp[2].data)
        for r in range(R):
            data[r][n,:] = spec[r,:]
            ivars[r][n,:] = 1./errs[r,:]**2
            xs[r][n,:] = waves[r,:]
            
    drift = np.zeros(N)        
    d = np.genfromtxt('/Users/mbedell/python/wobble/data/keckrv_bedell/KIC7108883.txt', usecols=(3,4), skip_header=1) # hack
    bervs = d[:,0] # m/s
    dates = d[:,1] + 2450000.
    #true_rvs = np.genfromtxt('/Users/mbedell/python/wobble/data/keckrv_bedell/kic4253860a_unwrapped_rv.csv', usecols=(1), 
    #        skip_header=1, delimiter=',') # hackity hack hack
    true_rvs = np.genfromtxt('/Users/mbedell/python/wobble/data/keckrv_bedell/kic7108883_predicted_rv.csv', usecols=(1), delimiter=',')
    true_rvs *= 1.e3 # m/s
    true_sigmas = np.zeros_like(true_rvs) # HACK
    return data, ivars, xs, true_rvs, true_sigmas, dates, bervs, airm, drift
    

if __name__ == "__main__":
   
    data_dir = "/Users/mbedell/python/wobble/data/keckrv_bedell/"
    starname = 'KIC7108883'
    filelist = glob.glob(data_dir+starname+'_b*.fits')
    data, ivars, xs, pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts = read_data(filelist)

    hdffile = '../data/'+starname+'.hdf5'
    write_data(data, ivars, xs, pipeline_rvs, pipeline_sigmas, dates, bervs, airms, drifts, filelist, hdffile)