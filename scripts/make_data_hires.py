import numpy as np
from scipy.interpolate import interp1d
import h5py
import math
import glob
from astropy.io import fits
import pdb

from make_data import write_data


def read_data(filelist):
    """Takes a list of blue chip files"""
    N = len(filelist)  # number of epochs 
    M = 4021 # pixels per order
    R = 23 + 16 + 10 # orders (b + r + i)
    data = [np.zeros((N,M)) for r in range(R)]
    ivars = [np.zeros((N,M)) for r in range(R)]
    xs = [np.zeros((N,M)) for r in range(R)]
    dates, airm = np.zeros(N), np.zeros(N)
    for n,f in enumerate(filelist):
        sp = fits.open(f)
        airm[n] = sp[0].header['AIRMASS']
        dates[n] = float(sp[0].header['MJD']) + 2450000.5
        spec = np.copy(sp[0].data)
        errs = np.copy(sp[1].data)
        waves = np.copy(sp[2].data)
        sp = fits.open(f.replace('_b', '_r')) # add red
        spec = np.concatenate((spec, sp[0].data))
        errs = np.concatenate((errs, sp[1].data))
        waves = np.concatenate((waves, sp[2].data))
        sp = fits.open(f.replace('_b', '_i')) # add IR
        spec = np.concatenate((spec, sp[0].data))
        errs = np.concatenate((errs, sp[1].data))
        waves = np.concatenate((waves, sp[2].data))
        for r in range(R):
            data[r][n,:] = spec[r,:]
            ivars[r][n,:] = 1./errs[r,:]**2
            xs[r][n,:] = waves[r,:]
            

    return data, ivars, xs, dates, airm
    
def make_data(data_dir, starname):
    filelist = glob.glob(data_dir+starname+'_b*.fits')
    filelist.sort() # CHECK THIS if N > 10!!
    data, ivars, xs, dates, airms = read_data(filelist)
    
    drifts = np.zeros_like(dates)  # hack      
    d = np.genfromtxt(data_dir+'{0}.txt'.format(starname), usecols=(3,4), skip_header=1) # brittle to naming & ordering
    bervs = d[:,0] # m/s
    #dates = d[:,1] + 2450000.5 # MJD to BJD
    
    try:
        true_rvs = np.genfromtxt(data_dir+'{0}_predicted_rv.csv'.format(starname), 
                                 usecols=(1), delimiter=',') # brittle
    except:
        true_rvs = np.genfromtxt(data_dir+'{0}a_predicted_rv.csv'.format(starname), 
                                 usecols=(1), delimiter=',') # brittle        
    true_rvs *= 1.e3 # m/s
    true_rvs -= bervs # turn into prediction of OBSERVED rvs
    true_sigmas = np.zeros_like(true_rvs) # HACK
    hdffile = '../data/'+starname+'.hdf5'
    write_data(data, ivars, xs, true_rvs, true_sigmas, dates, bervs, airms, drifts, filelist, hdffile)
    
    
    

if __name__ == "__main__":
   
    data_dir = "/Users/mbedell/python/wobble/data/keckrv/"
    make_data(data_dir, 'KIC7108883')
    #make_data(data_dir, 'KIC4253860')
    #make_data(data_dir, 'KIC4269337')
    #make_data(data_dir, 'KIC5370646')
    make_data(data_dir, 'KIC10416779')
    
    
