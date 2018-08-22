import numpy as np
import h5py

from .utils import fit_continuum

class Data(object):
    """
    The data object: contains the spectra and associated data.
    All objects in `data` are numpy arrays or lists of arrays.
    Includes all orders and epochs.
    """
    def __init__(self, filename, filepath='../data/', 
                    N = 0, orders = [30], min_flux = 1.,
                    max_norm_flux = 2.,
                    mask_epochs = None, padding = 1):
        self.R = len(orders) # number of orders to be analyzed
        self.orders = orders
        self.origin_file = filepath+filename
        with h5py.File(self.origin_file) as f:
            if N < 1:
                self.N = len(f['dates']) # all epochs
            else:
                self.N = N
            self.ys = [f['data'][i][:self.N,:] for i in orders]
            self.xs = [np.log(f['xs'][i][:self.N,:]) for i in orders]
            self.ivars = [f['ivars'][i][:self.N,:] for i in orders]
            self.pipeline_rvs = np.copy(f['pipeline_rvs'])[:self.N]
            self.dates = np.copy(f['dates'])[:self.N]
            self.bervs = np.copy(f['bervs'])[:self.N]
            self.drifts = np.copy(f['drifts'])[:self.N]
            self.airms = np.copy(f['airms'])[:self.N]
            
        # mask out low pixels:
        for r in range(self.R):
            bad = self.ys[r] < min_flux
            self.ys[r][bad] = min_flux
            for pad in range(padding):
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.
            
        # mask out bad epochs:
        self.epoch_mask = [True for n in range(self.N)]
        if mask_epochs is not None:
            for n in mask_epochs:
                self.epoch_mask[n] = False

        # log and normalize:
        self.ys = np.log(self.ys) 
        self.continuum_normalize() 
        
        # mask out high pixels:
        for r in range(self.R):
            bad = self.ys[r] > max_norm_flux
            self.ys[r][bad] = 1.
            for pad in range(padding):
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.
        
    def continuum_normalize(self):
        for r in range(self.R):
            for n in range(self.N):
                self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n])
                