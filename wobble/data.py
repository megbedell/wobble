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
                    orders = [30], 
                    epochs = None,
                    min_flux = 1.,
                    max_norm_flux = 2.,
                    padding = 3):
        self.R = len(orders) # number of orders to be analyzed
        self.orders = orders
        self.origin_file = filepath+filename
        with h5py.File(self.origin_file) as f:
            if epochs is None:
                self.N = len(f['dates']) # all epochs
                self.epochs = np.arange(self.N)
            else:
                self.epochs = epochs
                self.N = len(epochs)
                for e in epochs:
                    assert (e >= 0) & (e < len(f['dates'])), \
                        "epoch #{0} is not in datafile {1}".format(e, self.origin_file)
            self.ys = [f['data'][i][self.epochs,:] for i in orders]
            self.xs = [np.log(f['xs'][i][self.epochs,:]) for i in orders]
            self.ivars = [f['ivars'][i][self.epochs,:] for i in orders]
            self.pipeline_rvs = np.copy(f['pipeline_rvs'])[self.epochs]
            self.dates = np.copy(f['dates'])[self.epochs]
            self.bervs = np.copy(f['bervs'])[self.epochs]
            self.drifts = np.copy(f['drifts'])[self.epochs]
            self.airms = np.copy(f['airms'])[self.epochs]
            
        # mask out low pixels:
        for r in range(self.R):
            bad = self.ys[r] < min_flux
            self.ys[r][bad] = min_flux
            for pad in range(padding):
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.

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
                