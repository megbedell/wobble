import numpy as np
import h5py

from .utils import fit_continuum

class Data(object):
    """
    The data object: contains the spectra and associated data.
    All objects in `data` are numpy arrays or lists of arrays.
    Includes all orders and epochs.
    
    Parameters
    ----------
    filename : `str`
        Name of HDF5 file storing the data.
    filepath : `str` (default `../data/`)
        Path to append to filename.
    orders : `list` (default `None`)
        Indices of spectral orders to read in. If `None` include all. 
        Even if it's only one index, it must be a list.
    epochs : `int` or `list` (default `None`)
        Indices of epochs to read in. If `None` include all.
    min_flux : `float` (default `1.`)
        Flux in counts/pixel below which a pixel is masked out.
    max_norm_flux : `float` (default `2.`)
        Flux in normalized counts/pixel above which a pixel is masked out.
    padding : `int` (default `3`)
        Number of pixels to additionally mask on either side of a bad high pixel.
    """
    def __init__(self, filename, filepath='../data/', 
                    orders = None, 
                    epochs = None,
                    min_flux = 1.,
                    max_norm_flux = 2.,
                    padding = 3):
        self.origin_file = filepath+filename
        with h5py.File(self.origin_file) as f:
            if orders is None:
                orders = np.arange(len(f['data']))
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
            
        self.R = len(orders) # number of orders
        self.orders = orders # indices of orders in origin_file
                    
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
            for pad in range(padding): # mask out neighbors of high pixels
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.

        
    def continuum_normalize(self, **kwargs):
        """Continuum-normalize all spectra using a polynomial fit. Takes kwargs of utils.fit_continuum"""
        for r in range(self.R):
            for n in range(self.N):
                try:
                    self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n], **kwargs)
                except:
                    print("ERROR: Data: order {0}, epoch {1} could not be continuum normalized!".format(r,n))
                