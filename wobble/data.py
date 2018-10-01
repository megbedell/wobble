import numpy as np
import h5py
import pdb

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
        self.read_data(orders=orders, epochs=epochs)
        
        orders = np.asarray(self.orders)
        epochs = np.asarray(self.epochs)
        min_snr = 10.
        chis = np.asarray(self.fluxes) * np.sqrt(np.asarray(self.ivars))
        snrs_by_epoch = np.sqrt(np.nanmean(chis, axis=(0,2)))
        epochs_to_cut = snrs_by_epoch < min_snr
        if np.sum(epochs_to_cut) > 0:
            epochs = epochs[~epochs_to_cut]
            self.read_data(orders=orders, epochs=epochs) # overwrite with new data
        chis = np.asarray(self.fluxes) * np.sqrt(np.asarray(self.ivars))
        snrs_by_order = np.sqrt(np.nanmean(chis, axis=(1,2)))
        orders_to_cut = snrs_by_order < min_snr
        if np.sum(orders_to_cut) > 0:
            orders = orders[~orders_to_cut]
            self.read_data(orders=orders, epochs=epochs) # overwrite with new data
                           
        # mask out low pixels:
        for r in range(self.R):
            bad = self.fluxes[r] < min_flux
            self.fluxes[r][bad] = min_flux
            for pad in range(padding): # mask out neighbors of low pixels
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.
            
        # log and normalize:
        self.ys = np.log(self.fluxes) 
        self.ivars = [self.fluxes[r]**2 * self.ivars[r] for r in range(self.R)]
        self.continuum_normalize() 
                
        # mask out high pixels:
        for r in range(self.R):
            bad = self.ys[r] > max_norm_flux
            self.ys[r][bad] = 1.
            for pad in range(padding): # mask out neighbors of high pixels
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.
            
    def read_data(self, orders = None, epochs = None):
        """Read origin file and set up data attributes from it"""
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
            self.fluxes = [f['data'][i][self.epochs,:] for i in orders]
            self.xs = [np.log(f['xs'][i][self.epochs,:]) for i in orders]
            self.ivars = [f['ivars'][i][self.epochs,:] for i in orders]
            self.pipeline_rvs = np.copy(f['pipeline_rvs'])[self.epochs]
            self.dates = np.copy(f['dates'])[self.epochs]
            self.bervs = np.copy(f['bervs'])[self.epochs]
            self.drifts = np.copy(f['drifts'])[self.epochs]
            self.airms = np.copy(f['airms'])[self.epochs]
            self.R = len(orders) # number of orders
            self.orders = orders # indices of orders in origin_file

        
    def continuum_normalize(self, **kwargs):
        """Continuum-normalize all spectra using a polynomial fit. Takes kwargs of utils.fit_continuum"""
        for r in range(self.R):
            for n in range(self.N):
                try:
                    self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n], **kwargs)
                except:
                    print("ERROR: Data: order {0}, epoch {1} could not be continuum normalized!".format(r,n))
                