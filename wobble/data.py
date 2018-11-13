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
    padding : `int` (default `2`)
        Number of pixels to additionally mask on either side of a bad high pixel.
    min_snr : `float` (default `5.`)
        Mean SNR below which which we discard sections of data.
    """
    def __init__(self, filename, filepath='../data/', 
                    orders = None, 
                    epochs = None,
                    min_flux = 1.,
                    max_norm_flux = 2.,
                    padding = 2,
                    min_snr = 5.):
        self.origin_file = filepath+filename
        self.read_data(orders=orders, epochs=epochs)
        self.mask_low_pixels(min_flux=min_flux, padding=padding, min_snr=min_snr)
        
        orders = np.asarray(self.orders)
        snrs_by_order = np.sqrt(np.nanmean(self.ivars, axis=(1,2)))
        orders_to_cut = snrs_by_order < min_snr
        if np.sum(orders_to_cut) > 0:
            print("Data: Dropping orders {0} because they have average SNR < {1:.0f}".format(orders[orders_to_cut], min_snr))
            orders = orders[~orders_to_cut]
            self.read_data(orders=orders, epochs=epochs) # overwrite with new data
            self.mask_low_pixels(min_flux=min_flux, padding=padding, min_snr=min_snr)
        if len(orders) == 0:
            print("All orders failed the quality cuts with min_snr={0:.0f}.".format(min_snr))
            return
        epochs = np.asarray(self.epochs)
        snrs_by_epoch = np.sqrt(np.nanmean(self.ivars, axis=(0,2)))
        epochs_to_cut = snrs_by_epoch < min_snr
        if np.sum(epochs_to_cut) > 0:
            print("Data: Dropping epochs {0} because they have average SNR < {1:.0f}".format(epochs[epochs_to_cut], min_snr))
            epochs = epochs[~epochs_to_cut]
            self.read_data(orders=orders, epochs=epochs) # overwrite with new data
            self.mask_low_pixels(min_flux=min_flux, padding=padding, min_snr=min_snr)
        if len(epochs) == 0:
            print("All epochs failed the quality cuts with min_snr={0:.0f}.".format(min_snr))
            return
            
        # log and normalize:
        self.ys = np.log(self.fluxes) 
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
        # TODO: add asserts to check data are finite, no NaNs, non-negative ivars, etc
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
            self.flux_ivars = [f['ivars'][i][self.epochs,:] for i in orders] # ivars for linear fluxes
            self.pipeline_rvs = np.copy(f['pipeline_rvs'])[self.epochs]
            self.dates = np.copy(f['dates'])[self.epochs]
            self.bervs = np.copy(f['bervs'])[self.epochs]
            self.drifts = np.copy(f['drifts'])[self.epochs]
            self.airms = np.copy(f['airms'])[self.epochs]
            self.filelist = [a.decode('utf8') for a in np.copy(f['filelist'])[self.epochs]]
            self.R = len(orders) # number of orders
            self.orders = orders # indices of orders in origin_file
            self.ivars = [self.fluxes[i]**2 * self.flux_ivars[i] for i in range(self.R)] # ivars for log(fluxes)
    
    def mask_low_pixels(self, min_flux = 1., padding = 2, min_snr = 5.):
        """Set ivars to zero for pixels and edge regions that are bad."""
        # mask out low pixels:
        for r in range(self.R):
            bad = self.fluxes[r] < min_flux
            self.fluxes[r][bad] = min_flux
            for pad in range(padding): # mask out neighbors of low pixels
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.flux_ivars[r][bad] = 0.
            self.ivars[r][bad] = 0.
            
        # find bad regions in masked spectra:
        for r in range(self.R):
            self.trim_bad_edges(r, min_snr=min_snr) # HACK
            
    def trim_bad_edges(self, r, window_width = 128, min_snr = 5.):
        """
        Find edge regions that contain no information and trim them.
        
        Parameters
        ----------
        r : `int`
            order index
        window_width : `int`
            number of pixels to average over for local SNR            
        min_snr : `float`
            SNR threshold below which we discard the data
        """
        for n in range(self.N):
            n_pix = len(self.xs[0][n])
            for window_start in range(n_pix - window_width):
                mean_snr = np.sqrt(np.nanmean(self.ivars[r][n,window_start:window_start+window_width]))
                if mean_snr > min_snr:
                    self.ivars[r][n,:window_start] = 0. # trim everything to left of window
                    break
            for window_start in reversed(range(n_pix - window_width)):
                mean_snr = np.sqrt(np.nanmean(self.ivars[r][n,window_start:window_start+window_width]))
                if mean_snr > min_snr:
                    self.ivars[r][n,window_start+window_width:] = 0. # trim everything to right of window
                    break

        
    def continuum_normalize(self, **kwargs):
        """Continuum-normalize all spectra using a polynomial fit. Takes kwargs of utils.fit_continuum"""
        for r in range(self.R):
            for n in range(self.N):
                try:
                    self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n], **kwargs)
                except:
                    print("ERROR: Data: order {0}, epoch {1} could not be continuum normalized!".format(r,n))
                