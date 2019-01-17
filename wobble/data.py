import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt

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
    log_flux : `bool` (default `True`)
        Determines whether fitting will happen using logarithmic flux (default) 
        or linear flux.
    """
    def __init__(self, filename, filepath='', 
                    orders = None, 
                    epochs = None,
                    min_flux = 1.,
                    max_norm_flux = 2.,
                    padding = 2,
                    min_snr = 5.,
                    log_flux = True,
                    **kwargs):
        origin_file = filepath+filename
        self.read_data(origin_file, orders=orders, epochs=epochs)
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
        self.continuum_normalize(**kwargs) 
        
        # HACK - optionally un-log it:
        if not log_flux:
            self.ys = np.exp(self.ys)
            self.ivars = self.flux_ivars
                
        # mask out high pixels:
        for r in range(self.R):
            bad = self.ys[r] > max_norm_flux
            self.ys[r][bad] = 1.
            for pad in range(padding): # mask out neighbors of high pixels
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.
            
    def read_data(self, origin_file, orders = None, epochs = None):
        """Read origin file and set up data attributes from it"""
        # TODO: add asserts to check data are finite, no NaNs, non-negative ivars, etc
        with h5py.File(origin_file) as f:
            self.origin_file = origin_file
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
            self.epoch_groups = [list(np.arange(self.N))]
            self.fluxes = [f['data'][i][self.epochs,:] for i in orders]
            self.xs = [np.log(f['xs'][i][self.epochs,:]) for i in orders]
            self.flux_ivars = [f['ivars'][i][self.epochs,:] for i in orders] # ivars for linear fluxes
            self.pipeline_rvs = np.copy(f['pipeline_rvs'])[self.epochs]
            self.pipeline_sigmas = np.copy(f['pipeline_sigmas'])[self.epochs]
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

        
    def continuum_normalize(self, plot_continuum=False, plot_dir='../results/', **kwargs):
        """Continuum-normalize all spectra using a polynomial fit. Takes kwargs of utils.fit_continuum"""
        for r in range(self.R):
            for n in range(self.N):
                try:
                    fit = fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n], **kwargs)
                    if plot_continuum:
                        fig, ax = plt.subplots(1, 1, figsize=(8,5))
                        ax.scatter(self.xs[r][n], self.ys[r][n], marker=".", alpha=0.5, c='k', s=40)
                        mask = self.ivars[r][n] <= 1.e-8
                        ax.scatter(self.xs[r][n][mask], self.ys[r][n][mask], marker=".", alpha=1., c='white', s=20)                        
                        ax.plot(self.xs[r][n], fit)
                        fig.savefig(plot_dir+'continuum_o{0}_e{1}.png'.format(r, n))
                        plt.close(fig)
                    self.ys[r][n] -= fit
                except:
                    print("ERROR: Data: order {0}, epoch {1} could not be continuum normalized!".format(r,n))
                    
    def append(self, data2):
        """Append another dataset to the current one(s)."""
        assert self.R == data2.R, "ERROR: Number of orders must be the same."
        for attr in ['dates', 'bervs', 'pipeline_rvs', 'pipeline_sigmas', 
                        'airms', 'drifts', 'filelist', 'origin_file']:
            setattr(self, attr, np.append(getattr(self, attr), getattr(data2, attr)))
        for attr in ['fluxes', 'xs', 'flux_ivars', 'ivars', 'ys']:
            attr1 = getattr(self, attr)
            attr2 = getattr(data2, attr)
            full_attr = [np.append(attr1[i], attr2[i], axis=0) for i in range(self.R)]
            setattr(self, attr, full_attr)
        self.epochs = [self.epochs, data2.epochs] # this is a hack that needs to be fixed
        self.epoch_groups.append((self.N + data2.epochs))
        self.N = self.N + data2.N
                