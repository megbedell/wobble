import numpy as np
import h5py
import matplotlib.pyplot as plt
from itertools import compress
from astropy.io import fits
import pandas as pd

from .utils import fit_continuum

# attributes!
REQUIRED_3D = ['xs', 'ys', 'ivars'] # R-order lists of (N-epoch, M-pixel) arrays
REQUIRED_1D = ['bervs', 'airms'] # N-epoch arrays
OPTIONAL_1D = ['pipeline_rvs', 'pipeline_sigmas', 'dates', 'drifts', 'filelist'] # N-epoch arrays
                    # optional attributes always exist in Data() but may be filled with placeholders.
                    # they do not need to exist at all in an individual Spectrum().

class Data(object):
    """
    The Data object: contains a block of time-series Data 
    and associated data. 
    Includes all orders and epochs. 
    Can be loaded from an HDF5 file using the `filename` keyword, 
    or initialized as an empty dataset and built by appending
    `wobble.Spectrum` objects.
    
    Parameters
    ----------
    filename : string (optional)
        Name of HDF5 file storing the data.
    orders : list (optional)
        Indices of echelle orders to load (if reading from file).
    epochs : list (optional)
        Indices of observations to load (if reading from file).
    
    Attributes
    ----------
    N : `int`
        Number of epochs.
    R : `int`
        Number of echelle orders.
    xs : `list`
        R-order list of (N-epoch, M-pixel) arrays. May be wavelength or ln(wavelength).    
    ys : `list`
        R-order list of (N-epoch, M-pixel) arrays. May be continuum-normalized flux or ln(flux).    
    ivars : `list`
        R-order list of (N-epoch, M-pixel) arrays corresponding to inverse variances for `ys`.
    bervs : `np.ndarray`
        N-epoch array of Barycentric Earth Radial Velocity (m/s).
    airms : `np.ndarray`
        N-epoch array of observation airmasses.
    pipeline_rvs : `np.ndarray`, optional
        Optional N-epoch array of expected RVs.
    pipeline_sigs : `np.ndarray`, optional
        Optional N-epoch array of uncertainties on `pipeline_rvs`.    
    dates : `np.ndarray`, optional
        Optional N-epoch array of observation dates.
    drifts : `np.ndarray`, optional
        Optional N-epoch array of instrumental drifts.
    filelist : `np.ndarray`, optional
        Optional N-epoch array of file locations for each observation.
    epochs : `np.ndarray`
        N-epoch array of indices. In most cases this will be equivalent
        to np.arange(N), but differences will arise when individual epochs have been
        dropped from the original data file in post-read-in processing.
    orders : `list`
        R-order list of echelle orders contained in this Data object. 
        Represents a mapping from the order indices in this object to
        the overall reference indices used in e.g. other data that may be 
        used elsewhere, regularization parameter dictionaries, etc.
    """
    def __init__(self, filename=None, **kwargs):
        self.empty = True
        for attr in REQUIRED_3D:
            setattr(self, attr, [])
        for attr in np.append(REQUIRED_1D, OPTIONAL_1D):
            setattr(self, attr, np.array([]))
        self.epochs = []
        self.N = 0 # number of epochs
        self.R = 0 # number of echelle orders
        
        if filename is not None:
            self.read(filename, **kwargs)
            
    def __repr__(self):
        return "wobble.Data object containing {0} echelle orders & {1} observation epochs.".format(self.R, self.N)
            
    def append(self, sp):
        """
        Append a spectrum.
        
        Parameters
        ----------
        sp : `wobble.Spectrum`
            The spectrum to be appended.
        """
        if self.empty:
            self.orders = sp.orders
            self.R = sp.R
            for attr in REQUIRED_3D:
                setattr(self, attr, getattr(sp, attr))
        else:
            assert np.all(np.array(self.orders) == np.array(sp.orders)), "Echelle orders do not match." 
            for attr in REQUIRED_3D:           
                old = getattr(self, attr)
                new = getattr(sp, attr)
                setattr(self, attr, [np.vstack([old[r], new[r]]) for r in range(self.R)])
            
        for attr in REQUIRED_1D:
            try:
                new = getattr(sp,attr)
            except: # fail with warning
                new = 1.
                print('WARNING: {0} missing; resulting solutions may be non-optimal.'.format(attr))
            setattr(self, attr, np.append(getattr(self,attr), new))
        for attr in OPTIONAL_1D:
            try:
                new = getattr(sp,attr)
            except: # fail silently
                new = 0.
            setattr(self, attr, np.append(getattr(self,attr), new))     
        self.epochs = np.append(self.epochs, self.N)       
        self.N += 1
        self.empty = False
        
    def pop(self, i):
        """
        Remove and return spectrum at epoch index i.
        
        Parameters
        ----------
        i : `int`
            The epoch index of the spectrum to be removed.
        
        Returns
        -------
        sp : `wobble.Spectrum`
            The removed spectrum.
        """
        assert 0 <= i < self.N, "ERROR: invalid index."
        sp = Spectrum()
        sp.R = self.R
        sp.orders = self.orders
        for attr in REQUIRED_3D:
            epoch_to_split = [r[i] for r in getattr(self, attr)]
            epochs_to_keep = [np.delete(r, i, axis=0) for r in getattr(self, attr)]
            setattr(sp, attr, epoch_to_split)
            setattr(self, attr, epochs_to_keep)
        for attr in np.append(np.append(REQUIRED_1D, OPTIONAL_1D), ['epochs']):
            all_epochs = getattr(self, attr)
            setattr(sp, attr, all_epochs[i])
            setattr(self, attr, np.delete(all_epochs, i))
        self.N -= 1
        return sp

    def append_data(self, data2):
        """Append another dataset to the current one(s)."""
        assert self.R == data2.R, "ERROR: Number of orders must be the same."
        for attr in np.append(REQUIRED_1D, OPTIONAL_1D):
            setattr(self, attr, np.append(getattr(self, attr), getattr(data2, attr)))
        for attr in REQUIRED_3D:
            attr1 = getattr(self, attr)
            attr2 = getattr(data2, attr)
            full_attr = [np.append(attr1[i], attr2[i], axis=0) for i in range(self.R)]
            setattr(self, attr, full_attr)
        self.epochs = [self.epochs, data2.epochs] # this is a hack that needs to be fixed
        #self.epoch_groups.append((self.N + data2.epochs)) # inconsistency in the meaning of epochs
        self.N = self.N + data2.N
            
       
    def read(self, filename, orders=None, epochs=None):
        """
        Read from file.
        
        Parameters
        ----------
        filename : `str`
            The filename (including path).
        
        orders : `list` or `None` (default `None`)
            List of echelle order indices to read. If `None`, read all.
        
        epochs : `list` or `None` (default `None`)
            List of observation epoch indices to read. If `None`, read all.
        """
        if not self.empty:
            print("WARNING: overwriting existing contents.")
        # TODO: add asserts to check data are finite, no NaNs, non-negative ivars, etc
        with h5py.File(filename, 'r') as f:
            if orders is None:
                orders = np.arange(len(f['data']))
            self.orders = orders
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
            self.xs = [f['xs'][i][self.epochs,:] for i in orders]
            self.ivars = [f['ivars'][i][self.epochs,:] for i in orders]
            for attr in np.append(REQUIRED_1D, OPTIONAL_1D):
                if not f[attr].dtype.type is np.bytes_:
                    setattr(self, attr, np.copy(f[attr])[self.epochs])
                else:
                    strings = [a.decode('utf8') for a in np.copy(f[attr])[self.epochs]]
                    setattr(self, attr, strings)
            self.R = len(orders) # number of orders
            #self.epoch_groups = [list(np.arange(self.N))] # indices into this object with data from this origin file
        self.empty = False
        
    def write(self, filename):
        """
        Write the currently loaded object to file.
    
        Parameters
        ----------
        filename : `str`
            The filename (including path).
        """
        with h5py.File(filename, 'a') as f:
            dset = f.create_dataset('data', data=self.ys)
            dset = f.create_dataset('ivars', data=self.ivars)
            dset = f.create_dataset('xs', data=self.xs)
            for attr in np.append(REQUIRED_1D, OPTIONAL_1D):
                if not getattr(self, attr).dtype.type is np.str_:
                    dset = f.create_dataset(attr, data=getattr(self, attr))
                else:
                    strings = [a.encode('utf8') for a in getattr(self, attr)] # h5py workaround
                    dset = f.create_dataset(attr, data=strings)    
                    
    def drop_bad_orders(self, min_snr=5.):
        """
        Automatically drop all echelle orders with average SNR (over all epochs) < min_snr.
        """
        try: 
            orders = np.asarray(self.orders)
        except:
            orders = np.arange(self.R)
        snrs_by_order = [np.sqrt(np.nanmean(i)) for i in self.ivars]
        bad_order_mask = np.array(snrs_by_order) < min_snr
        if np.sum(bad_order_mask) > 0:
            print("Data: Dropping orders {0} because they have average SNR < {1:.0f}".format(orders[bad_order_mask], min_snr))
            self.delete_orders(bad_order_mask)
        if self.R == 0:
            print("All orders failed the quality cuts with min_snr={0:.0f}.".format(min_snr))
            
    def drop_bad_epochs(self, min_snr=5.):
        """
        Automatically drop all epochs with average SNR (over all orders) < min_snr.
        """
        try:
            epochs = np.asarray(self.epochs)
        except:
            epochs = np.arange(self.N)
        snrs_by_epoch = np.sqrt(np.nanmean(self.ivars, axis=(0,2)))
        epochs_to_cut = snrs_by_epoch < min_snr
        if np.sum(epochs_to_cut) > 0:
            print("Data: Dropping epochs {0} because they have average SNR < {1:.0f}".format(epochs[epochs_to_cut], min_snr))
            epochs = epochs[~epochs_to_cut]
            for attr in REQUIRED_3D:
                old = getattr(self, attr)
                setattr(self, attr, [o[~epochs_to_cut] for o in old]) # might fail if self.N = 1
            for attr in np.append(REQUIRED_1D, OPTIONAL_1D):
                setattr(self, attr, getattr(self,attr)[~epochs_to_cut])
            self.epochs = epochs
            self.N = len(epochs)
        if self.N == 0:
            print("All epochs failed the quality cuts with min_snr={0:.0f}.".format(min_snr))
            return  
        
    def delete_orders(self, bad_order_mask):
        """
        Take an R-order length boolean mask & drop all orders marked True.
        """
        good_order_mask = ~bad_order_mask
        for attr in REQUIRED_3D:
            old = getattr(self, attr)
            new = list(compress(old, good_order_mask))
            setattr(self, attr, new)
        self.orders = self.orders[good_order_mask]
        self.R = len(self.orders)
        
class Spectrum(object):
    """
    An individual spectrum, including all orders at one
    epoch. Can be initialized by passing data as function 
    arguments or by calling a method to read from a 
    known file format.
    """
    def __init__(self, *arg, **kwarg):
        self.empty = True # flag indicating object contains no data
        if len(arg) > 0:
            self.populate(*arg, **kwarg) 
            
    def __repr__(self):
        if self.empty:
            return "An empty wobble.Spectrum object."
        else:
            return "A wobble.Spectrum object containing data loaded from: {0}".format(self.filelist)
            
    def populate(self, xs, ys, ivars, **kwargs):
        """
        Takes data and saves it to the object.
        
        Parameters
        ----------
        xs : list of np.ndarrays
            R-order list of (M-pixel) arrays with x-values; 
            may be wavelengths, or ln(wavelengths). Number of
            pixels may be different for different orders.
        ys : list of np.ndarrays
            Same shape as `xs`; contains y-values, which may
            be fluxes or ln(fluxes).
        ivars : list of np.ndarrays
            Same shape as `ys`; contains inverse variance 
            estimates on `ys`.
        **kwargs : dict
            Metadata to be associated with this observation. 
            See `wobble.Data()` documentation for a list
            of recommended keywords.
        """
        if not self.empty:
            print("WARNING: overwriting existing contents.")
        self.R = len(xs) # number of echelle orders
        self.orders = np.arange(self.R) # will be overwritten if kwargs has orders
        self.filelist = 'input arguments' # will be overwritten if kwargs has filelist
        self.xs = xs
        self.ys = ys
        self.ivars = ivars
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.empty = False      
        
    def continuum_normalize(self, plot_continuum=False, plot_dir='../results/', **kwargs):
        """
        Continuum-normalize all orders using polynomial fits.
        
        Parameters
        ----------
        plot_continuum : bool, optional (default `False`)
            If `True`, generate and save a plot of the continuum fit for each order.
        plot_dir : string, optional (default `../results/`)
            Directory in which to save generated plots.
        **kwargs : dict
            Passed to `wobble.utils.fit_continuum`.
        """
        for r in range(self.R):
            try:
                fit = fit_continuum(self.xs[r], self.ys[r], self.ivars[r], **kwargs)
            except:
                print("WARNING: Data: order {0} could not be continuum normalized. Setting to zero.".format(r))
                self.ys[r] = np.zeros_like(self.ys[r])
                self.ivars[r] = np.zeros_like(self.ivars[r])
                continue
            if plot_continuum:
                fig, ax = plt.subplots(1, 1, figsize=(8,5))
                ax.scatter(self.xs[r], self.ys[r], marker=".", alpha=0.5, c='k', s=40)
                mask = self.ivars[r] <= 1.e-8
                ax.scatter(self.xs[r][mask], self.ys[r][mask], marker=".", alpha=1., c='white', s=20)                        
                ax.plot(self.xs[r], fit)
                fig.savefig(plot_dir+'continuum_o{0}.png'.format(r))
                plt.close(fig)
            self.ys[r] -= fit
            
    def mask_low_pixels(self, min_flux = 1., padding = 2):
        """Set ivars to zero for pixels that fall below some minimum value (e.g. negative flux)."""
        for r in range(self.R):
            bad = np.logical_or(self.ys[r] < min_flux, ~np.isfinite(self.ys[r]))
            self.ys[r][bad] = min_flux
            for pad in range(padding): # mask out neighbors of low pixels
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.
            
    def mask_high_pixels(self, max_flux = 2., padding = 2):
        """Set ivars to zero for pixels that fall above some maximum value (e.g. cosmic rays)."""
        for r in range(self.R):
            bad = self.ys[r] > max_flux
            self.ys[r][bad] = 1.
            for pad in range(padding): # mask out neighbors of high pixels
                bad = np.logical_or(bad, np.roll(bad, pad+1))
                bad = np.logical_or(bad, np.roll(bad, -pad-1))
            self.ivars[r][bad] = 0.
            
    def mask_bad_edges(self, window_width = 128, min_snr = 5.):
        """
        Find edge regions that contain no information and set ivars there to zero.
        
        Parameters
        ----------
        window_width : `int`
            number of pixels to average over for local SNR            
        min_snr : `float`
            SNR threshold below which we discard the data
        """
        for r in range(self.R):
            n_pix = len(self.xs[r])
            for window_start in range(0, n_pix - window_width, int(window_width/10)):
                window_end = window_start+window_width
                mean_snr = np.sqrt(np.nanmean(self.ys[r][window_start:window_end]**2 * 
                                              self.ivars[r][window_start:window_end]))
                if mean_snr > min_snr:
                    self.ivars[r][:window_start] = 0. # trim everything to left of window
                    break
            for window_start in reversed(range(0, n_pix - window_width, int(window_width/10))):
                window_end = window_start+window_width
                mean_snr = np.sqrt(np.nanmean(self.ys[r][window_start:window_end]**2 * 
                                              self.ivars[r][window_start:window_end]))
                if mean_snr > min_snr:
                    self.ivars[r][window_end:] = 0. # trim everything to right of window
                    break
                
    def transform_log(self, xs=True, ys=True):
        """Transform xs and/or ys attributes to log-space."""
        if xs:
            self.xs = [np.log(x) for x in self.xs]
        if ys:
            self.ivars = [self.ys[i]**2 * self.ivars[i] for i in range(self.R)]
            self.ys = [np.log(y) for y in self.ys]     
        
    def from_HARPS(self, filename, process=True):
        """
        Takes a HARPS CCF file; reads metadata and associated spectrum + wavelength files.
        Note: these files must all be located in the same directory.
        
        Parameters
        ----------
        filename : string
            Location of the CCF file to be read.
        process : bool, optional (default `True`)
            If `True`, do some data processing, including masking of low-SNR 
            regions and strong outliers; continuum normalization; and 
            transformation to ln(wavelength) and ln(flux).
        """
        if not self.empty:
            print("WARNING: overwriting existing contents.")
        R = 72 # number of echelle orders
        metadata = {}
        metadata['filelist'] = filename
        with fits.open(filename) as sp: # load up metadata
            metadata['pipeline_rvs'] = sp[0].header['HIERARCH ESO DRS CCF RVC'] * 1.e3 # m/s
            metadata['pipeline_sigmas'] = sp[0].header['HIERARCH ESO DRS CCF NOISE'] * 1.e3 # m/s
            metadata['drifts'] = sp[0].header['HIERARCH ESO DRS DRIFT SPE RV']
            metadata['dates'] = sp[0].header['HIERARCH ESO DRS BJD']        
            metadata['bervs'] = sp[0].header['HIERARCH ESO DRS BERV'] * 1.e3 # m/s
            metadata['airms'] = sp[0].header['HIERARCH ESO TEL AIRM START'] 
            metadata['pipeline_rvs'] -= metadata['bervs'] # move pipeline rvs back to observatory rest frame
            #metadata['pipeline_rvs'] -= np.mean(metadata['pipeline_rvs']) # just for plotting convenience
        spec_file = str.replace(filename, 'ccf_G2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_M2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_K5', 'e2ds')
        snrs = np.arange(R, dtype=np.float) # order-by-order SNR
        with fits.open(spec_file) as sp:  # assumes same directory
            spec = sp[0].data
            for i in np.nditer(snrs, op_flags=['readwrite']):
                i[...] = sp[0].header['HIERARCH ESO DRS SPE EXT SN{0}'.format(str(int(i)))]
            wave_file = sp[0].header['HIERARCH ESO DRS CAL TH FILE']
        path = spec_file[0:str.rfind(spec_file,'/')+1]
        with fits.open(path+wave_file) as ww: # assumes same directory
            wave = ww[0].data
        xs = [wave[r] for r in range(R)]
        ys = [spec[r] for r in range(R)]
        ivars = [snrs[r]**2/spec[r]/np.nanmean(spec[r,:]) for r in range(R)] # scaling hack
        self.populate(xs, ys, ivars, **metadata)
        if process:
            self.mask_low_pixels()
            self.mask_bad_edges()  
            self.transform_log()  
            self.continuum_normalize()
            self.mask_high_pixels()
                  
        
    def from_HARPSN(self, filename, process=True):
        """
        Takes a HARPS-N CCF file; reads metadata and associated spectrum + wavelength files.
        Note: these files must all be located in the same directory.
        
        Parameters
        ----------
        filename : string
            Location of the CCF file to be read.
        process : bool, optional (default `True`)
            If `True`, do some data processing, including masking of low-SNR 
            regions and strong outliers; continuum normalization; and 
            transformation to ln(wavelength) and ln(flux).
        """
        if not self.empty:
            print("WARNING: overwriting existing contents.")
        R = 69 # number of echelle orders
        metadata = {}
        metadata['filelist'] = filename
        with fits.open(filename) as sp: # load up metadata
            metadata['pipeline_rvs'] = sp[0].header['HIERARCH TNG DRS CCF RVC'] * 1.e3 # m/s
            metadata['pipeline_sigmas'] = sp[0].header['HIERARCH TNG DRS CCF NOISE'] * 1.e3 # m/s
            metadata['drifts'] = sp[0].header['HIERARCH TNG DRS DRIFT RV USED']
            metadata['dates'] = sp[0].header['HIERARCH TNG DRS BJD']        
            metadata['bervs'] = sp[0].header['HIERARCH TNG DRS BERV'] * 1.e3 # m/s
            metadata['airms'] = sp[0].header['AIRMASS']
        spec_file = str.replace(filename, 'ccf_G2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_M2', 'e2ds') 
        spec_file = str.replace(spec_file, 'ccf_K5', 'e2ds')
        snrs = np.arange(R, dtype=np.float)
        with fits.open(spec_file) as sp:
            spec = sp[0].data
            for i in np.nditer(snrs, op_flags=['readwrite']):
                i[...] = sp[0].header['HIERARCH TNG DRS SPE EXT SN{0}'.format(str(int(i)))]
            wave_file = sp[0].header['HIERARCH TNG DRS CAL TH FILE']
        path = spec_file[0:str.rfind(spec_file,'/')+1]
        with fits.open(path+wave_file) as ww:
            wave = ww[0].data
        xs = [wave[r] for r in range(R)]
        ys = [spec[r] for r in range(R)]
        ivars = [snrs[r]**2/spec[r]/np.nanmean(spec[r,:]) for r in range(R)] # scaling hack 
        metadata['pipeline_rvs'] -= metadata['bervs'] # move pipeline rvs back to observatory rest frame
        #metadata['pipeline_rvs'] -= np.mean(metadata['pipeline_rvs']) # just for plotting convenience
        self.populate(xs, ys, ivars, **metadata)
        if process:
            self.mask_low_pixels()
            self.mask_bad_edges()   
            self.transform_log()  
            self.continuum_normalize()
            self.mask_high_pixels()
    
    def from_EXPRES(self, filename, rv_file_name, process=True):
        """
        Takes an EXPRES optimally extracted file; reads metadata and associated
        spectrum + wavelength files.
        Note: these files mst all be located in the same directory.
        
        Parameters
        ----------
        filename : string
            Location of the extracted file to be read.
        rv_file_name : string
            Location of the CCF RV file to be read.
            (Note: likely temporary until RVs are written into extracted file headers)
        process : bool, optional (default `True`)
            If `True`, do some data processing, including masking of low-SNR 
            regions and strong outliers; continuum normalization; and 
            transformation to ln(wavelength) and ln(flux).
        """
        if not self.empty:
            print("WARNING: overwriting existing contents.")
        R = 86 # number of echelle orders (may change? but initial order will be the same)
        
        # For now, we have to read in the RV file separately
        ccf_rvs = pd.read_csv(rv_file_name,index_col=5)
        ccf_rvs = ccf_rvs.drop(ccf_rvs.index[ccf_rvs['ACCEPT']==False]) # Get rid of bad observations
        
        metadata = {}
        metadata['filelist'] = filename
        with fits.open(filename) as sp: # load up metadata
            extr_file = '{}_{}.fits'.format(sp[0].header['OBJECT'].strip(),
                                            sp[0].header['OBS_ID'].strip())
            try:
                metadata['pipeline_rvs']    = ccf_rvs.at[extr_file,'V'] / 100   # m/s
                metadata['pipeline_sigmas'] = ccf_rvs.at[extr_file,'E_V'] / 100 # m/s
            except KeyError:
                print(f'No CCF RV for: {extr_file}')
            
            # Drift is dealt with via the wavelength solution
            # WILL HAVE TO THINK HARDER ABOUT HOW WOBBLE WILL FEEL ABOUT THISs
            
            metadata['dates'] = float(sp[2].header['HIERARCH wtd_mdpt']) # Geometric midpoint in JD, not BJD
            metadata['bervs'] = float(sp[2].header['HIERARCH wtd_single_channel_bc']) * 299792458. # m/s
            metadata['airms'] = float(sp[0].header['AIRMASS']) # Average airmass at center of exposure (though I can also get beginning or end)
            
            # Load Spectrum
            xs = sp[1].data['wavelength'].copy()
            if process: # continuum normalize
                ys = sp[1].data['spectrum'].copy()/sp[1].data['continuum'].copy()
                us = sp[1].data['uncertainty'].copy()/sp[1].data['continuum'].copy()
            else:
                ys = sp[1].data['spectrum'].copy()
                us = sp[1].data['uncertainty'].copy()
            
            #snrs = (int(sp[0].header['EXPCOUNT'])**0.5)*0.357 # SNR of entire observation from exposure meter
            #snrs = np.nanmean(ys/us, axis=1) # Empirical SNR order by order
            #invars = (snr**2/ys)/np.nanmean(ys,axis=1) # Scaling hack
            
            # EXPRES does have individual-pixel uncertainty estimates, should we just use those?
            invars = us.copy()**-2
            
            sp.close()
        self.populate(xs, ys, invars, **metadata)
        
        if process:
            self.mask_low_pixels(min_flux=0.001)
            self.mask_bad_edges(min_snr=10)  
            self.transform_log()
            #self.continuum_normalize() # Done using EXPRES continuum
            self.mask_high_pixels(max_flux=.5)

            
    def from_ESPRESSO(self, filename, process=True):
        """
        Takes an ESPRESSO CCF file; reads metadata and associated spectrum + wavelength files.
        Note: these files must all be located in the same directory.
        
        Parameters
        ----------
        filename : string
            Location of the CCF file to be read.
        process : bool, optional (default `True`)
            If `True`, do some data processing, including masking of low-SNR 
            regions and strong outliers; continuum normalization; and 
            transformation to ln(wavelength) and ln(flux).
        """
        if not self.empty:
            print("WARNING: overwriting existing contents.")
        R = 170 # number of echelle orders
        metadata = {}
        metadata['filelist'] = filename
        with fits.open(filename) as sp: # load up metadata
            metadata['pipeline_rvs'] = sp[0].header['HIERARCH ESO QC CCF RV'] * 1.e3 # m/s
            metadata['pipeline_sigmas'] = sp[0].header['HIERARCH ESO QC CCF RV ERROR'] * 1.e3 # m/s
            metadata['drifts'] = sp[0].header['HIERARCH ESO QC DRIFT DET0 MEAN']
            metadata['dates'] = sp[0].header['HIERARCH ESO QC BJD']        
            metadata['bervs'] = sp[0].header['HIERARCH ESO QC BERV'] * 1.e3 # m/s
            aa = np.zeros(4) + np.nan
            for t in range(1,5): # loop over telescopes
                try:
                    aa[t] = sp[0].header['HIERARCH ESO TEL{0} AIRM START'.format(t)]
                except:
                    continue
            metadata['airms'] = np.nanmedian(aa)
            metadata['pipeline_rvs'] -= metadata['bervs'] # move pipeline rvs back to observatory rest frame
            metadata['pipeline_rvs'] -= np.mean(metadata['pipeline_rvs']) # just for plotting convenience
        spec_file = str.replace(filename, 'CCF', 'S2D') 
        snrs = np.arange(R, dtype=np.float)
        with fits.open(spec_file) as sp:  # assumes same directory
            spec = sp[1].data
            for i in np.nditer(snrs, op_flags=['readwrite']):
                i[...] = sp[0].header['HIERARCH ESO QC ORDER{0} SNR'.format(str(int(i)+1))]
        wave_file = str.replace(spec_file, 'S2D', 'WAVE_MATRIX')
        with fits.open(wave_file) as ww: # assumes same directory
            wave = ww[1].data
        xs = [wave[r] for r in range(R)]
        ys = [spec[r] for r in range(R)]
        ivars = [snrs[r]**2/spec[r]/np.nanmean(spec[r,:]) for r in range(R)] # scaling hack
        self.populate(xs, ys, ivars, **metadata)
        if process:
            self.mask_low_pixels()
            self.mask_bad_edges()  
            self.transform_log()  
            self.continuum_normalize()
            self.mask_high_pixels()
            
    def from_HIRES(self, filename, process=True):
        """
        Takes a HIRES blue chip spectrum file; reads data from it + red + I
        counterparts.
        Note: these files must all be located in the same directory.
        Currently this function does NOT support calculating barycentric shifts 
        from the observation dates. 
        For best performance, you should set the 'bervs' attribute to approximate
        values before analyzing these data.
        
        Parameters
        ----------
        filename : string
            Location of the blue chip spectrum to be read.
        process : bool, optional (default `True`)
            If `True`, do some data processing, including masking of low-SNR 
            regions and strong outliers; continuum normalization; and 
            transformation to ln(wavelength) and ln(flux).
        """
        if not self.empty:
            print("WARNING: overwriting existing contents.")
        R = 23 + 16 + 10 # orders (b + r + i)
        metadata = {}
        metadata['filelist'] = filename
        with fits.open(filename) as sp: # load up blue chip
            metadata['dates'] = float(sp[0].header['MJD']) + 2450000.5       
            metadata['airms'] = sp[0].header['AIRMASS']       
            #TODO: needs BERVs!!!
            spec = np.copy(sp[0].data)
            errs = np.copy(sp[1].data)
            waves = np.copy(sp[2].data)
        other_files = [filename.replace('_b', '_r'), filename.replace('_b', '_i')]
        for f in other_files: # load up red + iodine chips
            with fits.open(f) as sp:
                spec = np.concatenate((spec, sp[0].data))
                errs = np.concatenate((errs, sp[1].data))
                waves = np.concatenate((waves, sp[2].data))        
        xs = [waves[r] for r in range(R)]
        ys = [spec[r] for r in range(R)]
        ivars = [1./errs[r]**2 for r in range(R)]
        self.populate(xs, ys, ivars, **metadata)
        if process:
            self.mask_low_pixels()
            self.mask_bad_edges()  
            self.transform_log()  
            self.continuum_normalize()
            self.mask_high_pixels()     
        
        
