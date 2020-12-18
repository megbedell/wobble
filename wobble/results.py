import numpy as np
from scipy.optimize import minimize
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from astropy.table import Table, Column
T = tf.float64

from .utils import get_session

COMMON_ATTRS = ['R', 'N', 'orders', 'component_names', 'bervs', 'pipeline_rvs', 
                'pipeline_sigmas', 'drifts', 'dates', 'airms', 'epochs'] # common across all orders & components
COMPONENT_NP_ATTRS = ['K', 'r', 'rvs_fixed', 'ivars_rvs', 'template_ivars', 'scale_by_airmass', 'learning_rate_rvs', 
                      'learning_rate_template', 'L1_template', 'L2_template']
OPT_COMPONENT_NP_ATTRS = ['learning_rate_basis', 'L1_basis_vectors', 'L2_basis_vectors', 'L2_basis_weights'] # it's ok if these don't exist
COMPONENT_TF_ATTRS = ['rvs', 'template_xs', 'template_ys']
OPT_COMPONENT_TF_ATTRS = ['basis_vectors', 'basis_weights'] # only present if component K > 0
POST_COMPONENT_ATTRS = ['time_rvs', 'time_sigmas', 'order_rvs', 'order_sigmas', 
                        'bary_corr', 'drift_corr'] # these won't exist until post-processing

class Results(object):
    """A read/writeable object which stores RV & template results across all orders. 
    At the end of each Model optimize() call, the associated Results object is 
    updated with numpy outputs from each optimized TensorFlow variable. 
    This allows us to clear out the graph and retain the solution.
    
    One of the two keywords is required for initialization.
    
    Parameters
    ----------
    data : `object`, optional
        a wobble Data object
    filename : `str`, optional
        a file path pointing to a saved Results object (HDF5 format).
    """
    def __init__(self, data=None, filename=None):
        if (filename is None) & (data is not None):
            assert data is not None, "ERROR: must supply either data or filename keywords."
            for attr in COMMON_ATTRS:
                if attr == 'component_names':
                    setattr(self, attr, [])
                else:
                    setattr(self, attr, getattr(data, attr))
            self.ys_predicted = [0 for r in range(self.R)]
        elif (data is None) & (filename is not None):
            assert filename is not None, "ERROR: must supply either data or filename keywords."
            self.read(filename)
        else:
            assert False, "ERROR: must supply EITHER data OR filename keywords."
        
    def __repr__(self):
        string = 'wobble.Results object consisting of the following components: '
        for i,c in enumerate(self.component_names):
            string += '\n{0}: {1}; '.format(i, c)
            if getattr(self, '{0}_bary_corr'.format(c)):
                string += 'RVs barycentric corrected; '
            if getattr(self, '{0}_drift_corr'.format(c)):
                string += 'RVs drift corrected; '
            #string += '{0} variable basis components'.format(c.K)
        return string
                            
    def add_component(self, c):
        """Initialize a new model component and prepare to save its optimized outputs. 
        The component name should be consistent across all order models. 
        
        Note that if a component name was initialized in the models for 1+ orders but 
        was not included in all order models, its RV values/uncertainties will be set 
        to NaNs and all other properties set to 0 for the excluded order(s).
        
        Parameters
        ----------
        c : a wobble.Model.Component object
        """
        if np.isin(c.name, self.component_names):
            print("Results: A component of name {0} has already been added.".format(c.name))
            return
        self.component_names.append(c.name)
        basename = c.name+'_'
        setattr(self, basename+'rvs', np.full((self.R,self.N), np.nan, dtype=np.float64))
        setattr(self, basename+'ivars_rvs', np.full((self.R,self.N), np.nan, dtype=np.float64))
        setattr(self, basename+'template_xs', [0 for r in range(self.R)])
        setattr(self, basename+'template_ys', [0 for r in range(self.R)])
        setattr(self, basename+'template_ivars', [0 for r in range(self.R)])
        if c.K > 0:
            setattr(self, basename+'basis_vectors', [0 for r in range(self.R)])
            setattr(self, basename+'basis_weights', [0 for r in range(self.R)])
        setattr(self, basename+'ys_predicted', [0 for r in range(self.R)])
        setattr(self, basename+'drift_corr', False)
        setattr(self, basename+'bary_corr', False)
        attrs = COMPONENT_NP_ATTRS
        if c.K > 0:
            attrs = np.append(attrs, OPT_COMPONENT_NP_ATTRS)
        for attr in attrs:
            setattr(self, basename+attr, [0 for r in range(self.R)])
                
    def update(self, c, **kwargs):
        """Update the attributes of a component from the current values of Model.
        
        Parameters
        ----------
        c : a wobble.Model.Component object
        """
        basename = c.name+'_'
        attrs = np.copy(COMPONENT_NP_ATTRS)
        if c.K > 0:
            attrs = np.append(attrs, OPT_COMPONENT_NP_ATTRS)
        for attr in attrs:
            getattr(self, basename+attr)[c.r] = np.copy(getattr(c,attr))
        session = get_session()
        getattr(self, basename+'ys_predicted')[c.r] = session.run(c.synth, **kwargs)
        attrs = np.copy(COMPONENT_TF_ATTRS)
        if c.K > 0:
            attrs = np.append(attrs, OPT_COMPONENT_TF_ATTRS)
        for attr in attrs:
            getattr(self, basename+attr)[c.r] = session.run(getattr(c,attr), **kwargs)
                
    def read(self, filename):
        """Read from HDF5 file."""
        print("Results: reading from {0}".format(filename))
        with h5py.File(filename,'r') as f:
            for attr in COMMON_ATTRS:
                try:
                    setattr(self, attr, np.copy(f[attr]))
                except KeyError:
                    print("WARNING: attribute {0} could not be read.".format(attr))
            self.component_names = np.copy(f['component_names'])
            self.component_names = [a.decode('utf8') for a in self.component_names] # h5py workaround
            self.ys_predicted = [0 for r in range(self.R)]
            all_order_attrs = ['ys_predicted']
            for name in self.component_names:
                basename = name + '_'
                for attr in POST_COMPONENT_ATTRS:
                    try:
                        setattr(self, basename+attr, np.copy(f[basename+attr]))
                    except KeyError:
                        continue
                setattr(self, basename+'ys_predicted', [0 for r in range(self.R)])
                all_order_attrs.append(basename+'ys_predicted')
                attrs = np.append(COMPONENT_NP_ATTRS, COMPONENT_TF_ATTRS)
                opt_attrs = np.append(OPT_COMPONENT_NP_ATTRS, OPT_COMPONENT_TF_ATTRS)
                attrs = np.append(attrs, opt_attrs)
                for attr in attrs:
                    try: # if attribute exists in hdf5, set it up
                        test = f['order0'][basename+attr]
                        setattr(self, basename+attr, [0 for r in range(self.R)])
                        all_order_attrs.append(basename+attr)                        
                    except KeyError: # if attribute doesn't exist, move on
                        continue
            for r in range(self.R):
                for attr in all_order_attrs:
                    getattr(self, attr)[r] = np.copy(f['order{0}'.format(r)][attr])
                
                    
    def write(self, filename):
        """Write to HDF5 file."""
        print("Results: writing to {0}".format(filename))
        with h5py.File(filename,'w') as f:            
            for r in range(self.R):
                g = f.create_group('order{0}'.format(r))
                g.create_dataset('ys_predicted', data=self.ys_predicted[r])
                for n in self.component_names:
                    g.create_dataset(n+'_ys_predicted', data=getattr(self, n+'_ys_predicted')[r])
                    attrs = np.append(COMPONENT_NP_ATTRS, COMPONENT_TF_ATTRS)
                    if np.any(np.ravel(getattr(self, n+'_K')) > 0):
                        attrs = np.append(attrs, np.append(OPT_COMPONENT_NP_ATTRS, OPT_COMPONENT_TF_ATTRS))
                    for attr in attrs:
                        g.create_dataset(n+'_'+attr, data=getattr(self, n+'_'+attr)[r])
            for n in self.component_names:
                for attr in POST_COMPONENT_ATTRS:
                    try:
                        f.create_dataset(n+'_'+attr, data=getattr(self, n+'_'+attr))
                    except:
                        continue
            self.component_names = [a.encode('utf8') for a in self.component_names] # h5py workaround
            for attr in COMMON_ATTRS:
                f.create_dataset(attr, data=getattr(self, attr))  
            self.component_names = [a.decode('utf8') for a in self.component_names] # h5py workaround                  
                
    def combine_orders(self, component_name):
        """Calculate and save final time-series RVs for a given component after all 
        orders have been optimized.
        
        Parameters
        ----------
        component_name : `str`
            Name of the model component to use.
        """
        if not np.isin(component_name, self.component_names):
            print("Results: component name {0} not recognized. Valid options are: {1}".format(component_name, 
                    self.component_names))
        basename = component_name+'_'
        self.all_rvs = np.asarray(getattr(self, basename+'rvs'))
        self.all_ivars = np.asarray(getattr(self, basename+'ivars_rvs'))
        # initial guess
        x0_order_rvs = np.median(self.all_rvs, axis=1)
        x0_time_rvs = np.median(self.all_rvs - np.tile(x0_order_rvs[:,None], (1, self.N)), axis=0)
        rv_predictions = np.tile(x0_order_rvs[:,None], (1,self.N)) + np.tile(x0_time_rvs, (self.R,1))
        x0_sigmas = np.log(np.var(self.all_rvs - rv_predictions, axis=1))
        self.M = None
        # optimize
        soln = minimize(self.opposite_lnlike_sigmas, x0_sigmas, method='BFGS')
        soln_sigmas = soln['x']
        if not soln['success']:
            print(soln.status)
            print(soln.message)
            if not np.isfinite(soln['fun']):
                print("ERROR: non-finite likelihood encountered in optimization. Setting combined RVs to non-optimal values.")
                soln_sigmas = x0_sigmas
        # save results
        lnlike, rvs_N, rvs_R, Cinv = self.lnlike_sigmas(soln_sigmas, return_rvs=True) # Cinv is inverse covariance matrix
        setattr(self, basename+'time_rvs', rvs_N)
        setattr(self, basename+'order_rvs', rvs_R)
        setattr(self, basename+'order_jitters', soln_sigmas)
        setattr(self, basename+'time_sigmas', np.sqrt(np.diag(np.linalg.inv(Cinv))[:self.N])) # really really a bad idea
        for tmp_attr in ['M', 'all_rvs', 'all_ivars']:
            delattr(self, tmp_attr) # cleanup
        #return Cinv
        
    def lnlike_sigmas(self, sigmas, return_rvs = False, restart = False):
        """Internal code used by combine_orders()"""
        assert len(sigmas) == self.R
        M = self.get_design_matrix(restart = restart)
        something = np.zeros_like(M[0,:])
        something[self.N:] = 1. / self.R # last datum will be mean of order velocities is zero
        M = np.append(M, something[None, :], axis=0) # last datum
        Rs, Ns = self.get_index_lists()
        ivars = 1. / ((1. / self.all_ivars) + sigmas[Rs]**2) # not zero-safe
        ivars = ivars.flatten()
        ivars = np.append(ivars, 1.) # last datum: MAGIC - implicit units of 1/velocity**2
        MTM = np.dot(M.T, ivars[:, None] * M)
        ys = self.all_rvs.flatten()
        ys = np.append(ys, 0.) # last datum
        MTy = np.dot(M.T, ivars * ys)
        xs = np.linalg.solve(MTM, MTy)
        resids = ys - np.dot(M, xs)
        lnlike = -0.5 * np.sum(resids * ivars * resids - np.log(2. * np.pi * ivars))
        if return_rvs:
            return lnlike, xs[:self.N], xs[self.N:], MTM # must be synchronized with get_design_matrix(), and last datum removal
        return lnlike
        
    def opposite_lnlike_sigmas(self, pars, **kwargs):
        """...the opposite of lnlike_sigmas()"""
        return -1. * self.lnlike_sigmas(pars, **kwargs)    

    def get_index_lists(self):
        """Internal code used by combine_orders()"""
        return np.mgrid[:self.R, :self.N]

    def get_design_matrix(self, restart = False):
        """Internal code used by combine_orders()"""
        if (self.M is None) or restart:
            Rs, Ns = self.get_index_lists()
            ndata = self.R * self.N
            self.M = np.zeros((ndata, self.N + self.R)) # note design choices
            self.M[range(ndata), Ns.flatten()] = 1.
            self.M[range(ndata), self.N + Rs.flatten()] = 1.
            return self.M
        else:
            return self.M
            
    def apply_drifts(self, component_name):
        """Apply instrumental drifts to all RVs for a given component. 
        Will modify both `rvs` and `time_rvs` (if applicable). 
        Will not modify pipeline RVs; these are assumed to have drift
        corrections already.
        
        Parameters
        ----------
        component_name : `str`
            Name of the model component to use.
        """
        if not np.isin(component_name, self.component_names):
            print("Results: component name {0} not recognized. Valid options are: {1}".format(component_name, 
                    self.component_names))
            return
        if not hasattr(self, 'drifts'):
            print("No instrumental drifts measured.")
            return
        basename = component_name+'_'
        all_rvs = np.asarray(getattr(self, basename+'rvs'))
        setattr(self, basename+'rvs', all_rvs - np.tile(self.drifts[None,:], (self.R,1)))
        if hasattr(self, basename+'time_rvs'):
            time_rvs = np.asarray(getattr(self, basename+'time_rvs'))
            setattr(self, basename+'time_rvs', time_rvs - self.drifts)
        setattr(self, basename+'drift_corr', True)
            
    def apply_bervs(self, component_name):
        """Apply barycentric corrections to all RVs for a given component. 
        Will modify both `rvs` and `time_rvs` (if applicable).
        BERVs follow standard convention: +ve = Earth is moving 
        toward the observed object, so 
        barycentric-corrected stellar RV = measured RV + BERV.
        Correction is applied to wobble RVs and to pipeline RVs to keep 
        them in the same frame.
        
        Parameters
        ----------
        component_name : `str`
            Name of the model component to use.
        """
        if not np.isin(component_name, self.component_names):
            print("Results: component name {0} not recognized. Valid options are: {1}".format(component_name, 
                    self.component_names))
            return
        if not hasattr(self, 'bervs'):
            print("No barycentric shifts computed.")
            return 
        basename = component_name+'_'
        all_rvs = np.asarray(getattr(self, basename+'rvs'))
        all_bervs = np.tile(self.bervs[None,:], (self.R,1))
        setattr(self, basename+'rvs', all_rvs + all_bervs)
        if hasattr(self, basename+'time_rvs'):
            time_rvs = np.asarray(getattr(self, basename+'time_rvs'))
            setattr(self, basename+'time_rvs', time_rvs + self.bervs) 
        self.pipeline_rvs += self.bervs
        setattr(self, basename+'bary_corr', True)
        
    def write_rvs(self, component_name, filename, all_orders=False):
        """Write out a text file containing time series RVs.
        
        Parameters
        ----------
        component_name : `str`
            Name of the model component to use.
        all_orders : `bool`
            If True, include one column for each individual orders
        """
        if not np.isin(component_name, self.component_names):
            print("Results: component name {0} not recognized. Valid options are: {1}".format(component_name, 
                    self.component_names))
            return
        t = Table()
        t['dates'] = self.dates
        t.meta['comments'] = ['Table of wobble RVs for {0} component; all units m/s.'.format(component_name)]
        if getattr(self, '{0}_bary_corr'.format(component_name)):
            t.meta['comments'].append('RVs have been barycentric corrected.')
        else:
            t.meta['comments'].append('RVs are in observatory rest frame.')
        if getattr(self, '{0}_drift_corr'.format(component_name)):
            t.meta['comments'].append('RVs have been corrected for instrumental drift.')
        
        if hasattr(self, '{0}_time_rvs'.format(component_name)):
            t['RV'] = getattr(self, '{0}_time_rvs'.format(component_name))
            t['RV_err'] = getattr(self, '{0}_time_sigmas'.format(component_name))
        if all_orders:
            all_rvs = getattr(self, '{0}_rvs'.format(component_name))
            all_ivars = getattr(self, '{0}_ivars_rvs'.format(component_name))
            for r,o in enumerate(self.orders):
                t['RV_order{0}'.format(o)] = all_rvs[r]
                t['RV_order{0}_err'.format(o)] = 1./np.sqrt(all_ivars[r])
        t['pipeline_rv'] = self.pipeline_rvs
        t['pipeline_rv_err'] = self.pipeline_sigmas
        t.write(filename, format='ascii')
        print('Output saved to file: {0}'.format(filename))
        
    def plot_spectrum(self, r, n, data, filename, xlim=None, ylim=[0., 1.3], ylim_resids=[-0.1,0.1]):
        """Output a figure showing synthesized fit to a section of spectrum.
        
        Parameters
        ----------
        r : `int`
            Index of echelle order to plot, within range (0,R]
        n : `int`
            Index of observation epoch to plot.
        data : `wobble.Data` object
            Pointer to the Data object containing spectra.
        filename : `str`
            Name & path under which to save the output plot.
        xlim : 
            Optional x-range; passed to matplotlib.
        ylim : 
            Optional y-range for main plot; passed to matplotlib.
        ylim_resids : 
            Optional y-range for residuals plot; passed to matplotlib.
        """
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(12,5))
        xs = np.exp(data.xs[r][n])
        ax.scatter(xs, np.exp(data.ys[r][n]), marker=".", alpha=0.5, c='k', label='data', s=40)
        mask = data.ivars[r][n] <= 1.e-8
        ax.scatter(xs[mask], np.exp(data.ys[r][n][mask]), marker=".", alpha=1., c='white', s=20)
        for c in self.component_names:
            ax.plot(xs, np.exp(getattr(self, "{0}_ys_predicted".format(c))[r][n]), alpha=0.8)
        ax2.scatter(xs, np.exp(data.ys[r][n]) - np.exp(self.ys_predicted[r][n]), 
                    marker=".", alpha=0.5, c='k', label='data', s=40)
        ax2.scatter(xs[mask], np.exp(data.ys[r][n][mask]) - np.exp(self.ys_predicted[r][n][mask]), 
                    marker=".", alpha=1., c='white', s=20)
        ax.set_ylim(ylim)
        ax2.set_ylim(ylim_resids)
        ax.set_xticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.savefig(filename)
        plt.close(fig)
        
    def plot_chromatic_rvs(self, min_order=None, max_order=None, percentiles=(16,84), wavelengths=None, scale='log',  ylim=None, center=True, filename=None):
        """Output a representative percentile plot showing the chromaticity apparent in the rv signal.
        
        Parameters
        ----------
        min_order : 'int'
                    Minimum order to plot.
        max_order : 'int'
                    Maximum order to plot. 
        percentiles : 'tuple'
                    Optional upper and lower percentile to plot.
        wavelengths : 'tuple'
                    Optional wavelength range (in Angstroms) to use instead of orders. 
        scale : 'str'
                    Optional scale; passed to matplotlib.
        ylim : 'tuple'
                    Optional ylim; passed to matplotlib.
        center : 'boolean' 
                    Determines whether the epochs are median centered before percentiles are calculated. 
        filename : 'str'
                    Saves plot if given. Optional filename; passed to matplotlib e.g. 'filename.png'
   	"""
        if min_order == None:
            min_order = 0
        if max_order == None:
            max_order = len(self.orders)
        orders = np.arange(min_order, max_order)
        x = np.array([np.exp(np.mean(order)) for order in self.star_template_xs[min_order:max_order]])
        if wavelengths != None:
            orders = ((x > int(wavelengths[0])) & (x < int(wavelengths[1])))
            x = x[orders]
        if center == True:
            median = np.median(np.array(self.star_rvs)[orders], axis=1).T
        else:
            median = 0
        upper = np.percentile(np.array(self.star_rvs)[orders].T - median, percentiles[1], axis=0).T
        upper_sigma = np.sqrt(1/np.array(self.star_ivars_rvs))[orders, np.argmin(abs(np.array(self.star_rvs)[orders].T - upper.T).T, axis=1)]
        lower = np.percentile(np.array(self.star_rvs)[orders].T - median, percentiles[0], axis=0).T
        lower_sigma = np.sqrt(1/np.array(self.star_ivars_rvs))[orders, np.argmin(abs(np.array(self.star_rvs)[orders].T - lower.T).T, axis=1)]
        m, b = np.polyfit(x, upper, 1)
        m2, b2 = np.polyfit(x, lower, 1)
        plt.errorbar(x, upper, upper_sigma, fmt='o')
        plt.errorbar(x, lower, lower_sigma, fmt='o')
        plt.plot(x, m*x + b, color='tab:blue')
        plt.plot(x, m2*x + b2, color='tab:orange')
        plt.ylabel('Radial Velocity [m/s]')
        plt.xlabel('Wavelength λ [Å]')
        plt.ylim(ylim)
        plt.xscale(scale)
        if filename != None:
            plt.savefig(filename)
        plt.show()
        
        
    
    def chromatic_index(self, min_order=None, max_order=None, wavelengths=None):
        """ Returns a 2 by n array representing the chromatic index along with uncertainty for each epoch (calculated as slope of the linear least squares fit).
        
        Parameters
        ----------
        min_order : 'int'
                    Minimum order to use in the calculation of the chromatic indices.
        max_order : 'int'
                    Maximum order to use in the calculation of the chromatic indices.
        wavelengths : 'tuple'
                    Optional wavelength range (in Angstroms) to use instead of orders. 
        """
        if min_order == None:
            min_order = 0
        if max_order == None:
            max_order = len(self.orders)
        orders = np.arange(min_order, max_order)
        x = np.array([np.mean(order) for order in self.star_template_xs[min_order:max_order]])
        if wavelengths != None:
            orders = ((x > int(wavelengths[0])) & (x < int(wavelengths[1])))
            x = x[orders]
        chromatic_indices = []
        sigmas = []
        for epoch in range(len(self.epochs)): 
           coefs = np.polyfit(x, np.array(self.star_rvs)[orders, epoch], 1, full=True)
           chromatic_indices.append(coefs[0][0])
           sigmas.append(np.sqrt(coefs[1][0]))
        return [chromatic_indices, sigmas]
