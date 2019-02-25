import numpy as np
from tqdm import tqdm
import sys
import tensorflow as tf
import h5py
T = tf.float64

from .utils import bin_data, doppler, get_session
from .interp import interp
from .history import History

import os
pwd = os.path.dirname(os.path.realpath(__file__))+'/'

class Model(object):
    """
    Keeps track of all components in the model.
    Model is specific to order `r` of data object `data`.
    
    Parameters
    ----------
    data : `object`
        a wobble Data object
    results: `object`
        a wobble Results object
    r : `int`
        the index of the order to be fit in Data
    """
    def __init__(self, data, results, r):
        self.components = []
        self.component_names = []
        self.data = data
        self.results = results
        self.r = r # order index
        self.order = data.orders[r] # order number

    def __str__(self):
        string = 'Model for order {0} consisting of the following components: '.format(self.order)
        for c in self.components:
            string += '\n{0}: '.format(c.name)
            if c.rvs_fixed:
                string += 'RVs fixed; '
            else:
                string += 'RVs variable; '
            string += '{0} variable basis components'.format(c.K)
        return string

    def add_component(self, name, starting_rvs, epochs=None, **kwargs):
        """Add a new Component object to the model."""
        if np.isin(name, self.component_names):
            print("The model already has a component named {0}. Try something else!".format(name))
            return
        if epochs is None: # component is used in all data epochs by default
            epoch_mask = np.ones(self.data.N, dtype='bool')
        else:
            epoch_mask = np.isin(np.arange(self.data.N), epochs)
            starting_rvs[~epoch_mask] = np.nan # NaNs at unused epochs for initialization
        c = Component(name, self.r, starting_rvs, epoch_mask, **kwargs)
        self.components.append(c)
        self.component_names.append(name)
        if not np.isin(name, self.results.component_names):
            self.results.add_component(c)

    def add_star(self, name, starting_rvs=None, **kwargs):
        """Add a component with RVs initialized to zero in the barycentric-corrected rest frame."""
        if starting_rvs is None:
            starting_rvs = -1. * np.copy(self.data.bervs) + np.mean(self.data.bervs)
        kwargs['regularization_par_file'] = kwargs.get('regularization_par_file', 
                                                       pwd+'regularization/default_star.hdf5')
        kwargs['learning_rate_template'] = kwargs.get('learning_rate_template', 0.1)
        self.add_component(name, starting_rvs, **kwargs)

    def add_telluric(self, name, starting_rvs=None, **kwargs):
        """Add a component with RVs initialized to zero in the observatory rest frame."""
        if starting_rvs is None:
            starting_rvs = np.zeros(self.data.N)
        kwargs['learning_rate_template'] = kwargs.get('learning_rate_template', 0.01)
        kwargs['scale_by_airmass'] = kwargs.get('scale_by_airmass', True)
        kwargs['rvs_fixed'] = kwargs.get('rvs_fixed', True)
        kwargs['regularization_par_file'] = kwargs.get('regularization_par_file', 
                                                       pwd+'regularization/default_t.hdf5')
        self.add_component(name, starting_rvs, **kwargs)
        
    def add_continuum(self, degree, **kwargs):
        if np.isin("continuuum", self.component_names):
            print("The model already has a continuum component.")
            return
        c = Continuum(self.r, self.data.N, degree, **kwargs)
        self.components.append(c)
        self.component_names.append(c.name)
        if not np.isin(c.name, self.results.component_names):
            self.results.add_component(c)
        

    def initialize_templates(self):
        """Initialize spectral templates for all components. 
        
        *NOTE:* this will initialize each subsequent component from the residuals 
        of the previous, so make sure you have added the components in order of 
        largest to smallest contribution to the net spectrum.
        """
        data_xs = self.data.xs[self.r]
        data_ys = np.copy(self.data.ys[self.r])
        data_ivars = np.copy(self.data.ivars[self.r])
        for c in self.components:
            data_ys = c.initialize_template(data_xs, data_ys, data_ivars)

    def setup(self):
        """Initialize component templates and do TensorFlow magic in prep for optimizing"""
        self.initialize_templates()
        self.synth = tf.zeros(np.shape(self.data.xs[self.r]), dtype=T, name='synth')
        for c in self.components:
            c.setup(self.data, self.r)
            self.synth += c.synth
            
        self.nll = 0.5*tf.reduce_sum(tf.square(tf.constant(self.data.ys[self.r], dtype=T) 
                                               - self.synth) 
                                    * tf.constant(self.data.ivars[self.r], dtype=T))
        for c in self.components:
            self.nll += c.nll

        # Set up optimizers
        self.updates = []
        for c in self.components:
            if not c.template_fixed:
                c.dnll_dtemplate_ys = tf.gradients(self.nll, c.template_ys)
                c.opt_template = tf.train.AdamOptimizer(c.learning_rate_template).minimize(self.nll,
                            var_list=[c.template_ys])
                self.updates.append(c.opt_template)
            if not c.rvs_fixed:
                c.dnll_drvs = tf.gradients(self.nll, c.rvs)
                c.opt_rvs = tf.train.AdamOptimizer(c.learning_rate_rvs).minimize(self.nll,
                            var_list=[c.rvs])
                self.updates.append(c.opt_rvs)
            if c.K > 0:
                c.opt_basis_vectors = tf.train.AdamOptimizer(c.learning_rate_basis).minimize(self.nll,
                            var_list=[c.basis_vectors])
                self.updates.append(c.opt_basis_vectors)
                c.opt_basis_weights = tf.train.AdamOptimizer(c.learning_rate_basis).minimize(self.nll,
                            var_list=[c.basis_weights])
                self.updates.append(c.opt_basis_weights)
        
        
        session = get_session()
        session.run(tf.global_variables_initializer())

    def optimize(self, niter=100, save_history=False, basename='wobble',
                 feed_dict=None, verbose=True, rv_uncertainties=True, 
                 template_uncertainties=False, **kwargs):
        """Optimize the model!
            
        Parameters
        ----------
        niter : `int` (default `100`)
            Number of iterations.
        save_history : `bool` (default `False`)
            If `True`, create a wobble History object to track progress across 
            iterations and generate plots.
        basename : `str` (default `wobble`)
            Path/name to use when saving plots. Only accessed if save_history = `True`.
        feed_dict : `dict` (default `None`)
            TensorFlow magic; passed to the optimizer. If `None`, does nothing.
        verbose : `bool` (default `True`)
            Toggle print statements and progress bars.
        rv_uncertainties : `bool` (default `True`)
            Toggle whether RV uncertainty estimates should be calculated.
        template_uncertainties : `bool` (default `False`)
            Toggle whether template uncertainty estimates should be calculated.        
        """
        # initialize helper classes:
        if save_history:
            history = History(self, niter+1)
            history.save_iter(self, 0)
        # optimize:
        session = get_session()
        if verbose:
            print("optimize: iterating through {0} optimization steps...".format(niter))
            iterator = tqdm(range(niter), total=niter, miniters=int(niter/10))
        else:
            iterator = range(niter)
        for i in iterator:
            session.run(self.updates, feed_dict=feed_dict)
            if save_history:
                history.save_iter(self, i+1)
        self.estimate_uncertainties(verbose=verbose, rvs=rv_uncertainties, 
                                    templates=template_uncertainties)
        # copy over the outputs to Results:
        for c in self.components:
            self.results.update(c)
        self.results.ys_predicted[self.r] = session.run(self.synth)
        # save optimization plots:
        if save_history:
            history.save_plots(basename, **kwargs)
            return history
            
    def estimate_uncertainties(self, verbose=True, rvs=True, templates=False):
        """Estimate uncertainties using the second derivative of the likelihood. 
        
        Parameters
        ----------
        verbose : `bool` (default `True`)
            Toggle print statements and progress bars.
        rvs : `bool` (default `True`)
            Calculate uncertainties for rvs.
        templates : `bool` (default `False`)
            Calculate uncertainties for template_ys. (NOTE: this may take a while!)
        """
        session = get_session()

        for c in self.components:
            attrs = []
            ivar_attrs = []
            if rvs and not c.rvs_fixed:
                attrs.append('rvs')
                ivar_attrs.append('ivars_rvs')  # TODO: make ivars names consistent
            if templates and not c.template_fixed:
                attrs.append('template_ys')
                ivar_attrs.append('template_ivars')
            for attr, ivar_attr in zip(attrs, ivar_attrs):
                best_values = session.run(getattr(c, attr))
                N_var = len(best_values) # number of variables in attribute
                N_grid = 10
                if verbose:
                    print("optimize: calculating uncertainties on {0} {1}...".format(c.name, attr))
                    iterator = tqdm(range(N_var), total=N_var, 
                                        miniters=int(N_var/20))
                else:
                    iterator = range(N_var)
                for n in iterator: # get d2nll/drv2 from gradients
                    grid = np.tile(best_values, (N_grid,1))
                    grid[:,n] += np.linspace(-1.e1, 1.e1, N_grid) * best_values[n] # vary by 10% - may fail in some cases
                    dnll_dattr_grid = [session.run(getattr(c,'dnll_d{0}'.format(attr)), 
                                                    feed_dict={getattr(c,attr):g})[0][n] for g in grid]
                    # fit a slope with linear algebra
                    A = np.array(grid[:,n]) - best_values[n]
                    ATA = np.dot(A, A)
                    ATy = np.dot(A, np.array(dnll_dattr_grid))
                    getattr(c,ivar_attr)[n] = ATy / ATA
            # TODO: set ivars for basis vectors, basis weights
            
        

class Component(object):
    """
    Generic class for an additive component in the spectral model. 
    """
    def __init__(self, name, r, starting_rvs, epoch_mask, regularization_par_file=None,
                 L1_template=0., L2_template=0., L1_basis_vectors=0.,
                 L2_basis_vectors=0., L2_basis_weights=1., learning_rate_rvs=1.,
                 learning_rate_template=0.01, learning_rate_basis=0.01,
                 rvs_fixed=False, template_fixed=False, variable_bases=0, 
                 scale_by_airmass=False,
                 template_xs=None, template_ys=None, initialize_at_zero=False):
        self.name = name
        self.r = r
        self.K = variable_bases # number of variable basis vectors
        self.N = len(starting_rvs)
        self.rvs_fixed = rvs_fixed
        self.template_fixed = template_fixed
        self.scale_by_airmass = scale_by_airmass
        self.learning_rate_rvs = learning_rate_rvs
        self.learning_rate_template = learning_rate_template
        self.learning_rate_basis = learning_rate_basis
        regularization_par = ['L1_template', 'L2_template']
        if self.K > 0:
            regularization_par = np.append(regularization_par, 
                    ['L1_basis_vectors', 'L2_basis_vectors', 'L2_basis_weights'])
        self.regularization_par = regularization_par
        for par in regularization_par:
            setattr(self, par, eval(par)) # set to input values/defaults
        if regularization_par_file is not None:
            try:
                with h5py.File(regularization_par_file,'r') as f:
                    for par in regularization_par:
                        setattr(self, par, np.copy(f[par][r])) # overwrite with value from file
            except: # TODO: allow use of keywords to be explicitly specified
                print('Regularization parameter file {0} not recognized; using keywords instead.'.format(regularization_par_file))
        self.starting_rvs = starting_rvs
        self.ivars_rvs = np.zeros_like(starting_rvs) + 10. # will be overwritten
        self.template_xs = template_xs
        self.template_ys = template_ys
        self.initialize_at_zero = initialize_at_zero # initialize template ys to flat continuum        
        self.epoch_mask = epoch_mask  
        

    def setup(self, data, r):
        """Do TensorFlow magic in prep for optimizing"""
        self.starting_rvs[np.isnan(self.starting_rvs)] = 0. # because introducing NaNs to synth will fail
        
        # Make some TENSORS (hell yeah)
        self.rvs = tf.Variable(self.starting_rvs, dtype=T, name='rvs_'+self.name)
        self.template_xs = tf.constant(self.template_xs, dtype=T, name='template_xs_'+self.name)
        self.template_ys = tf.Variable(self.template_ys, dtype=T, name='template_ys_'+self.name)
        if self.K > 0:
            self.basis_vectors = tf.Variable(self.basis_vectors, dtype=T, name='basis_vectors_'+self.name)
            self.basis_weights = tf.Variable(self.basis_weights, dtype=T, name='basis_weights_'+self.name)
        self.data_xs = tf.constant(data.xs[r], dtype=T, name='data_xs_'+self.name)

        # Set up the regularization
        for name in self.regularization_par:
            setattr(self, name+'_tensor', tf.constant(getattr(self,name), dtype=T, name=name+'_'+self.name))
        self.nll = self.L1_template_tensor * tf.reduce_sum(tf.abs(self.template_ys))
        self.nll += self.L2_template_tensor * tf.reduce_sum(self.template_ys**2)
        if self.K > 0:
            self.nll += self.L1_basis_vectors_tensor * tf.reduce_sum(tf.abs(self.basis_vectors))
            self.nll += self.L2_basis_vectors_tensor * tf.reduce_sum(self.basis_vectors**2)
            self.nll += self.L2_basis_weights_tensor * tf.reduce_sum(self.basis_weights**2)

        # Apply doppler and synthesize component model predictions
        shifted_xs = self.data_xs + tf.log(doppler(self.rvs[:, None]))
        inner_zeros = tf.zeros(shifted_xs.shape[:-1], dtype=T)
        expand_inner = lambda x: x + inner_zeros[..., None]
        if self.K == 0:
            self.synth = interp(shifted_xs,
                                expand_inner(self.template_xs),
                                expand_inner(self.template_ys))
        else:
            full_template = self.template_ys[None,:] + tf.matmul(self.basis_weights,
                                                                self.basis_vectors)
            self.synth = interp(shifted_xs, expand_inner(self.template_xs), full_template)
            
        # Apply other scaling factors to model
        if self.scale_by_airmass:
            self.synth = tf.einsum('n,nm->nm', tf.constant(data.airms, dtype=T), self.synth)        
        A = tf.constant(self.epoch_mask.astype('float'), dtype=T) # identity matrix
        self.synth = tf.einsum('n,nm->nm', A, self.synth)


    def initialize_template(self, data_xs, data_ys, data_ivars):
        """Doppler-shift data into component rest frame and average 
        to make a composite spectrum. Returns residuals after removing this 
        component from the data.
        
        NOTE: if epochs are masked out, this code implicitly relies on their RVs being NaNs.
        """
        N = len(self.starting_rvs)
        shifted_xs = data_xs + np.log(doppler(self.starting_rvs[:, None], tensors=False)) # component rest frame
        if self.template_xs is None:
            dx = 2.*(np.log(6000.01) - np.log(6000.)) # log-uniform spacing
            tiny = 10.
            self.template_xs = np.arange(np.nanmin(shifted_xs)-tiny*dx,
                                   np.nanmax(shifted_xs)+tiny*dx, dx)
                                   
        if self.template_ys is None:
            if self.initialize_at_zero:
                template_ys = np.zeros_like(self.template_xs)
            else:
                template_ys = bin_data(shifted_xs, data_ys, data_ivars, self.template_xs)
            self.template_ys = template_ys
        self.template_ivars = np.zeros_like(template_ys)
            
        full_template = self.template_ys[None,:] + np.zeros((N,len(self.template_ys)))
        if self.K > 0:
            # initialize basis components
            resids = np.empty((np.sum(self.epoch_mask),len(self.template_ys)))
            i = 0
            for n in range(N): # populate resids with informative epochs
                if self.epoch_mask[n]: # this epoch contains the component
                    resids[i] = np.interp(self.template_xs, shifted_xs[n], data_ys[n]) - self.template_ys
                    i += 1
            u,s,v = np.linalg.svd(resids, compute_uv=True, full_matrices=False)
            basis_vectors = v[:self.K,:] # eigenspectra (K x M)
            basis_weights = u[:, :self.K] * s[None, :self.K] # weights (N x K)
            self.basis_vectors = basis_vectors
            # pad out basis_weights with zeros for data epochs not used:
            basis_weights_all = np.zeros((len(self.starting_rvs), self.K))
            basis_weights_all[self.epoch_mask,:] = basis_weights
            self.basis_weights = basis_weights_all
            full_template += np.dot(self.basis_weights, self.basis_vectors)
        data_resids = np.copy(data_ys)
        for n in range(N):
            if self.epoch_mask[n]:
                data_resids[n] -= np.interp(shifted_xs[n], self.template_xs, full_template[n])
        return data_resids
        
        
class Continuum(Component):
    """
    Polynomial continuum component which is modeled in data space
    """
    def __init__(self, r, N, degree, **kwargs):
        Component.__init__(self, 'continuum', r, np.zeros(N),
                 rvs_fixed=True, variable_bases=0, scale_by_airmass=False, **kwargs)
        self.degree = degree
        
    def setup(self, data, r):
        self.template_xs = tf.constant(self.wavelength_matrix, dtype=T, name='wavelength_matrix_'+self.name)
        self.template_ys = tf.Variable(self.weights, dtype=T, name='weights_'+self.name) # HACK to play well with Results
        self.rvs = tf.constant(self.starting_rvs, dtype=T, name='rvs_'+self.name) # HACK to play well with Results
        self.synth = tf.matmul(self.template_ys, self.template_xs)
        self.nll = tf.constant(0., dtype=T) # no regularization
        
    def initialize_template(self, data_xs, data_ys, data_ivars):
        assert np.all(data_xs[0] == data_xs[1]), "Continuum failed: wavelength grid must be constant in time"
        wavelength_vector = (data_xs[0] - np.mean(data_xs[0]))/(np.max(data_xs[0]) - np.min(data_xs[0]))
        self.wavelength_matrix = np.array([wavelength_vector**d for d in range(1,self.degree)]) # D_degrees x M_pixels
        self.weights = np.zeros((self.N, self.degree-1)) # N_epochs x D_degrees
        return data_ys

