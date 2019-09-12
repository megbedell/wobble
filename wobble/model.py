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

    def __repr__(self):
        string = 'wobble.Model for order {0} consisting of the following components: '.format(self.order)
        for i,c in enumerate(self.components):
            string += '\n{0}: {1}; '.format(i, c.name)
            if c.rvs_fixed:
                string += 'RVs fixed; '
            else:
                string += 'RVs variable; '
            string += '{0} variable basis components'.format(c.K)
        return string

    def add_component(self, name, starting_rvs, epochs=None, **kwargs):
        """
        Append a new Component object to the model.
        
        Parameters
        ----------
        name : `str`
            The name of the component. Must be unique.
        starting_rvs : `np.ndarray`
            N-epoch length vector of initial guesses for RVs; will be used
            to stack & average data resids for initialization of the template.
        epochs : `np.ndarray`
            Indices between 0:N denoting epochs where this component is present 
            in the data. Defaults to all N epochs.
         **kwargs : `dict`
            Keywords to be passed to wobble.Component()
        """
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
        """
        Convenience function to add a component with RVs initialized to zero 
        in the barycentric-corrected rest frame.
        Will have regularization parameters and learning rates set to default values
        for a stellar spectrum.
        """
        if starting_rvs is None:
            starting_rvs = -1. * np.copy(self.data.bervs) + np.mean(self.data.bervs)
        kwargs['regularization_par_file'] = kwargs.get('regularization_par_file', 
                                                       pwd+'regularization/default_star.hdf5')
        kwargs['learning_rate_template'] = kwargs.get('learning_rate_template', 0.1)
        self.add_component(name, starting_rvs, **kwargs)

    def add_telluric(self, name, starting_rvs=None, **kwargs):
        """
        Convenience function to add a component with RVs fixed to zero 
        in the observatory rest frame. Component contribution scales with airmass by default.
        Will have regularization parameters and learning rates set to default values
        for a telluric spectrum.
        """
        if starting_rvs is None:
            starting_rvs = np.zeros(self.data.N)
        kwargs['learning_rate_template'] = kwargs.get('learning_rate_template', 0.01)
        kwargs['scale_by_airmass'] = kwargs.get('scale_by_airmass', True)
        kwargs['rvs_fixed'] = kwargs.get('rvs_fixed', True)
        kwargs['regularization_par_file'] = kwargs.get('regularization_par_file', 
                                                       pwd+'regularization/default_t.hdf5')
        self.add_component(name, starting_rvs, **kwargs)
        
    def add_continuum(self, degree, **kwargs):
        """Untested code for adding a continuum component."""
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
        assert False not in np.isfinite(data_xs), "Non-finite value(s) or NaN(s) in wavelengths."
        assert False not in np.isfinite(data_ys), "Non-finite value(s) or NaN(s) in log spectral values."
        assert False not in np.isfinite(data_ivars), "Non-finite value(s) or NaN(s) in inverse variance."
        for c in self.components:
            data_ys = c.initialize_template(data_xs, data_ys, data_ivars)

    def setup(self):
        """Initialize component templates and do TensorFlow magic in prep for optimizing"""
        self.initialize_templates()
        self.synth = tf.zeros(np.shape(self.data.xs[self.r]), dtype=T, name='synth')
        for c in self.components:
            c.setup(self.data, self.r)
            self.synth = tf.add(self.synth, c.synth, name='synth_add_{0}'.format(c.name))
            
        self.nll = 0.5*tf.reduce_sum(tf.square(tf.constant(self.data.ys[self.r], dtype=T) 
                                               - self.synth, name='nll_data-model_sq') 
                                    * tf.constant(self.data.ivars[self.r], dtype=T), name='nll_reduce_sum')
        for c in self.components:
            self.nll = tf.add(self.nll, c.nll, name='nll_add_{0}'.format(c.name))

        # Set up optimizers
        self.updates = []
        for c in self.components:
            if not c.template_fixed:
                c.dnll_dtemplate_ys = tf.gradients(self.nll, c.template_ys)
                c.opt_template = tf.train.AdamOptimizer(c.learning_rate_template).minimize(self.nll,
                            var_list=[c.template_ys], name='opt_minimize_template_{0}'.format(c.name))
                self.updates.append(c.opt_template)
            if not c.rvs_fixed:
                c.dnll_drvs = tf.gradients(self.nll, c.rvs)
                c.opt_rvs = tf.train.AdamOptimizer(learning_rate=c.learning_rate_rvs,
                                                   epsilon=1.).minimize(self.nll,
                            var_list=[c.rvs], name='opt_minimize_rvs_{0}'.format(c.name))
                self.updates.append(c.opt_rvs)
            if c.K > 0:
                c.opt_basis_vectors = tf.train.AdamOptimizer(c.learning_rate_basis).minimize(self.nll,
                            var_list=[c.basis_vectors], name='opt_minimize_basis_vectors_{0}'.format(c.name))
                self.updates.append(c.opt_basis_vectors)
                c.opt_basis_weights = tf.train.AdamOptimizer(c.learning_rate_basis).minimize(self.nll,
                            var_list=[c.basis_weights], name='opt_minimize_basis_weights_{0}'.format(c.name))
                self.updates.append(c.opt_basis_weights)
        
        
        session = get_session()
        session.run(tf.global_variables_initializer())

    def optimize(self, niter=100, save_history=False, basename='wobble',
                 movies=False, epochs_to_plot=[0,1,2], verbose=True, 
                 rv_uncertainties=True, template_uncertainties=False, **kwargs):
        """Optimize the model!
            
        Parameters
        ----------
        niter : `int` (default `100`)
            Number of iterations.
        save_history : `bool` (default `False`)
            If `True`, create a wobble History object to track progress across 
            iterations and generate plots.
        movies : `bool` (default `False`)
            Use with `save_history`; if `True`, will generate animations of 
            optimization progress.
        epochs_to_plot : `list` (default `[0,1,2]`)
            Use with `save_history`; indices of epochs to plot fits for. Each
            epoch will generate its own plot/movies.
        basename : `str` (default `wobble`)
            Use with `save_history`; path/name stem to use when saving plots.
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
            session.run(self.updates, **kwargs)
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
            history.save_plots(basename, movies=movies, epochs_to_plot=epochs_to_plot)
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
            Calculate uncertainties for template_ys. (NOTE: this will take a while!)
        """
        session = get_session()

        for c in self.components:
            attrs = []
            ivar_attrs = []
            epsilon = []
            if rvs and not c.rvs_fixed:
                attrs.append('rvs')
                ivar_attrs.append('ivars_rvs')  # TODO: make ivars names consistent
            if templates and not c.template_fixed:
                attrs.append('template_ys')
                ivar_attrs.append('template_ivars')
            for attr, ivar_attr in zip(attrs, ivar_attrs):
                if attr == 'rvs':
                    epsilon = 10. # perturb by a few m/s - TODO: scale this with spectrum SNR!
                else:
                    epsilon = 0.01 # perturb by 1%
                best_values = session.run(getattr(c, attr))
                N_var = len(best_values) # number of variables in attribute
                N_grid = 5
                if verbose:
                    print("optimize: calculating uncertainties on {0} {1}...".format(c.name, attr))
                    iterator = tqdm(range(N_var), total=N_var, 
                                        miniters=int(N_var/20))
                else:
                    iterator = range(N_var)
                for n in iterator: # get d2nll/drv2 from gradients
                    grid = np.tile(best_values, (N_grid,1))
                    grid[:,n] += np.linspace(-epsilon, epsilon, N_grid) # vary according to epsilon scale
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
    You will probably never need to call this class directly. 
    Instead, use wobble.Model.add_component() to append an instance 
    of this object to the list saved as wobble.Model.components.
    
    Parameters
    ----------
    name : `str`
        The name of the component. Must be unique within the model.
    r : `int`
        the index of the order to be fit in the data. Must be the same as
        `Model.r`.
    starting_rvs : `np.ndarray`
        N-epoch length vector of initial guesses for RVs; will be used
        to stack & average data resids for initialization of the template.
    epoch_mask : `np.ndarray` of type `bool`
        N-epoch mask where epoch_mask[n] = `True` indicates that this 
        component contributes to the model at epoch n.
    rvs_fixed : `bool` (default `False`)
        If `True`, fix the RVs to their initial values and do not
        optimize.
    template_fixed : `bool` (default `False`)
        If `True`, fix the template to its initial values and do not
        optimize.
    variable_bases : `int` (default `0`)
        Number of basis vectors to use in time variability of `template_ys`. 
        If zero, no time variability is allowed.
    scale_by_airmass : `bool` (default `False`)
        If `True`, component contribution to the model scales linearly with
        airmass.
    template_xs : `np.ndarray` or `None` (default `None`)
        Grid of x-values for the spectral template in the same units as 
        data `xs`. If `None`, generate automatically upon initialization.
    template_ys : `np.ndarray` or `None` (default `None`)
        Grid of starting guess y-values for the spectral template 
        in the same units as data `ys`.
        If `None`, generate automatically upon initialization.  
        If not `None`, `template_xs` must be provided in the same shape.
    initialize_at_zero : `bool` (default `False`)
        If `True`, initialize template as a flat continuum. Equivalent to
        providing a vector of zeros with `template_ys` keyword but does
        not require passing a `template_xs` keyword.
    learning_rate_rvs : `float` (default 1.)
        Learning rate for Tensorflow Adam optimizer to use in `rvs` 
        optimization step.
    learning_rate_template : `float` (default 0.01)
        Learning rate for Tensorflow Adam optimizer to use in `template_ys` 
        optimization step.
    learning_rate_basis : `float` (default 0.01)
        Learning rate for Tensorflow Adam optimizer to use in `basis_vectors` 
        optimization step.
    regularization_par_file : `str` or `None` (default `None`)
        Name of HDF5 file containing the expected regularization amplitudes.
        If keyword arguments are set for any of these amplitudes, that value
        is used instead of file contents. If `None` & no keywords, no
        regularization is used.
    L1_template : `float` (default `0`)
        L1 regularization amplitude on `self.template_ys`. If zero, no 
        regularization is used. If not explicitly specified and a valid
        `regularization_par_file` is given, value from there is used.
    L2_template : `float` (default `0`)
        L2 regularization amplitude on `self.template_ys`. If zero, no 
        regularization is used. If not explicitly specified and a valid
        `regularization_par_file` is given, value from there is used.
    L1_basis_vectors : `float` (default `0`)
        L1 regularization amplitude on `self.basis_vectors`. If zero, no 
        regularization is used. If not explicitly specified and a valid
        `regularization_par_file` is given, value from there is used. 
        Only set if `variable_bases` > 0.
    L2_basis_vectors : `float` (default `1`)
        L2 regularization amplitude on `self.basis_vectors`. If zero, no 
        regularization is used. If not explicitly specified and a valid
        `regularization_par_file` is given, value from there is used.
        Only set if `variable_bases` > 0.
    L2_basis_weights : `float` (default `0`)
        L1 regularization amplitude on `self.basis_weights`. If zero, no 
        regularization is used. If not explicitly specified and a valid
        `regularization_par_file` is given, value from there is used.
        Only set if `variable_bases` > 0.
        Not recommended to change, as this is degenerate with basis vectors.
    """
    def __init__(self, name, r, starting_rvs, epoch_mask, 
                 rvs_fixed=False, template_fixed=False, variable_bases=0, 
                 scale_by_airmass=False,
                 template_xs=None, template_ys=None, initialize_at_zero=False,    
                 learning_rate_rvs=1., learning_rate_template=0.01, 
                 learning_rate_basis=0.01, regularization_par_file=None, 
                 **kwargs):
        for attr in ['name', 'r', 'starting_rvs', 'epoch_mask',
                    'rvs_fixed', 'template_fixed', 'template_xs',
                    'template_ys', 'initialize_at_zero', 
                    'learning_rate_rvs', 'learning_rate_template',
                    'learning_rate_basis', 'scale_by_airmass']:
            setattr(self, attr, eval(attr))

        self.K = variable_bases # number of variable basis vectors
        self.N = len(starting_rvs)
        self.ivars_rvs = np.zeros_like(starting_rvs) + 10. # will be overwritten
        
        regularization_par = ['L1_template', 'L2_template']
        if self.K > 0:
            regularization_par = np.append(regularization_par, 
                    ['L1_basis_vectors', 'L2_basis_vectors', 'L2_basis_weights'])
        self.regularization_par = regularization_par # the names of the regularization parameters
        
        default_regularization_par = {'L1_template':0., 'L2_template':0.,
                                      'L1_basis_vectors':0., 'L2_basis_vectors':0.,
                                      'L1_basis_weights':1.}
        for par in regularization_par: 
            if par in kwargs.keys(): # prioritize explicitly set keywords over all else
                setattr(self, par, kwargs[par])
            elif regularization_par_file is not None: # try setting from file
                try:
                    with h5py.File(regularization_par_file,'r') as f:
                        setattr(self, par, np.copy(f[par][r]))
                except:
                    print('Regularization parameter file {0} not recognized; \
                            adopting default values instead.'.format(regularization_par_file)) 
                    setattr(self, par, default_regularization_par[par])
            else:  # if no file & no keyword argument, set to defaults
                setattr(self, par, default_regularization_par[par])

    def __repr__(self):
        return "wobble.Component named {0}".format(self.name)
        

    def setup(self, data, r):
        """Do TensorFlow magic & define likelihoods in prep for optimizing"""
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
        self.nll = tf.multiply(self.L1_template_tensor, tf.reduce_sum(tf.abs(self.template_ys)), 
                               name='L1_template_'+self.name)
        self.nll = tf.add(self.nll, tf.multiply(self.L2_template_tensor, 
                                                tf.reduce_sum(tf.square(self.template_ys)),
                                                name='L2_template_'+self.name), 
                          name='L1_plus_L2_template_'+self.name)
        if self.K > 0:
            self.nll = tf.add(self.nll, tf.multiply(self.L1_basis_vectors_tensor, 
                                                    tf.reduce_sum(tf.abs(self.basis_vectors))))
            self.nll = tf.add(self.nll, tf.multiply(self.L2_basis_vectors_tensor, 
                                                     tf.reduce_sum(tf.square(self.basis_vectors))))
            self.nll = tf.add(self.nll, tf.multiply(self.L2_basis_weights_tensor, 
                                                    tf.reduce_sum(tf.square(self.basis_weights))))

        # Apply doppler and synthesize component model predictions
        shifted_xs = tf.add(self.data_xs, tf.log(doppler(self.rvs))[:, None], name='shifted_xs_'+self.name)
        inner_zeros = tf.zeros(shifted_xs.shape[:-1], dtype=T)
        expand_inner = lambda x: tf.add(x, inner_zeros[..., None], name='expand_inner_'+self.name)
        if self.K == 0:
            self.synth = interp(shifted_xs,
                                expand_inner(self.template_xs),
                                expand_inner(self.template_ys))
        else:
            full_template = tf.add(self.template_ys[None,:], tf.matmul(self.basis_weights,
                                                                self.basis_vectors))
            self.synth = interp(shifted_xs, expand_inner(self.template_xs), full_template)
            
        # Apply other scaling factors to model
        if self.scale_by_airmass:
            self.synth = tf.einsum('n,nm->nm', tf.constant(data.airms, dtype=T), self.synth, 
                                   name='airmass_einsum_'+self.name)        
        A = tf.constant(self.epoch_mask.astype('float'), dtype=T) # identity matrix
        self.synth = tf.multiply(A[:,None], self.synth, name='epoch_masking_'+self.name)
        #self.synth = tf.einsum('n,nm->nm', A, self.synth, name='epoch_masking_'+self.name)


    def initialize_template(self, data_xs, data_ys, data_ivars):
        """Doppler-shift data into component rest frame and average 
        to make a composite spectrum. Returns residuals after removing this 
        component from the data.
        Must be done BEFORE running `Component.setup()`.
        
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
        self.template_ivars = np.zeros_like(self.template_ys)
            
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

