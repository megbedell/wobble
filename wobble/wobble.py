import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from scipy.optimize import minimize
from tqdm import tqdm
import sys
import h5py
import copy
import pickle
import tensorflow as tf
T = tf.float64
import pdb

from .utils import fit_continuum, bin_data
from .interp import interp

speed_of_light = 2.99792458e8   # m/s
COMPONENT_NP_ATTRS = ['K', 'rvs_fixed', 'scale_by_airmass', 'learning_rate_rvs', 'learning_rate_template', 
                      'learning_rate_basis', 'L1_template', 'L2_template', 'L1_basis_vectors', 
                      'L2_basis_vectors', 'L2_basis_weights']
COMPONENT_TF_ATTRS = ['rvs', 'ivars', 'template_xs', 'template_ys', 'basis_vectors', 'basis_weights']

__all__ = ["get_session", "doppler", "Data", "Model", "History", "Results", "optimize_order", "optimize_orders"]

def get_session(restart=False):
  """Get the globally defined TensorFlow session.
  If the session is not already defined, then the function will create
  a global session.
  Returns:
    _SESSION: tf.Session.
  (Code from edward package.)
  """
  global _SESSION
  if tf.get_default_session() is None:
    _SESSION = tf.InteractiveSession()
    
  else:
    _SESSION = tf.get_default_session()

  if restart:
      _SESSION.close()
      _SESSION = tf.InteractiveSession()

  save_stderr = sys.stderr
  return _SESSION

def doppler(v, tensors=True):
    frac = (1. - v/speed_of_light) / (1. + v/speed_of_light)
    if tensors:
        return tf.sqrt(frac)
    else:
        return np.sqrt(frac)
        

class Data(object):
    """
    The data object: contains the spectra and associated data.
    All objects in `data` are numpy arrays or lists of arrays.
    Includes all orders and epochs.
    """
    def __init__(self, filename, filepath='../data/', 
                    N = 0, orders = [30], min_flux = 1., 
                    mask_epochs = None):
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
            self.pipeline_rvs = np.copy(f['pipeline_rvs'])[:self.N] * -1.
            self.dates = np.copy(f['dates'])[:self.N]
            self.bervs = np.copy(f['bervs'])[:self.N] * -1.
            self.drifts = np.copy(f['drifts'])[:self.N]
            self.airms = np.copy(f['airms'])[:self.N]
            
        # mask out bad pixels:
        for r in range(self.R):
            bad = np.where(self.ys[r] < min_flux)
            self.ys[r][bad] = min_flux
            self.ivars[r][bad] = 0.
            
        # mask out bad epochs:
        self.epoch_mask = [True for n in range(self.N)]
        if mask_epochs is not None:
            for n in mask_epochs:
                self.epoch_mask[n] = False

        # log and normalize:
        self.ys = np.log(self.ys) 
        self.continuum_normalize() 
        
        
    def continuum_normalize(self):
        for r in range(self.R):
            for n in range(self.N):
                self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n])
                
class Results(object):
    """
    Stores RV & template results across all orders.
    """
    def __init__(self, data):
        self.component_names = None
        self.R = data.R
        self.N = data.N
        self.orders = data.orders
        self.origin_file = data.origin_file
        
    def add_component(self, c):
        if np.isin(c.name, self.component_names):
            print("A component of name {0} has already been added to the results object.".format(c.name))
            return
        basename = c.name+'_'
        setattr(self, basename+'rvs', np.empty((self.R,self.N)) + np.nan)
        setattr(self, basename+'ivars', np.empty((self.R,self.N)) + np.nan)
        setattr(self, basename+'template_xs', [0 for r in range(self.R)])
        setattr(self, basename+'template_ys', [0 for r in range(self.R)])
        setattr(self, basename+'basis_vectors', [0 for r in range(self.R)])
        setattr(self, basename+'basis_weights', [0 for r in range(self.R)])
        setattr(self, basename+'ys_predicted', [0 for r in range(self.R)])
        for attr in COMPONENT_NP_ATTRS:
            setattr(self, basename+attr, [0 for r in range(self.R)])
                
    def update(self, c):
        basename = c.name+'_'
        for attr in COMPONENT_NP_ATTRS:
            getattr(self, basename+attr)[c.r] = np.copy(getattr(c,attr))
        session = get_session()
        getattr(self, basename+'ys_predicted')[c.r] = session.run(c.synth)
        for attr in COMPONENT_TF_ATTRS:
            try:
                getattr(self, basename+attr)[c.r] = session.run(getattr(c,attr))
            except: # catch when basis vectors/weights don't exist
                assert c.K == 0, "Results: update() failed on attribute {0}".format(attr)
                
    def read(self, filename): # THIS PROBABLY WON'T WORK
        print("Results: reading from {0}".format(filename))
        with h5py.File(filename,'r') as f:
            for attr in np.append(DATA_NP_ATTRS, DATA_TF_ATTRS):
                setattr(self, attr, np.copy(f[attr]))
            self.component_names = np.copy(f['component_names'])
            self.component_names = [a.decode('utf8') for a in self.component_names] # h5py workaround
            self.ys_predicted = np.copy(f['ys_predicted'])
            for name in self.component_names:
                basename = name + '_'
                for attr in np.append(COMPONENT_NP_ATTRS, COMPONENT_TF_ATTRS):
                    try:
                        setattr(self, basename+attr, np.copy(f[basename+attr]))
                    except: # catch when basis vectors are Nones
                        assert np.copy(f[basename+'K']) == 0, "Results: read() failed on attribute {0}".format(basename+attr)
                setattr(self, basename+'ys_predicted', np.copy(f[basename+'ys_predicted']))
            for attr in f['attrs_to_resize']: # trim off the padding
                data = [np.trim_zeros(np.asarray(getattr(self, attr)[r]), 'b') for r in range(self.R)]
                setattr(self, attr, data)
                
                    
    def write(self, filename):
        print("Results: writing to {0}".format(filename))
        self.attrs_to_resize = [['{0}_template_xs'.format(n), '{0}_template_ys'.format(n)] for n in self.component_names]
        self.component_names = [n.encode('utf8') for n in self.component_names] # h5py workaround
        with h5py.File(filename,'w') as f:
            for attr in vars(self):
                if np.isin(attr, self.attrs_to_resize): # pad with NaNs to make rectangular arrays bc h5py is infuriating
                    data = getattr(self, attr)
                    max_len = np.max([len(x) for x in data])
                    for r in range(self.R):
                        data[r] = np.append(data[r], np.zeros(max_len - len(data[r])))
                    f.create_dataset(attr, data=data)   
                else:
                    f.create_dataset(attr, data=getattr(self, attr))
                    
                
    def combine_orders(self, component_name):
        basename = component_name+'_'
        self.all_rvs = np.asarray(getattr(self, basename+'rvs'))
        self.all_ivars = np.asarray(getattr(self, basename+'ivars'))
        # initial guess
        x0_order_rvs = np.median(self.all_rvs, axis=1)
        x0_time_rvs = np.median(self.all_rvs - np.tile(x0_order_rvs[:,None], (1, self.N)), axis=0)
        rv_predictions = np.tile(x0_order_rvs[:,None], (1,self.N)) + np.tile(x0_time_rvs, (self.R,1))
        x0_sigmas = np.log(np.var(self.all_rvs - rv_predictions, axis=1))
        self.M = None
        # optimize
        soln_sigmas = minimize(self.opposite_lnlike_sigmas, x0_sigmas, method='BFGS', options={'disp':True})['x'] # HACK
        # save results
        lnlike, rvs_N, rvs_R = self.lnlike_sigmas(soln_sigmas, return_rvs=True)
        setattr(self, basename+'time_rvs', rvs_N)
        setattr(self, basename+'order_rvs', rvs_R)
        setattr(self, basename+'order_sigmas', soln_sigmas)
        for tmp_attr in ['M', 'all_rvs', 'all_ivars', 'time_rvs', 'order_rvs', 'order_sigmas']:
            delattr(self, tmp_attr) # cleanup
        
    def pack_rv_pars(self, time_rvs, order_rvs, order_sigmas):
        rv_pars = np.append(time_rvs, order_rvs)
        rv_pars = np.append(rv_pars, order_sigmas)
        return rv_pars
    
    def unpack_rv_pars(self, rv_pars):
        self.time_rvs = np.copy(rv_pars[:self.data.N])
        self.order_rvs = np.copy(rv_pars[self.data.N:self.data.R + self.data.N])
        self.order_sigmas = np.copy(rv_pars[self.data.R + self.data.N:])
        return self.time_rvs, self.order_rvs, self.order_sigmas
        
    def lnlike_sigmas(self, sigmas, return_rvs = False, restart = False):
        assert len(sigmas) == self.R
        M = self.get_design_matrix(restart = restart)
        something = np.zeros_like(M[0,:])
        something[self.N:] = 1. / self.data.R # last datum will be mean of order velocities is zero
        M = np.append(M, something[None, :], axis=0) # last datum
        Rs, Ns = self.get_index_lists()
        ivars = 1. / ((1. / self.all_ivars) + sigmas[Rs]**2) # not zero-safe
        ivars = ivars.flatten()
        ivars = np.append(ivars, 1.) # last datum: MAGIC
        MTM = np.dot(M.T, ivars[:, None] * M)
        ys = self.all_rvs.flatten()
        ys = np.append(ys, 0.) # last datum
        MTy = np.dot(M.T, ivars * ys)
        xs = np.linalg.solve(MTM, MTy)
        resids = ys - np.dot(M, xs)
        lnlike = -0.5 * np.sum(resids * ivars * resids - np.log(2. * np.pi * ivars))
        if return_rvs:
            return lnlike, xs[:self.N], xs[self.N:] # must be synchronized with get_design_matrix(), and last datum removal
        return lnlike
        
    def opposite_lnlike_sigmas(self, pars, **kwargs):
        return -1. * self.lnlike_sigmas(pars, **kwargs)    

    def get_index_lists(self):
        return np.mgrid[:self.R, :self.N]

    def get_design_matrix(self, restart = False):
        if (self.M is None) or restart:
            Rs, Ns = self.get_index_lists()
            ndata = self.R * self.N
            self.M = np.zeros((ndata, self.N + self.R)) # note design choices
            self.M[range(ndata), Ns.flatten()] = 1.
            self.M[range(ndata), self.N + Rs.flatten()] = 1.
            return self.M
        else:
            return self.M
                
class Model(object):
    """
    Keeps track of all components in the model.
    Model is specific to order `r` of data object `data`.
    """
    def __init__(self, data, r):
        self.components = []
        self.component_names = []
        self.data = data
        self.r = r
        
    def __str__(self):
        string = 'Model for order {0} consisting of the following components: '.format(self.data.orders[self.r])
        for c in self.components:
            string += '\n{0}: '.format(c.name)
            if c.rvs_fixed:
                string += 'RVs fixed; '
            else:
                string += 'RVs variable; '
            string += '{0} variable basis components'.format(c.K)
        return string
        
    def add_star(self, name, starting_rvs=None, **kwargs):
        if np.isin(name, self.component_names):
            print("The model already has a component named {0}. Try something else!".format(name))
            return
        if starting_rvs is None:
            starting_rvs = np.copy(self.data.bervs) - np.mean(self.data.bervs)
        c = Component(name, starting_rvs, **kwargs)
        self.components.append(c)
        self.component_names.append(name)
        
    def add_telluric(self, name, starting_rvs=None, **kwargs):
        if np.isin(name, self.component_names):
            print("The model already has a component named {0}. Try something else!".format(name))
            return
        if starting_rvs is None:
            starting_rvs = np.zeros(self.data.N)
        kwargs['learning_rate_template'] = kwargs.get('learning_rate_template', 0.1)
        kwargs['scale_by_airmass'] = kwargs.get('scale_by_airmass', True)
        kwargs['rvs_fixed'] = kwargs.get('rvs_fixed', True)
        c = Component(name, starting_rvs, **kwargs)
        self.components.append(c)
        self.component_names.append(name)
        
    def initialize_templates(self):
        data_xs = self.data.xs[self.r]
        data_ys = np.copy(self.data.ys[self.r])
        template_xs = None
        for c in self.components:
            data_ys = c.initialize_template(data_xs, data_ys, template_xs)        
        
    def setup(self):
        self.initialize_templates()
        self.synth = tf.zeros(np.shape(self.data.xs[self.r]), dtype=T)
        for c in self.components:
            c.setup(self.data, self.r)
            self.synth += c.synth
        self.nll = (self.data.ys[self.r] - self.synth)**2 * self.data.ivars[self.r]
        for c in self.components:
            self.nll += c.nll
            
    def optimize(self, results, niter=100, save_history=False, basename='wobble'):
        # TODO
        # initialize tensorflow optimizers etc
        # initialize helper classes:
        if save_history:
            history = History(self, self.data, self.r, niter)
        # optimize
        results.update(self)
        # save history
                                
class Component(object):
    """
    Generic class for an additive component in the spectral model.
    """
    def __init__(self, name, starting_rvs, L1_template=0., L2_template=0., L1_basis_vectors=0.,
                 L2_basis_vectors=0., L2_basis_weights=1., learning_rate_rvs=10.,
                 learning_rate_template=0.01, learning_rate_basis=0.01,
                 rvs_fixed=False, variable_bases=0, scale_by_airmass=False):
        self.name = name
        self.K = variable_bases # number of variable basis vectors
        self.N = len(starting_rvs)
        self.rvs_fixed = rvs_fixed
        self.scale_by_airmass = scale_by_airmass
        self.learning_rate_rvs = learning_rate_rvs
        self.learning_rate_template = learning_rate_template
        self.learning_rate_basis = learning_rate_basis
        self.L1_template = L1_template
        self.L2_template = L2_template
        self.L1_basis_vectors = L1_basis_vectors
        self.L2_basis_vectors = L2_basis_vectors
        self.L2_basis_weights = L2_basis_weights
        self.starting_rvs = starting_rvs  
            
    def setup(self, data, r):
        self.rvs = tf.Variable(self.starting_rvs, dtype=T)
        self.ivars = tf.constant(np.zeros(data.N) + 10., dtype=T) # TODO
        self.template_xs = tf.constant(self.template_xs, dtype=T)
        self.template_ys = tf.Variable(self.template_ys, dtype=T)
        if self.K > 0:
            self.basis_vectors = tf.Variable(self.basis_vectors, dtype=T)
            self.basis_weights = tf.Variable(self.basis_weights, dtype=T)
        
        self.data_xs = tf.constant(data.xs[r], dtype=T)
        
        # Set up the regularization
        self.L1_template_tensor = tf.constant(self.L1_template, dtype=T) # maybe change to Variable?
        self.L2_template_tensor = tf.constant(self.L2_template, dtype=T)
        self.L1_basis_vectors_tensor = tf.constant(self.L1_basis_vectors, dtype=T)
        self.L2_basis_vectors_tensor = tf.constant(self.L2_basis_vectors, dtype=T)
        self.L2_basis_weights_tensor = tf.constant(self.L2_basis_weights, dtype=T)
        
        self.nll = self.L1_template_tensor * tf.norm(self.template_ys, 1)
        self.nll += self.L2_template_tensor * tf.norm(self.template_ys, 2)
        if self.K > 0:
            self.nll += self.L1_basis_vectors_tensor * tf.norm(self.basis_vectors, 1)
            self.nll += self.L2_basis_vectors_tensor * tf.norm(self.basis_vectors, 2)
            self.nll += self.L2_basis_weights_tensor * tf.norm(self.basis_weights, 2)
        
        # Apply doppler
        shifted_xs = self.data_xs + tf.log(doppler(self.rvs[:, None]))
        if self.K == 0:
            self.synth = interp(shifted_xs, self.template_xs, self.template_ys) 
        else:
            full_template = self.template_ys[None,:] + tf.matmul(self.basis_weights, 
                                                                self.basis_vectors)
            synth = []
            for n in range(self.N):
                synth.append(interp(shifted_xs, self.template_xs, full_template[n])[n])  # TODO: change interp to handle this in one?
            self.synth = tf.stack(synth)
        if self.scale_by_airmass:
            self.synth = tf.einsum('n,nm->nm', tf.constant(data.airms, dtype=T), self.synth)
        
        
    def initialize_template(self, data_xs, data_ys, template_xs=None):
        """
        Doppler-shift data into component rest frame, subtract off other components, 
        and average to make a composite spectrum.
        """
        shifted_xs = data_xs + np.log(doppler(self.starting_rvs[:, None], tensors=False)) # component rest frame
        if template_xs is None:
            dx = 2.*(np.log(6000.01) - np.log(6000.)) # log-uniform spacing
            tiny = 10.
            template_xs = np.arange(np.min(shifted_xs)-tiny*dx, 
                                   np.max(shifted_xs)+tiny*dx, dx) 
        
        template_xs, template_ys = bin_data(shifted_xs, data_ys, template_xs)
        self.template_xs = template_xs
        self.template_ys = template_ys
        full_template = template_ys[None,:] + np.zeros((len(self.starting_rvs),len(template_ys)))
        if self.K > 0:
            # initialize basis components
            resids = np.empty((len(self.starting_rvs),len(template_ys)))
            for n in range(len(self.starting_rvs)):
                resids[n] = np.interp(template_xs, shifted_xs[n], data_ys[n]) - template_ys
            u,s,v = np.linalg.svd(resids, compute_uv=True, full_matrices=False)
            basis_vectors = v[:self.K,:] # eigenspectra (K x M)
            basis_weights = u[:, :self.K] * s[None, :self.K] # weights (N x K)
            self.basis_vectors = basis_vectors
            self.basis_weights = basis_weights
            full_template += np.dot(basis_weights, basis_vectors)
        data_resids = np.copy(data_ys)
        for n in range(len(self.starting_rvs)):
            data_resids[n] -= np.interp(shifted_xs[n], template_xs, full_template[n])
        return data_resids
         
        
    def make_optimizers(self, r, nll, learning_rate_rvs=None, 
            learning_rate_template=None, learning_rate_basis=None):
        # TODO: make each one an R-length list rather than overwriting each order?
        if learning_rate_rvs == None:
            learning_rate_rvs = self.learning_rate_rvs
        if learning_rate_template == None:
            learning_rate_template = self.learning_rate_template
        if learning_rate_basis == None:
            learning_rate_basis = self.learning_rate_basis
        self.gradients_template = tf.gradients(nll, self.template_ys[r])
        self.opt_template = tf.train.AdamOptimizer(learning_rate_template).minimize(nll, 
                            var_list=[self.template_ys[r]])
        if not self.rvs_fixed:
            self.gradients_rvs = tf.gradients(nll, self.rvs_block[r])
            self.opt_rvs = tf.train.AdamOptimizer(learning_rate_rvs).minimize(nll, 
                            var_list=[self.rvs_block[r]])
        if self.K > 0:
            self.gradients_basis = tf.gradients(nll, [self.basis_vectors[r], self.basis_weights[r]])
            self.opt_basis = tf.train.AdamOptimizer(learning_rate_basis).minimize(nll, 
                            var_list=[self.basis_vectors[r], self.basis_weights[r]]) 
        # TODO: put initialization here
        

        
class History(object):
    """
    Information about optimization history of a single order stored in numpy arrays/lists
    """   
    def __init__(self, model, data, r, niter, filename=None):
        for c in model.components:
            assert c.template_exists[r], "ERROR: Cannot initialize History() until templates are initialized."
        self.nll_history = np.empty(niter)
        self.rvs_history = [np.empty((niter, data.N)) for c in model.components]
        self.template_history = [np.empty((niter, int(c.template_ys[r].shape[0]))) for c in model.components]
        self.basis_vectors_history = [np.empty((niter, c.K, 4096)) for c in model.components] # HACK
        self.basis_weights_history = [np.empty((niter, data.N, c.K)) for c in model.components]
        self.chis_history = np.empty((niter, data.N, 4096)) # HACK
        self.r = r
        self.niter = niter
        if filename is not None:
            self.read(filename)
        
    def save_iter(self, model, data, i, nll, chis):
        """
        Save all necessary information at optimization step i
        """
        session = get_session()
        self.nll_history[i] = session.run(nll)
        self.chis_history[i,:,:] = np.copy(session.run(chis))
        for j,c in enumerate(model.components):
            template_state = session.run(c.template_ys[self.r])
            rvs_state = session.run(c.rvs_block[self.r])
            self.template_history[j][i,:] = np.copy(template_state)
            self.rvs_history[j][i,:] = np.copy(rvs_state)  
            if c.K > 0:
                self.basis_vectors_history[j][i,:,:] = np.copy(session.run(c.basis_vectors[self.r])) 
                self.basis_weights_history[j][i,:,:] = np.copy(session.run(c.basis_weights[self.r]))
        
    def write(self, filename=None):
        """
        Write to hdf5
        """
        if filename is None:
            filename = 'order{0}_history.hdf5'.format(self.r)
        print("saving optimization history to {0}".format(filename))
        with h5py.File(filename,'w') as f:
            for attr in ['nll_history', 'chis_history', 'r', 'niter']:
                f.create_dataset(attr, data=getattr(self, attr))
            for attr in ['rvs_history', 'template_history', 'basis_vectors_history', 'basis_weights_history']:
                for i in range(len(self.template_history)):
                    f.create_dataset(attr+'_{0}'.format(i), data=getattr(self, attr)[i])   
                    
    
    def read(self, filename):
        """
        Read from hdf5
        """         
        with h5py.File(filename, 'r') as f:
            for attr in ['nll_history', 'chis_history', 'r', 'niter']:
                setattr(self, attr, np.copy(f[attr]))
            for attr in ['rvs_history', 'template_history', 'basis_vectors_history', 'basis_weights_history']:
                d = []
                for i in range(len(self.template_history)):
                    d.append(np.copy(f[attr+'_{0}'.format(i)]))
                setattr(self, attr, d)
                
        
    def animfunc(self, i, xs, ys, xlims, ylims, ax, driver):
        """
        Produces each frame; called by History.plot()
        """
        ax.cla()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title('Optimization step #{0}'.format(i))
        s = driver(xs, ys[i,:])
        
    def plot(self, xs, ys, linestyle, nframes=None, ylims=None):
        """
        Generate a matplotlib animation of xs and ys
        Linestyle options: 'scatter', 'line'
        """
        if nframes is None:
            nframes = self.niter
        fig = plt.figure()
        ax = plt.subplot() 
        if linestyle == 'scatter':
            driver = ax.scatter
        elif linestyle == 'line':
            driver = ax.plot
        else:
            print("linestyle not recognized.")
            return
        x_pad = (np.max(xs) - np.min(xs)) * 0.1
        xlims = (np.min(xs)-x_pad, np.max(xs)+x_pad)
        if ylims is None:
            y_pad = (np.max(ys) - np.min(ys)) * 0.1
            ylims = (np.min(ys)-y_pad, np.max(ys)+y_pad)
        ani = animation.FuncAnimation(fig, self.animfunc, np.linspace(0, self.niter-1, nframes, dtype=int), 
                    fargs=(xs, ys, xlims, ylims, ax, driver), interval=150)
        plt.close(fig)
        return ani  
                         
    def plot_rvs(self, ind, model, data, compare_to_pipeline=True, **kwargs):
        """
        Generate a matplotlib animation of RVs vs. time
        ind: index of component in model to be plotted
        compare_to_pipeline keyword subtracts off the HARPS DRS values (useful for removing BERVs)
        """
        xs = data.dates
        ys = self.rvs_history[ind]
        if compare_to_pipeline:
            ys -= np.repeat([data.pipeline_rvs], self.niter, axis=0)    
        return self.plot(xs, ys, 'scatter', **kwargs)     
    
    def plot_template(self, ind, model, data, **kwargs):
        """
        Generate a matplotlib animation of the template inferred from data
        ind: index of component in model to be plotted
        """
        session = get_session()
        template_xs = session.run(model.components[ind].template_xs[self.r])
        xs = np.exp(template_xs)
        ys = np.exp(self.template_history[ind])
        return self.plot(xs, ys, 'line', **kwargs) 
    
    def plot_chis(self, epoch, model, data, **kwargs):
        """
        Generate a matplotlib animation of model chis in data space
        epoch: index of epoch to plot
        """
        session = get_session()
        data_xs = session.run(data.xs[self.r][epoch,:])
        xs = np.exp(data_xs)
        ys = self.chis_history[:,epoch,:]
        return self.plot(xs, ys, 'line', **kwargs)   
     
            

def optimize_order(model, data, r, **kwargs):
    '''
    optimize the model for order r in data
    '''      
    model.setup()    
    results = model.optimize(**kwargs)
    return results

def optimize_orders(model, data, **kwargs):
    """
    optimize model for all orders in data
    """
    results = Results(model=model, data=data)
    for r in range(data.R):
        print("--- ORDER {0} ---".format(r))
        if r == 0: 
            results = optimize_order(model, data, r, **kwargs)
        else:
            results = optimize_order(model, data, r, results=results, **kwargs)
        #if (r % 5) == 0:
        #    results.write('results_order{0}.hdf5'.format(data.orders[r]))
    #results.write('results.hdf5')   
    results.compute_final_rvs(model) 
    return results    