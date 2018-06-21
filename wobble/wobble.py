import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from tqdm import tqdm
import sys
import h5py
import copy
import tensorflow as tf
T = tf.float64
import pdb

from .utils import fit_continuum, bin_data
from .interp import interp
from .old_wobble import Results

speed_of_light = 2.99792458e8   # m/s

def get_session():
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

  save_stderr = sys.stderr
  return _SESSION

def doppler(v):
    frac = (1. - v/speed_of_light) / (1. + v/speed_of_light)
    return tf.sqrt(frac)

class Data(object):
    """
    The data object: contains the spectra and associated data.
    """
    def __init__(self, filename, filepath='../data/', 
                    N = 0, orders = [30], min_flux = 1., tensors=True):
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

        # mask out bad data:
        for r in range(self.R):
            bad = np.where(self.ys[r] < min_flux)
            self.ys[r][bad] = min_flux
            self.ivars[r][bad] = 0.
            
        # log and normalize:
        self.ys = np.log(self.ys) 
        self.continuum_normalize() 
        
        # HACK -- for storing results
        self.wobble_obj = Results(R=self.R, N=self.N)
        
        # convert to tensors
        if tensors:
            self.ys = [tf.constant(y, dtype=T) for y in self.ys]
            self.xs = [tf.constant(x, dtype=T) for x in self.xs]
            self.ivars = [tf.constant(i, dtype=T) for i in self.ivars]
        
    def continuum_normalize(self):
        for r in range(self.R):
            for n in range(self.N):
                self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n])
        
                
class Model(object):
    """
    Keeps track of all components in the model.
    """
    def __init__(self, data):
        self.components = []
        self.component_names = []
        self.components_rvs_fixed = []
        self.data = data
        
    def __str__(self):
        string = 'Model consisting of the following components: '
        for c in self.components:
            string += '\n{0}: '.format(c.name)
            if c in self.components_rvs_fixed:
                string += 'RVs fixed; '
            else:
                string += 'RVs variable; '
            string += '{0} variable basis components'.format(c.K)
        return string
        
    def add_star(self, name, rvs_fixed=False, variable_bases=0):
        c = Star(name, self.data, variable_bases=variable_bases)
        self.components.append(c)
        self.component_names.append(name)
        if rvs_fixed:
            self.components_rvs_fixed.append(c)
        
    def add_telluric(self, name, rvs_fixed=True, variable_bases=0):
        c = Telluric(name, self.data, variable_bases=variable_bases)
        self.components.append(c)
        self.component_names.append(name)
        if rvs_fixed:
            self.components_rvs_fixed.append(c)
                
    def load(self, filename):
        print("loading...")
        # TODO: write this code
    
    def save(self, filename):
        print("saving...")
        # TODO: write this code
                                
class Component(object):
    """
    Generic class for an additive component in the spectral model.
    """
    def __init__(self, name, data, variable_bases=0):
        self.name = name
        self.K = variable_bases # number of variable basis vectors
        self.rvs_block = [tf.Variable(np.zeros(data.N), dtype=T, name='rvs_order{0}'.format(r)) for r in range(data.R)]
        self.template_xs = [None for r in range(data.R)] # this will be replaced
        self.template_ys = [None for r in range(data.R)] # this will be replaced
        self.basis_vectors = [None for r in range(data.R)] # this will be replaced
        self.basis_weights = [None for r in range(data.R)] # this will be replaced
        self.learning_rate_rvs = 10. # default
        self.learning_rate_template = 0.01 # default
        self.learning_rate_basis = 0.01 # default
        
    def shift_and_interp(self, r, xs, rvs):
        """
        Apply Doppler shift of rvs to the model at order r and output interpolated values at xs.
        """
        shifted_xs = xs + tf.log(doppler(rvs[:, None]))
        return interp(shifted_xs, self.template_xs[r], self.template_ys[r]) 
        
    def initialize_template(self, r, data, other_components=None, template_xs=None):
        """
        Doppler-shift data into component rest frame, subtract off other components, 
        and average to make a composite spectrum.
        """
        shifted_xs = data.xs[r] + tf.log(doppler(self.rvs_block[r][:, None])) # component rest frame
        if template_xs is None:
            dx = tf.constant(np.log(6000.01) - np.log(6000.), dtype=T) # log-uniform spacing
            tiny = tf.constant(10., dtype=T)
            template_xs = tf.range(tf.reduce_min(shifted_xs)-tiny*dx, 
                                   tf.reduce_max(shifted_xs)+tiny*dx, dx)           
        resids = 1. * data.ys[r]
        for c in other_components: # subtract off initialized components
            if c.template_ys[r] is not None:
                resids -= c.shift_and_interp(r, data.xs[r], c.rvs_block[r])
                
        session = get_session()
        session.run(tf.global_variables_initializer())
        template_xs, template_ys = bin_data(session.run(shifted_xs), session.run(resids), 
                                            session.run(template_xs)) # hack
        self.template_xs[r] = tf.Variable(template_xs, dtype=T, name='template_xs')
        self.template_ys[r] = tf.Variable(template_ys, dtype=T, name='template_ys') 
        if self.K > 0:
            # initialize basis components
            resids -= self.shift_and_interp(r, data.xs[r], self.rvs_block[r])
            s,u,v = tf.svd(resids, compute_uv=True)
            basis_vectors = tf.transpose(tf.conj(v[:,:self.K])) # eigenspectra (K x M)
            basis_weights = (u * s)[:,:self.K] # weights (N x K)
            self.basis_vectors[r] = tf.Variable(basis_vectors, dtype=T, name='basis_vectors')
            self.basis_weights[r] = tf.Variable(basis_weights, dtype=T, name='basis_weights') 
         
        
    def make_optimizers(self, r, nll, learning_rate_rvs=None, 
            learning_rate_template=None, learning_rate_basis=None):
        # TODO: make each one an R-length list rather than overwriting each order?
        if learning_rate_rvs == None:
            learning_rate_rvs = self.learning_rate_rvs
        if learning_rate_template == None:
            learning_rate_template = self.learning_rate_template
        if learning_rate_basis == None:
            learning_rate_basis = self.learning_rate_basis
        self.gradients_model = tf.gradients(nll, self.template_ys[r])
        self.gradients_rvs = tf.gradients(nll, self.rvs_block[r])
        self.opt_template = tf.train.AdamOptimizer(learning_rate_template).minimize(nll, 
                            var_list=[self.template_ys[r]])
        self.opt_rvs = tf.train.AdamOptimizer(learning_rate_rvs).minimize(nll, 
                            var_list=[self.rvs_block[r]])
        save = [self.template_xs[r], self.template_ys[r], self.rvs_block[r]]
        if self.K > 0:
            self.gradients_basis = tf.gradients(nll, [self.basis_vectors[r], self.basis_weights[r]])
            self.opt_basis = tf.train.AdamOptimizer(learning_rate_basis).minimize(nll, 
                            var_list=[self.basis_vectors[r], self.basis_weights[r]]) 
            save.append(self.basis_vectors[r])
            save.append(self.basis_weights[r])
        self.saver = tf.train.Saver(var_list=save)  
        

class Star(Component):
    """
    A star (or generic celestial object)
    """
    def __init__(self, name, data, variable_bases=0):
        Component.__init__(self, name, data, variable_bases=variable_bases)
        starting_rvs = np.copy(data.bervs) - np.mean(data.bervs)
        self.rvs_block = [tf.Variable(starting_rvs, dtype=T, name='rvs_order{0}'.format(r)) for r in range(data.R)]        
                            
class Telluric(Component):
    """
    Sky absorption
    """
    def __init__(self, name, data, variable_bases=0):
        Component.__init__(self, name, data, variable_bases=variable_bases)
        self.learning_rate_template = 0.1   
        

def optimize_order(model, data, r, niter=100, save_every=100, output_history=False):
    '''
    NOTE: regularization is "dumb ass" - DWH
    consider L1 vs. L2
    set regularization amplitude in a sensible way
    '''        
    for c in model.components:
        if c.template_ys[r] is None:
            c.initialize_template(r, data, other_components=[x for x in model.components if x!=c])         
                     
    if output_history:
        nll_history = np.empty(niter*2)
        rvs_history = [np.empty((niter, data.N)) for c in model.components]
        template_history = [np.empty((niter, int(c.template_ys[r].shape[0]))) for c in model.components]
        basis_vectors_history = [np.empty((niter, c.K, 4096)) for c in model.components] # HACK
        basis_weights_history = [np.empty((niter, data.N, c.K)) for c in model.components]
        chis_history = np.empty((niter, data.N, 4096)) # HACK
        
    # likelihood calculation:
    synth = tf.zeros_like(data.ys[r])
    L1_norm_template = 0.
    L2_norm_basis_vectors = 0.
    L2_norm_basis_weights = 0.
    #L1_amp_template = 1.e6 # magic number
    L1_amp_template = 0.
    L2_amp_basis_vectors = 1.e4 # magic number
    L2_amp_basis_weights = 1. # magic number
    for c in model.components:
        synth += c.shift_and_interp(r, data.xs[r], c.rvs_block[r])
        L1_norm_template += L1_amp_template * tf.reduce_sum(tf.abs(c.template_ys[r]))
        if c.K > 0: # variable components - assumes they're in observatory rest frame
            synth += tf.matmul(c.basis_weights[r], c.basis_vectors[r])
            L2_norm_basis_vectors = L2_amp_basis_vectors * tf.reduce_sum(tf.square(c.basis_vectors[r]))
            L2_norm_basis_weights = L2_amp_basis_weights * tf.reduce_sum(tf.square(c.basis_weights[r]))
    chis = (data.ys[r] - synth) * tf.sqrt(data.ivars[r])
    nll = 0.5*tf.reduce_sum(tf.square(data.ys[r] - synth) * data.ivars[r]) + L1_norm_template \
            + L2_norm_basis_vectors + L2_norm_basis_weights
        
    # set up optimizers: 
    for c in model.components:
        c.make_optimizers(r, nll)

    session = get_session()
    session.run(tf.global_variables_initializer())
        
    # optimize:
    print("--- ORDER {0} ---".format(r))
    for i in tqdm(range(niter)):
        if output_history: 
            nll_history[2*i] = session.run(nll)
            chis_history[i,:,:] = np.copy(session.run(chis))
            for j,c in enumerate(model.components):
                template_state = session.run(c.template_ys[r])
                rvs_state = session.run(c.rvs_block[r])
                template_history[j][i,:] = np.copy(template_state)
                rvs_history[j][i,:] = np.copy(rvs_state)  
                if c.K > 0:
                    basis_vectors_history[j][i,:,:] = np.copy(session.run(c.basis_vectors[r])) 
                    basis_weights_history[j][i,:,:] = np.copy(session.run(c.basis_weights[r]))             
        for c in model.components:
            if not c in model.components_rvs_fixed:            
                session.run(c.opt_rvs) # optimize RVs
        if output_history: 
            nll_history[2*i+1] = session.run(nll)
        for c in model.components:            
            session.run(c.opt_template) # optimize mean template
            if c.K > 0:
                session.run(c.opt_basis) # optimize variable components
        if (i+1 % save_every == 0):
            c.saver.save(session, "tf_checkpoints/{0}_order{1}".format(c.name, r), global_step=i)
        
    if output_history:
        return nll_history, rvs_history, template_history, chis_history, basis_vectors_history, basis_weights_history # hack
    else:
        return

def optimize_orders(model, data, **kwargs):
    session = get_session()
    session.run(tf.global_variables_initializer())    # should this be in get_session?
    for r in range(data.R):
        optimize_order(session, model, data, r, **kwargs)
        #if (r % 5) == 0:
        #    model.save('results_order{0}.hdf5'.format(r))

def animfunc(i, xs, ys, xlims, ylims, ax, driver):
    ax.cla()
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_title('Optimization step #{0}'.format(i))
    s = driver(xs, ys[i,:])
            
def plot_rv_history(data, rvs_history, niter, nframes, ylims=None, compare_to_pipeline=True):
    fig = plt.figure()
    ax = plt.subplot()
    xs = data.dates
    if compare_to_pipeline:
        ys = rvs_history - np.repeat([data.pipeline_rvs], niter, axis=0)    
    else:
        ys = rvs_history        
    x_pad = (np.max(xs) - np.min(xs)) * 0.1
    xlims = (np.min(xs)-x_pad, np.max(xs)+x_pad)
    if ylims is None:
        ylims = (np.min(ys)-10., np.max(ys)+10.)
    ani = animation.FuncAnimation(fig, animfunc, np.linspace(0, niter-1, nframes, dtype=int), 
                fargs=(xs, ys, xlims, ylims, ax, ax.scatter))
    plt.close(fig)
    return ani  
    
def plot_template_history(template_xs, template_history, niter, nframes, ylims=None):
    fig = plt.figure()
    ax = plt.subplot()
    xs = np.exp(template_xs)
    xlims = (np.min(xs), np.max(xs))
    ys = np.exp(template_history)
    if ylims is None:
        y_pad = (np.max(ys) - np.min(ys)) * 0.1
        ylims = (np.min(ys)-y_pad, np.max(ys)+y_pad)
    ani = animation.FuncAnimation(fig, animfunc, np.linspace(0, niter-1, nframes, dtype=int), 
                fargs=(xs, ys, xlims, ylims, ax, ax.plot))
    plt.close(fig)
    return ani
    
def plot_chis_history(epoch, data_xs, chis_history, niter, nframes, ylims=None):
    fig = plt.figure()
    ax = plt.subplot()
    xs = np.exp(data_xs[epoch,:])
    xlims = (np.min(xs), np.max(xs))
    ys = chis_history[:,epoch,:]
    if ylims is None:
        y_pad = (np.max(ys) - np.min(ys)) * 0.1
        ylims = (np.min(ys)-y_pad, np.max(ys)+y_pad)
    ani = animation.FuncAnimation(fig, animfunc, np.linspace(0, niter-1, nframes, dtype=int), 
                fargs=(xs, ys, xlims, ylims, ax, ax.plot))
    plt.close(fig)
    return ani
        