import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import animation
from tqdm import tqdm
import sys
import h5py
import copy
import tensorflow as tf
T = tf.float64

from .utils import fit_continuum
from .interp import interp
from .wobble import star as star_obj

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
    _SESSION = tf.Session()
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
                    N = 0, orders = [30], min_flux = 1.):
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
        
        # HACK -- for initializing models
        self.wobble_obj = star_obj(filename, filepath=filepath, orders=orders)
        
        # convert to tensors
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
        self.data = data
        
    def __str__(self):
        return "Model consisting of the following components: {0}".format(self.component_names)
        
    def add_star(self, name):
        c = Star(name, self.data)
        self.components.append(c)
        self.component_names.append(name)
        
    def add_telluric(self, name):
        c = Telluric(name, self.data)
        self.components.append(c)
        self.component_names.append(name)
                
    def save(self, filename):
        print("saving...")
        # TODO: write this code
                                
class Component(object):
    """
    Generic class for an additive component in the spectral model.
    """
    def __init__(self, name, data):
        self.name = name
        self.rvs_block = [tf.Variable(np.zeros(data.N), dtype=T, name='rvs_order{0}'.format(r)) for r in range(data.R)]
        self.model_xs = [None for r in range(data.R)] # this will be replaced
        self.model_ys = [None for r in range(data.R)] # this will be replaced
        self.learning_rate_rvs = 10. # default
        self.learning_rate_model = 0.01 # default
        
    def shift_and_interp(self, r, xs, rvs):
        """
        Apply Doppler shift of rvs to the model at order r and output interpolated values at xs.
        """
        shifted_xs = xs + tf.log(doppler(rvs[:, None]))
        return interp(shifted_xs, self.model_xs[r], self.model_ys[r]) 
        
    def make_optimizers(self, r, nll, learning_rate_rvs=None, 
            learning_rate_model=None):
        # TODO: make each one an R-length list rather than overwriting each order?
        if learning_rate_rvs == None:
            learning_rate_rvs = self.learning_rate_rvs
        if learning_rate_model == None:
            learning_rate_model = self.learning_rate_model
        self.gradients_model = tf.gradients(nll, self.model_ys[r])
        self.gradients_rvs = tf.gradients(nll, self.rvs_block[r])
        self.opt_model = tf.train.AdamOptimizer(learning_rate_model).minimize(nll, 
                            var_list=[self.model_ys[r]])
        self.opt_rvs = tf.train.AdamOptimizer(learning_rate_rvs).minimize(nll, 
                            var_list=[self.rvs_block[r]])   
        self.saver = tf.train.Saver([self.model_xs[r], self.model_ys[r], self.rvs_block[r]])  
        

class Star(Component):
    """
    A star (or generic celestial object)
    """
    def __init__(self, name, data):
        Component.__init__(self, name, data)
        starting_rvs = -np.copy(data.pipeline_rvs)+np.mean(data.pipeline_rvs)
        self.rvs_block = [tf.Variable(starting_rvs, dtype=T, name='rvs_order{0}'.format(r)) for r in range(data.R)]
    
    def initialize_model(self, r, data):
        # hackety fucking hack
        for r in range(data.R):
            data.wobble_obj.initialize_model(r, 'star')
        self.model_xs[r] = tf.Variable(data.wobble_obj.model_xs_star[0], dtype=T, name='model_xs_star')
        self.model_ys[r] = tf.Variable(data.wobble_obj.model_ys_star[0], dtype=T, name='model_ys_star')

                            
class Telluric(Component):
    """
    Sky absorption
    """
    def __init__(self, name, data):
        Component.__init__(self, name, data)
        self.learning_rate_model = 0.1
        
    def initialize_model(self, r, data):
        # hackety fucking hack
        for r in range(data.R):
            data.wobble_obj.initialize_model(r, 't')
        self.model_xs[r] = tf.Variable(data.wobble_obj.model_xs_t[0], dtype=T, name='model_xs_t')
        self.model_ys[r] = tf.Variable(data.wobble_obj.model_ys_t[0], dtype=T, name='model_ys_t')
        '''''
        session = get_session()
        session.run(tf.variables_initializer([self.model_xs[r], self.model_ys[r]]))
        session.run(tf.variables_initializer(self.rvs_block))
        '''
        

def optimize_order(model, data, r, niter=100, save_every=100, output_history=False):
    for c in model.components:
        if c.model_ys[r] is None:
            c.initialize_model(r, data)
            
    if output_history:
        nll_history = np.empty(niter*2)
        rvs_history = np.empty((niter, data.N))
        model_history = np.empty((niter, int(model.components[0].model_ys[r].shape[0])))
    
    # likelihood calculation:
    synth = tf.zeros_like(data.ys[r])
    for c in model.components:
        synth += c.shift_and_interp(r, data.xs[r], c.rvs_block[r])
    nll = 0.5*tf.reduce_sum(tf.square(data.ys[r] - synth) * data.ivars[r])
        
    # set up optimizers: 
    for c in model.components:
        c.make_optimizers(r, nll)

    # optimize:
    print("--- ORDER {0} ---".format(r))
    session = get_session()
    session.run(tf.global_variables_initializer())    # should this be in get_session?
    for i in tqdm(range(niter)):
        for c in model.components:            
            session.run(c.opt_rvs)
        if output_history: 
            nll_history[2*i] = session.run(nll)
        for c in model.components:            
            session.run(c.opt_model)
            if (i+1 % save_every == 0):
                c.saver.save(session, "tf_checkpoints/{0}_order{1}".format(c.name, r), global_step=i)
        if output_history: 
            nll_history[2*i+1] = session.run(nll)
            model_soln = session.run(model.components[0].model_ys[r])
            rvs_soln = session.run(model.components[0].rvs_block[r])
            model_history[i,:] = np.copy(model_soln)
            rvs_history[i,:] = np.copy(rvs_soln)
        
    return nll_history, rvs_history, model_history # hack

def optimize_orders(model, data, **kwargs):
    for r in range(data.R):
        optimize_order(model, data, r, **kwargs)
        if (r % 5) == 0:
            model.save('results_order{0}.hdf5'.format(r))
            
'''''
def init_plot():
    ln.set_data([], [])
    return ln,
            
def update_plot(i, xs, ys):
    ln.set_data(xs, ys[i,:])
    return ln,
            
def plot_rv_history(data, nll_history, rvs_history, model_history, niter):
    fig = plt.figure()
    ax = plt.axes(xlim=(-20000, 20000), ylim=(-300, 300))
    ln, = ax.plot([], [], 'ko')
    xs = data.pipeline_rvs
    ys = rvs_history + np.repeat([data.pipeline_rvs], 100, axis=0)
    ani = animation.FuncAnimation(fig, update_plot, init_func=init_plot, frames=range(niter), 
                fargs=(xs, ys), interval=1, blit=True)
    plt.show()
'''
            
def plot_rv_history(data, nll_history, rvs_history, model_history, niter):
    for i in range(niter // 10):
        plt.scatter(data.pipeline_rvs, rvs_history[i,:] + data.pipeline_rvs, color='blue', alpha=i/10.)
    plt.show()