import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import animation
import h5py
import copy
import tensorflow as tf
T = tf.float64

from .utils import fit_continuum
from .interp import interp
from .wobble import star as star_obj

speed_of_light = 2.99792458e8   # m/s

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
        with h5py.File(filepath+filename) as f:
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
        
    def continuum_normalize(self):
        for r in range(self.R):
            for n in range(self.N):
                self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n])
                
    def make_tensors(self):
        self.ys_tensor = [tf.constant(y, dtype=T) for y in self.ys]
        self.xs_tensor = [tf.constant(x, dtype=T) for x in self.xs]
        self.ivars_tensor = [tf.constant(i, dtype=T) for i in self.ivars]
        
                
class Model(object):
    """
    Keeps track of all components in the model.
    """
    def __init__(self, data):
        self.components = []
        self.data = data
        
    def add_star(self, name):
        name = Star(self.data)
        self.components.append(name)
        
    def add_telluric(self, name):
        name = Telluric(self.data)
        self.components.append(name)
                
    def save(self, filename):
        print("saving...")
        # TODO: write this code
                                
class Component(object):
    """
    Generic class for an additive component in the spectral model.
    """
    def __init__(self, data):
        self.rvs_block = [np.zeros(data.N) for r in range(data.R)]
        self.model_xs = [np.zeros(100) for r in range(data.R)]
        self.model_ys = [np.zeros(100) for r in range(data.R)]
        
        
    def make_tensors(self):
        """
        Convert attributes to TensorFlow variables in preparation for optimizing.
        """
        self.rvs_block_tensor = [tf.Variable(v, dtype=T) for v in self.rvs_block]
        self.model_ys_tensor = [tf.Variable(m, dtype=T) for m in self.model_ys]
        self.model_xs_tensor = [tf.Variable(m, dtype=T) for m in self.model_xs]
        
    def shift_and_interp(self, r, xs, rv):
        """
        Apply a Doppler shift to the model at order r and output interpolated values at xs.
        """
        shifted_xs = xs + tf.log(doppler(rv))
        return interp(shifted_xs, self.model_xs_tensor[r], self.model_ys_tensor[r])        
        

class Star(Component):
    """
    A star (or generic celestial object)
    """
    def __init__(self, data):
        Component.__init__(self, data)
        self.rvs_block = [-np.copy(data.pipeline_rvs)+np.mean(data.pipeline_rvs) for r in range(data.R)]
        self.initialize_model(data)
    
    def initialize_model(self, data):
        # hackety fucking hack
        for r in range(data.R):
            data.wobble_obj.initialize_model(r, 'star')
        self.model_xs = data.wobble_obj.model_xs_star
        self.model_ys = data.wobble_obj.model_ys_star        

class Telluric(Component):
    """
    Sky absorption
    """
    def __init__(self, data):
        Component.__init__(self, data)
        self.initialize_model(data)
        
    def initialize_model(self, data):
        # hackety fucking hack
        for r in range(data.R):
            data.wobble_obj.initialize_model(r, 't')
        self.model_xs = data.wobble_obj.model_xs_t
        self.model_ys = data.wobble_obj.model_ys_t
        

def optimize_order(model, data, r, niter=100, learning_rate_models=0.01, learning_rate_rvs=10., save_history=False):
    if not hasattr(data, 'xs_tensor'):
        data.make_tensors()
    for c in model.components:
        if not hasattr(c, 'model_xs_tensor'):
            c.make_tensors()
            
    if save_history:
        nll_history = np.empty(niter*2)
        rvs_history = np.empty((niter, data.N))
        model_history = np.empty((niter, len(model.components[0].model_ys[r])))
    
    # likelihood calculation:
    nll = tf.constant(0.0, dtype=T)
    for n in range(data.N):
        synth = tf.zeros_like(data.ys_tensor[r][n])
        for c in model.components:
            synth += c.shift_and_interp(r, data.xs_tensor[r][n], c.rvs_block_tensor[r][n])
        nll += 0.5*tf.reduce_sum((data.ys_tensor[r][n] - synth)**2 * data.ivars_tensor[r][n])
        
    # set up optimizers:    
    model_ys_tensor = [c.model_ys_tensor[r] for c in model.components]
    rvs_tensor = [c.rvs_block_tensor[r] for c in model.components]
    grad_models = tf.gradients(nll, model_ys_tensor)
    grad_rvs = tf.gradients(nll, rvs_tensor)
    opt_models = tf.train.AdagradOptimizer(learning_rate_models).minimize(nll, var_list=model_ys_tensor)
    #opt_rvs = tf.train.AdagradOptimizer(learning_rate_rvs).minimize(nll, var_list=rvs_tensor)
    
    # optimize:
    with tf.Session() as session:
        print("--- ORDER {0} ---".format(r))
        session.run(tf.global_variables_initializer())    
        for i in range(niter):
            #session.run(opt_rvs)
            if save_history: 
                nll_history[2*i] = session.run(nll)
                if (i % 10) == 0:
                    print("iter {0}: optimizing RVs".format(i))
                    print("nll: {0:.2e}".format(nll_history[2*i]))
            session.run(opt_models)
            if save_history: 
                nll_history[2*i+1] = session.run(nll)
                if (i % 10) == 0:
                    print("iter {0}: optimizing models".format(i))
                    print("nll: {0:.2e}".format(nll_history[2*i+1]))
                model_soln = session.run(model_ys_tensor)
                rvs_soln = session.run(rvs_tensor)
                model_history[i,:] = np.copy(model_soln[0])
                rvs_history[i,:] = np.copy(rvs_soln[0])
                      
        # save outputs:
        model_soln = session.run(model_ys_tensor)
        for m,c in zip(model_soln, model.components):
            c.model_ys = np.copy(m)
        rvs_soln = session.run(rvs_tensor)
        for v,c in zip(rvs_soln, model.components):
            c.rvs_block[r] = np.copy(v)
            
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