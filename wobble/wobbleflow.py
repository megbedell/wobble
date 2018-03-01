import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import h5py
import copy
import tensorflow as tf
T = tf.float64

from utils import fit_continuum
#from wobble.interp import interp

c = 2.99792458e8   # m/s

def doppler(v):
    frac = (1. - v/c) / (1. + v/c)
    return tf.sqrt(frac)

class Data(object):
    """
    The data object: contains the spectra and associated data.
    """
    def __init__(self, filename, filepath='../data/', 
                    N = 0, orders = [30], min_flux = 1.):
        filename = filepath + filename
        self.R = len(orders) # number of orders to be analyzed
        self.orders = orders
        with h5py.File(filename) as f:
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
        
        self.M = None
        
    def continuum_normalize(self):
        for r in range(self.R):
            for n in range(self.N):
                self.ys[r][n] -= fit_continuum(self.xs[r][n], self.ys[r][n], self.ivars[r][n])
                
    def make_tensors(self):
        self.ys_tensor = tf.constant(self.ys, dtype=T)
        self.xs_tensor = tf.constant(self.xs, dtype=T)
        self.ivars_tensor = tf.constant(self.ivars, dtype=T)
        
                
class Model(object):
    """
    Keeps track of all components in the model.
    """
    def __init__(self, data):
        self.components = []
        self.data = data
        self.data.make_tensors() # get ready to optimize with TF
        
    def add_star(self, name):
        name = Star(self.data)
        self.components.append(name)
        
    def add_telluric(self, name):
        name = Telluric(self.data)
        self.components.append(name)
        
    def optimize_order(self, r, niter=100, learning_rate_models=0.01, learning_rate_rvs=10.):
        model_xs_tensor = [tf.constant(c.model_xs[r], dtype=T) for c in self.components]
        model_ys_tensor = [tf.constant(c.model_ys[r], dtype=T) for c in self.components]
        rvs_tensor = [tf.Variable(c.rvs_block[r], dtype=T) for c in self.components]
        
        # likelihood calculation:
        nll = tf.constant(0.0, dtype=T)
        for n in range(self.data.N):
            model = tf.zeros_like(self.data.ys_tensor[r][n])
            for i in range(len(self.components)):
                shifted_xs = self.data.xs_tensor[r][n] + tf.log(doppler(rvs_tensor[i][n]))
                model += interp(shifted_xs, model_xs_tensor[i], model_ys_tensor[i])
            nll += 0.5*tf.reduce_sum((data.ys_tensor[r][n] - model)**2 * ivars_tensor[r][n])
            
        # set up optimizers:    
        grad_models = tf.gradients(nll, model_ys_tensor)
        grad_rvs = tf.gradients(nll, rvs_tensor)
        opt_models = tf.train.AdagradOptimizer(learning_rate_models).minimize(nll, var_list=model_ys_tensor)
        opt_rvs = tf.train.AdagradOptimizer(learning_rate_rvs).minimize(nll, var_list=rvs_tensor)
        
        # optimize:
        with tf.Session() as session:
            print "--- ORDER {0} ---".format(r)
            session.run(tf.global_variables_initializer())    
            nll_history = []
            for i in range(niter):
                session.run(opt_rvs)
                nll_history.append(session.run(nll))
                if (i % 10) == 0:
                    print "iter {0}: optimizing RVs".format(i)
                    print "nll: {0:.2e}".format(nll_history[-1])
                session.run(opt_models)
                nll_history.append(session.run(nll))
                if (i % 10) == 0:
                    print "iter {0}: optimizing models".format(i)
                    print "nll: {0:.2e}".format(nll_history[-1])            

            # save outputs:
            model_soln = session.run(model_ys_tensor)
            for m,c in zip(model_soln, self.components):
                c.model_ys = np.copy(m)
            rvs_soln = session.run(rvs_tensor)
            for v,c in zip(rvs_soln, self.components):
                c.rvs_block[r] = np.copy(v)
                
    def optimize_all(self, **kwargs):
        for r in range(self.data.R):
            self.optimize_order(r, **kwargs)
            if (r % 5) == 0:
                self.save_results('results_order{0}.hdf5'.format(r))
                
    def save_results(self, filename):
        print "saving..."
        # TODO: write this code
                
                
                
class Component(object):
    """
    Generic class for an additive component in the spectral model.
    """
    def __init__(self, data):
        self.rvs_block = [np.zeros(data.N) for r in range(data.R)]
        
    def initialize_model(self, data):
        # TODO: write this code!!
        self.model_xs = [np.zeros(100.) for r in range(data.R)]
        self.model_ys = [np.zeros(100.) for r in range(data.R)]
                
class Star(Component):
    """
    A star (or generic celestial object)
    """
    def __init__(self, data):
        self.rvs_block = [-np.copy(data.pipeline_rvs)+np.mean(data.pipeline_rvs) for r in range(data.R)]
        self.initialize_model(data)
    
class Telluric(Component):
    """
    Sky absorption
    """
    def __init__(self, data):
        self.initialize_model(data)

    