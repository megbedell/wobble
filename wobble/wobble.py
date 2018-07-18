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

speed_of_light = 2.99792458e8   # m/s
DATA_NP_ATTRS = ['N', 'R', 'origin_file', 'orders', 'dates', 'bervs', 'drifts', 'airms', 'pipeline_rvs']
DATA_TF_ATTRS = ['xs', 'ys', 'ivars']
MODEL_ATTRS = ['component_names'] # not actually used but defined for completeness
COMPONENT_NP_ATTRS = ['K', 'learning_rate_rvs', 'learning_rate_template', 'learning_rate_basis', 'L1_template',
                    'L2_template', 'L1_basis_vectors', 'L2_basis_vectors', 'L1_basis_weights',
                    'L2_basis_weights']
COMPONENT_TF_ATTRS = ['rvs_block', 'template_xs', 'template_ys', 'basis_vectors', 'basis_weights']

__all__ = ["get_session", "doppler", "Data", "Model", "History", "Results", "optimize_order", "optimize_orders"]

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
        self.data = data
        
    def __str__(self):
        string = 'Model consisting of the following components: '
        for c in self.components:
            string += '\n{0}: '.format(c.name)
            if c.rvs_fixed:
                string += 'RVs fixed; '
            else:
                string += 'RVs variable; '
            string += '{0} variable basis components'.format(c.K)
        return string
        
    def synthesize(self, r):
        synth = tf.zeros_like(self.data.xs[r])
        for c in self.components:
            synth += c.synthesize(r, self.data.xs[r], c.rvs_block[r])
        return synth
        
    def add_star(self, name, rvs_fixed=False, variable_bases=0):
        if np.isin(name, self.component_names):
            print("The model already has a component named {0}. Try something else!".format(name))
            return
        c = Star(name, self.data, rvs_fixed=rvs_fixed, variable_bases=variable_bases)
        self.components.append(c)
        self.component_names.append(name)
        
    def add_telluric(self, name, rvs_fixed=True, variable_bases=0):
        if np.isin(name, self.component_names):
            print("The model already has a component named {0}. Try something else!".format(name))
            return
        c = Telluric(name, self.data, rvs_fixed=rvs_fixed, variable_bases=variable_bases)
        self.components.append(c)
        self.component_names.append(name)
                                
class Component(object):
    """
    Generic class for an additive component in the spectral model.
    """
    def __init__(self, name, data, rvs_fixed=False, variable_bases=0):
        self.name = name
        self.K = variable_bases # number of variable basis vectors
        self.rvs_block = [tf.Variable(np.zeros(data.N), dtype=T, name='rvs_order{0}'.format(r)) for r in range(data.R)]
        self.rvs_fixed = rvs_fixed
        self.template_xs = [None for r in range(data.R)] # this will be replaced
        self.template_ys = [None for r in range(data.R)] # this will be replaced
        self.basis_vectors = [None for r in range(data.R)] # this will be replaced
        self.basis_weights = [None for r in range(data.R)] # this will be replaced
        self.learning_rate_rvs = 10. # default
        self.learning_rate_template = 0.01 # default
        self.learning_rate_basis = 0.01 # default
        self.L1_template = 0.
        self.L2_template = 0.
        self.L1_basis_vectors = 0.
        self.L2_basis_vectors = 0.
        self.L1_basis_weights = 0.
        self.L2_basis_weights = 0.
        
    def shift_and_interp(self, r, xs, rvs):
        """
        Apply Doppler shift of rvs to the model at order r and output interpolated values at xs.
        """
        shifted_xs = xs + tf.log(doppler(rvs[:, None]))
        return interp(shifted_xs, self.template_xs[r], self.template_ys[r]) 
        
    def synthesize(self, r, xs, rvs):
        """
        Output synthesized spectrum at log(wave)s xs and order r.
        """
        synth = self.shift_and_interp(r, xs, rvs)
        if self.K > 0:
            synth += tf.matmul(self.basis_weights[r], self.basis_vectors[r])
        return synth
        
    def initialize_template(self, r, data, other_components=None, template_xs=None):
        """
        Doppler-shift data into component rest frame, subtract off other components, 
        and average to make a composite spectrum.
        """
        shifted_xs = data.xs[r] + tf.log(doppler(self.rvs_block[r][:, None])) # component rest frame
        if template_xs is None:
            dx = tf.constant(2.*(np.log(6000.01) - np.log(6000.)), dtype=T) # log-uniform spacing
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
        session.run(tf.global_variables_initializer())  # TODO: is it bad to have this twice? am I overwriting?
         
        
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

class Star(Component):
    """
    A star (or generic celestial object)
    """
    def __init__(self, name, data, rvs_fixed=False, variable_bases=0):
        Component.__init__(self, name, data, rvs_fixed=rvs_fixed, variable_bases=variable_bases)
        starting_rvs = np.copy(data.bervs) - np.mean(data.bervs)
        self.rvs_block = [tf.Variable(starting_rvs, dtype=T, name='rvs_order{0}'.format(r)) for r in range(data.R)] 
        self.L1_template = 1.e4 # magic number set to 10x the threshold where it just begins to affect results
        self.L2_template = 1.e5 # magic number set to 10x the threshold where it just begins to affect results
                            
class Telluric(Component):
    """
    Sky absorption
    """
    def __init__(self, name, data, rvs_fixed=True, variable_bases=0):
        Component.__init__(self, name, data, rvs_fixed=rvs_fixed, variable_bases=variable_bases)
        self.airms = tf.constant(data.airms, dtype=T)
        self.learning_rate_template = 0.1 
        self.L1_template = 1.e5 # magic number set to 10x the threshold where it just begins to affect results
        self.L2_template = 1.e6 # magic number set to 10x the threshold where it just begins to affect results        
        self.L2_basis_vectors = 1.e4 # magic number
        self.L2_basis_weights = 1. # magic number
        
    def synthesize(self, r, xs, rvs):
        synth = Component.synthesize(self, r, xs, rvs)
        return tf.einsum('n,nm->nm', self.airms, synth)
        
class History(object):
    """
    Information about optimization history of a single order stored in numpy arrays/lists
    """   
    def __init__(self, model, data, r, niter, filename=None):
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
        
class Results(object):
    def __init__(self, model=None, data=None, filename=None):
        if data is not None and model is not None:
            self.copy_data(data)
            self.copy_model(model)
        elif filename is not None:
            self.read(filename)
        else:
            print("Error: Results() object must have model and data keywords OR filename keyword to initialize.")            
            
    def copy_data(self, data):
        for attr in DATA_NP_ATTRS:
            setattr(self, attr, getattr(data,attr))   
        session = get_session()
        for attr in DATA_TF_ATTRS:
            setattr(self, attr, session.run(getattr(data,attr)))
            
    def copy_model(self, model):
        self.component_names = model.component_names
        session = get_session()
        self.ys_predicted = [session.run(model.synthesize(r)) for r in range(self.R)]
        for c in model.components:
            basename = c.name+'_'
            ys_predicted = [session.run(c.synthesize(r, model.data.xs[r], c.rvs_block[r])) for r in range(self.R)]
            setattr(self, basename+'ys_predicted', ys_predicted)
            for attr in COMPONENT_NP_ATTRS:
                setattr(self, basename+attr, getattr(c,attr))
            for attr in COMPONENT_TF_ATTRS:
                try:
                    setattr(self, basename+attr, session.run(getattr(c,attr)))
                except: # catch when basis vectors are Nones
                    assert c.K == 0, "Results: copy_model() failed on attribute {0}".format(attr)
                        
    def read(self, filename):
        print("Results: reading from {0}".format(filename))
        # TODO: WRITE THIS
    
    def write(self, filename):
        print("Results: writing to {0}".format(filename))
        # TODO: WRITE THIS        
            

def optimize_order(model, data, r, niter=100, save_every=100, save_history=False, basename='wobble'):
    '''
    optimize the model for order r in data
    '''      
    for c in model.components:
        if c.template_ys[r] is None:
            c.initialize_template(r, data, other_components=[x for x in model.components if x!=c])         
                     
    if save_history:
        history = History(model, data, r, niter)
        
    results = Results(model=model, data=data) # initialize results
        
    # likelihood calculation:
    synth = model.synthesize(r)
    chis = (data.ys[r] - synth) * tf.sqrt(data.ivars[r])
    nll = 0.5*tf.reduce_sum(tf.square(data.ys[r] - synth) * data.ivars[r])
    
    # regularization:
    for c in model.components:
        nll += c.L1_template * tf.reduce_sum(tf.abs(c.template_ys[r]))
        nll += c.L2_template * tf.reduce_sum(tf.square(c.template_ys[r]))
        if c.K > 0:
            nll += c.L1_basis_vectors * tf.reduce_sum(tf.abs(c.basis_vectors[r]))
            nll += c.L2_basis_vectors * tf.reduce_sum(tf.square(c.basis_vectors[r]))
            nll += c.L1_basis_weights * tf.reduce_sum(tf.abs(c.basis_weights[r]))
            nll += c.L2_basis_weights * tf.reduce_sum(tf.square(c.basis_weights[r]))
        
    # set up optimizers: 
    for c in model.components:
        c.make_optimizers(r, nll)

    session = get_session()
    session.run(tf.global_variables_initializer())
        
    # optimize:
    for i in tqdm(range(niter)):
        if save_history:
            history.save_iter(model, data, i, nll, chis)           
        for c in model.components:
            if not c.rvs_fixed:            
                session.run(c.opt_rvs) # optimize RVs
            session.run(c.opt_template) # optimize mean template
            if c.K > 0:
                session.run(c.opt_basis) # optimize variable components
        if (i+1 % save_every == 0): # progress save
            results.copy_model(model) # update
            results.write(basename+'_results.hdf5'.format(r))
            if save_history:
                history.write(basename+'_o{0}_history.hdf5'.format(r))
    if save_history: # final post-optimization save
        history.write(basename+'_o{0}_history.hdf5'.format(r))
    results.copy_model(model) # update
    return results

def optimize_orders(model, data, **kwargs):
    """
    optimize model for all orders in data
    """
    session = get_session()
    session.run(tf.global_variables_initializer())    # should this be in get_session?
    for r in range(data.R):
        print("--- ORDER {0} ---".format(r))
        results = optimize_order(session, model, data, r, **kwargs)
        #if (r % 5) == 0:
        #    results.write('results_order{0}.hdf5'.format(r))
    results.write('results.hdf5')        