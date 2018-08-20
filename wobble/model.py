import numpy as np
from tqdm import tqdm
import sys
import tensorflow as tf
T = tf.float64

from .utils import bin_data
from .interp import interp
from .history import History

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