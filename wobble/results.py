import numpy as np
from scipy.optimize import minimize
import h5py
import tensorflow as tf
T = tf.float64

from .utils import get_session

COMPONENT_NP_ATTRS = ['K', 'r', 'rvs_fixed', 'scale_by_airmass', 'learning_rate_rvs', 'learning_rate_template', 
                      'learning_rate_basis', 'L1_template', 'L2_template', 'L1_basis_vectors', 
                      'L2_basis_vectors', 'L2_basis_weights']
COMPONENT_TF_ATTRS = ['rvs', 'ivars', 'template_xs', 'template_ys', 'basis_vectors', 'basis_weights']

class Results(object):
    """
    Stores RV & template results across all orders.
    """
    def __init__(self, data=None, filename=None):
        if filename is None:
            self.component_names = None
            self.R = data.R
            self.N = data.N
            self.orders = data.orders
            self.origin_file = data.origin_file
            return
        if data is None:
            self.read(filename)
            return
        print("Results: must supply either data or filename keywords.")
                
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
                if np.isin(attr, self.attrs_to_resize): # pad with zeros to make rectangular arrays bc h5py is infuriating
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