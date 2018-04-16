import numpy as np
from scipy.optimize import minimize
import h5py
import copy

__all__ = ["Results"]

class Results(object):
    """
    Holdover from original wobble - currently used only for saving/restoring and for combining order RVs
    """
    
    def __init__(self, N = 0, R = 0):
        self.R = R # number of orders to be analyzed
        self.N = N # number of epochs        
        
        # set up attributes for optimization:
        self.rvs_star = [np.zeros(self.N) for r in range(self.R)]
        self.rvs_t = [np.zeros(self.N) for r in range(self.R)]
        self.model_ys_star = [None for r in range(self.R)]
        self.model_xs_star = [None for r in range(self.R)]
        self.model_ys_t = [None for r in range(self.R)]
        self.model_xs_t = [None for r in range(self.R)]
        self.ivars_star = [None for r in range(self.R)]
        self.ivars_t = [None for r in range(self.R)]
        
        # set up design matrix:
        self.M = None
        

        
    def save_results(self, filename):
        max_len = np.max([len(x) for x in self.model_xs_star])
        for r in range(self.R): # resize to make rectangular arrays bc h5py is infuriating
            self.model_xs_star[r] = np.append(self.model_xs_star[r], np.zeros(max_len - len(self.model_xs_star[r])))
            self.model_ys_star[r] = np.append(self.model_ys_star[r], np.zeros(max_len - len(self.model_ys_star[r])))
            self.model_xs_t[r] = np.append(self.model_xs_t[r], np.zeros(max_len - len(self.model_xs_t[r])))
            self.model_ys_t[r] = np.append(self.model_ys_t[r], np.zeros(max_len - len(self.model_ys_t[r])))
        with h5py.File(filename,'w') as f:
            dset = f.create_dataset('rvs_star', data=self.rvs_star)
            dset = f.create_dataset('ivars_star', data=self.ivars_star)
            dset = f.create_dataset('rvs_t', data=self.rvs_t)
            dset = f.create_dataset('ivars_t', data=self.ivars_t)
            dset = f.create_dataset('model_xs_star', data=self.model_xs_star)
            dset = f.create_dataset('model_ys_star', data=self.model_ys_star)
            dset = f.create_dataset('model_xs_t', data=self.model_xs_t)
            dset = f.create_dataset('model_ys_t', data=self.model_ys_t)
            try:
                dset = f.create_dataset('time_rvs', data=self.time_rvs)
                dset = f.create_dataset('order_rvs', data=self.order_rvs)
                dset = f.create_dataset('order_sigmas', data=self.order_sigmas)
            except:
                pass
            
    def load_results(self, filename):
        with h5py.File(filename) as f:
            self.rvs_star = np.copy(f['rvs_star'])
            self.rvs_t = np.copy(f['rvs_t'])
            self.model_xs_star = np.copy(f['model_xs_star']).tolist()
            self.model_ys_star = np.copy(f['model_ys_star']).tolist()
            self.model_xs_t = np.copy(f['model_xs_t']).tolist()
            self.model_ys_t = np.copy(f['model_ys_t']).tolist()
            self.ivars_star = np.copy(f['ivars_star'])
            self.ivars_t = np.copy(f['ivars_t'])
            try:
                self.time_rvs = np.copy(f['time_rvs'])
                self.order_rvs = np.copy(f['order_rvs'])
                self.order_sigmas = np.copy(f['order_sigmas'])
            except:
                print("warning: you may need to run optimize_sigmas()" )              
        for r in range(self.R): # trim off that padding
            self.model_xs_star[r] = np.trim_zeros(np.asarray(self.model_xs_star[r]), 'b')
            self.model_ys_star[r] = np.trim_zeros(np.asarray(self.model_ys_star[r]), 'b')
            self.model_xs_t[r] = np.trim_zeros(np.asarray(self.model_xs_t[r]), 'b')
            self.model_ys_t[r] = np.trim_zeros(np.asarray(self.model_ys_t[r]), 'b')
            
    def pack_rv_pars(self, time_rvs, order_rvs, order_sigmas):
        rv_pars = np.append(time_rvs, order_rvs)
        rv_pars = np.append(rv_pars, order_sigmas)
        return rv_pars
    
    def unpack_rv_pars(self, rv_pars):
        self.time_rvs = np.copy(rv_pars[:self.N])
        self.order_rvs = np.copy(rv_pars[self.N:self.R + self.N])
        self.order_sigmas = np.copy(rv_pars[self.R + self.N:])
        return self.time_rvs, self.order_rvs, self.order_sigmas
        
    def lnlike_sigmas(self, sigmas, return_rvs = False, restart = False):
        assert len(sigmas) == self.R
        M = self.get_design_matrix(restart = restart)
        something = np.zeros_like(M[0,:])
        something[self.N:] = 1. / self.R # last datum will be mean of order velocities is zero
        M = np.append(M, something[None, :], axis=0) # last datum
        Rs, Ns = self.get_index_lists()
        ivars = 1. / ((1. / self.ivars_star) + sigmas[Rs]**2) # not zero-safe
        ivars = ivars.flatten()
        ivars = np.append(ivars, 1.) # last datum: MAGIC
        MTM = np.dot(M.T, ivars[:, None] * M)
        ys = self.rvs_star.flatten()
        ys = np.append(ys, 0.) # last datum
        MTy = np.dot(M.T, ivars * ys)
        xs = np.linalg.solve(MTM, MTy)
        resids = ys - np.dot(M, xs)
        lnlike = -0.5 * np.sum(resids * ivars * resids - np.log(2. * np.pi * ivars))
        if return_rvs:
            return lnlike, xs[:self.N], xs[self.N:] # must be synchronized with get_design_matrix(), and last datum removal
        return lnlike
        
    def opposite_lnlike_sigmas(self, pars, restart = False):
        return -1. * self.lnlike_sigmas(pars, restart = restart)    

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
        
    def optimize_sigmas(self, restart = False):
        # initial guess
        x0_order_rvs = np.median(self.rvs_star, axis=1)
        x0_time_rvs = np.median(self.rvs_star - np.tile(x0_order_rvs[:,None], (1, self.N)), axis=0)
        rv_predictions = np.tile(x0_order_rvs[:,None], (1,self.N)) + np.tile(x0_time_rvs, (self.R,1))
        x0_sigmas = np.log(np.var(self.rvs_star - rv_predictions, axis=1))
        # optimize
        print("optimize_sigmas: optimizing...")
        soln_sigmas = minimize(self.opposite_lnlike_sigmas, x0_sigmas, args=(restart), method='BFGS', options={'disp':True})['x'] # HACK
        # save results
        lnlike, rvs_N, rvs_R = self.lnlike_sigmas(soln_sigmas, return_rvs=True)
        self.order_rvs = rvs_R
        self.time_rvs = rvs_N
        self.order_sigmas = soln_sigmas