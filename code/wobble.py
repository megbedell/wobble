import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec

from scipy.optimize import fmin_cg, minimize
import h5py
from utils import fit_continuum
import copy
from twobody.wrap import cy_rv_from_elements
from twobody import KeplerOrbit
from astropy.time import Time
import astropy.units as u

c = 2.99792458e8   # m/s

class star(object):
    """
    The main interface to the wobble package
    
    Example use case:
    import wobble
    a = wobble.star('hip30037.hdf5', e2ds = False)
    a.optimize(niter=10)
    
    or:
    import numpy as np
    import wobble
    a = wobble.star('hip54287_e2ds.hdf5', orders=np.arange(72), N=40)
    a.optimize(niter=20)
    a.save_results('../results/hip54287_results.hdf5')

    Args: 
        filename: The name of the file which contains your radial velocity data (for now, must be 
            an HDF5 file containing particular information in HARPS format). 
        filepath: The directory relative to your current working directory where your 
            RV data are stored (default: ``../data/``)
        wl_lower: The lowest wavelength, in angstroms in air, for the region of the spectrum you 
            would like to analyze (default: ``5900``)
        wl_upper: The highest wavelength, in angstroms in air, for the region of the spectrum you 
            would like to analyze (default: ``5900``)
        N: The number of epochs of RV data to analyze. Will select the first N epochs (default: ``16``).

    """
    
    def __init__(self, filename, filepath='../data/', wl_lower = 5900, wl_upper = 6000, 
                    N = 0, e2ds = True, orders = [30], min_flux = 1.):
        filename = filepath + filename
        self.R = len(orders) # number of orders to be analyzed
        if e2ds:
            self.orders = orders
            with h5py.File(filename) as f:
                if N < 1:
                    self.N = len(f['date']) # all epochs
                else:
                    self.N = N
                self.data = [f['data'][i][:self.N,:] for i in orders]
                self.data_xs = [np.log(f['xs'][i][:self.N,:]) for i in orders]
                self.ivars = [f['ivars'][i][:self.N,:] for i in orders]
                self.drs_rvs = np.copy(f['true_rvs'])[:self.N]
                self.dates = np.copy(f['date'])[:self.N]
                self.bervs = np.copy(f['berv'])[:self.N] * -1.e3
                self.drifts = np.copy(f['drift'])[:self.N]
                self.airms = np.copy(f['airm'])[:self.N]
        else:
            self.wavelength_lower = wl_lower
            self.wavelength_upper = wl_upper
            self.R = 1
        
            with h5py.File(filename) as f:
                inds = (f['xs'][:] > self.wavelength_lower) & (f['xs'][:] < self.wavelength_upper)
                N_all, M_all = np.shape(inds)
                data = np.copy(f['data'])[inds]
                self.data = [np.reshape(data, (N_all, -1))[:self.N,:]]
                data_xs = np.log(np.copy(f['xs'][inds]))
                self.data_xs = [np.reshape(data_xs, (N_all, -1))[:self.N,:]]
                ivars = np.copy(f['ivars'])[inds]
                self.ivars = [np.reshape(ivars, (N_all, -1))[:self.N,:]]
                self.drs_rvs = np.copy(f['true_rvs'])[:self.N]
                self.bervs = np.copy(f['berv'])[:self.N] * -1.e3
                self.drifts = np.copy(f['drift'])[:self.N]
                self.airms = np.copy(f['airm'])[:self.N]

        # mask out bad data:
        for r in range(self.R):
            bad = np.where(self.data[r] < min_flux)
            self.data[r][bad] = min_flux
            self.ivars[r][bad] = 0.
            
        # log and normalize:
        self.data = np.log(self.data) 
        self.continuum_normalize() 
        
        # set up attributes for optimization:
        self.rvs_star = [-np.copy(self.drs_rvs)+np.mean(self.drs_rvs) for r in range(self.R)]
        self.rvs_t = [np.zeros(self.N) for r in range(self.R)]
        self.model_ys_star = [np.zeros(self.N) for r in range(self.R)] # not the right shape but whatevs, it's overwritten in the initialization of optimize_order
        self.model_xs_star = [np.zeros(self.N) for r in range(self.R)]
        self.model_ys_t = [np.zeros(self.N) for r in range(self.R)]
        self.model_xs_t = [np.zeros(self.N) for r in range(self.R)]
        #self.soln_star = [np.zeros(self.N) for r in range(self.R)]
        #self.soln_t = [np.zeros(self.N) for r in range(self.R)]
        self.ivars_star = [np.zeros(self.N) for r in range(self.R)]
        self.ivars_t = [np.zeros(self.N) for r in range(self.R)]
        
        self.M = None
        
    def continuum_normalize(self):
        for r in range(self.R):
            for n in range(self.N):
                self.data[r][n] -= fit_continuum(self.data_xs[r][n], self.data[r][n], self.ivars[r][n])
                
       
    def doppler(self, v):
        frac = (1. - v/c) / (1. + v/c)
        return np.sqrt(frac)

    def gamma(self, v):
        return 1. / np.sqrt(1. - (v/c) ** 2)

    def dlndopplerdv(self, v):
        dv = self.doppler(v)
        return -1. * self.gamma(v) / (2. * c) * (1. / dv  + dv)

    def state(self, v, xs, xps):
        '''
        outputs: (M, Mp, v, xs, ms, mps, ehs, bees, seas)
        M and Mp are the lengths of data and model wavelength grids
        v is the RV
        xs is the wavelength values of the data grid
        ms is the data index m at which there is an interpolated model value
        mps is the model index m' from which we interpolate to ms
        ehs, bees, and seas go into the coefficients of interpolation
        '''
        # every input must be 1-d
        M = len(xs)
        Mp = len(xps)
        xps_shifted = xps + np.log(self.doppler(v))
        ms = np.arange(M)
        mps = np.searchsorted(xps_shifted, xs, side='left')
        good = (mps > 0) * (mps < Mp)
        ms = ms[good]
        mps = mps[good]
        ehs = xps_shifted[mps] - xs[ms]
        bees = xs[ms] - xps_shifted[mps - 1]
        seas = ehs + bees
        return (M, Mp, v, xs, ms, mps, ehs, bees, seas)

    def Pdot(self, state, vec):
        # takes state and model flux vector, returns (shifted) model interpolated into data space
        # unpack state
        M, Mp, v, xs, ms, mps, ehs, bees, seas = state
        # do shit
        result = np.zeros(M)
        result[ms] = vec[mps - 1] * ehs / seas + vec[mps] * bees / seas
        return result

    def dotP(self, state, vec):
        # takes state and data flux vector, returns data interpolated into (shifted) model space
        # unpack state
        M, Mp, v, xs, ms, mps, ehs, bees, seas = state
        # do shit
        result = np.zeros(Mp)
        result[mps - 1] += vec[ms] * ehs / seas
        result[mps] += vec[ms] * bees / seas
        return result

    def dotdPdv(self, state, vec):
        # unpack state
        M, Mp, v, xs, ms, mps, ehs, bees, seas = state
        # do shit
        result = np.zeros(Mp)
        foos = vec[ms] / seas * self.dlndopplerdv(v) # * xs[ms] ??
        result[mps - 1] += foos
        result[mps] -= foos
        return result

    def dPdotdv(self, state, vec):
        # unpack state
        M, Mp, v, xs, ms, mps, ehs, bees, seas = state
        # do shit
        result = np.zeros(M)
        result[ms] = (vec[mps - 1] - vec[mps]) * self.dlndopplerdv(v) / seas
        return result

    def initialize_model(self, r, role, dx = np.log(6000.01) - np.log(6000.)):
        """
        `all_data`: `[N, M]` array of pixels
        `rvs`: `[N]` array of RVs
        """
        resids = 1. * self.data[r]            
        if role == 'star':
            print 'initializing star model...'
            rvs = self.rvs_star[r]
            if self.model_ys_t[r] is not None:
                for n in range(self.N):
                    pd, dpd_dv = self.calc_pds(r, n, self.rvs_t[r][n], 't')
                    resids[n] -= pd
        elif role == 't':
            rvs = self.rvs_t[r]
            if self.model_ys_star[r] is not None:
                for n in range(self.N):
                    pd, dpd_dv = self.calc_pds(r, n, self.rvs_star[r][n], 'star')
                    resids[n] -= pd        
        all_xs = np.empty_like(self.data[r])
        for i in range(self.N):
            all_xs[i,:] = self.data_xs[r][i,:] - np.log(self.doppler(rvs[i])) # shift to rest frame
        all_resids, all_xs = np.ravel(resids), np.ravel(all_xs)
        tiny = 10.
        template_xs = np.arange(min(all_xs)-tiny*dx, max(all_xs)+tiny*dx, dx)
        template_ys = np.nan + np.zeros_like(template_xs)
        for i,t in enumerate(template_xs):
            ind = (all_xs >= t-dx/2.) & (all_xs < t+dx/2.)
            if np.sum(ind) > 0:
                template_ys[i] = np.nanmedian(all_resids[ind])
        ind_nan = np.isnan(template_ys)
        template_ys.flat[ind_nan] = np.interp(template_xs[ind_nan], template_xs[~ind_nan], template_ys[~ind_nan])
        if role == 'star':
            self.model_xs_star[r], self.model_ys_star[r] = template_xs, template_ys
        elif role == 't':
            self.model_xs_t[r], self.model_ys_t[r] = template_xs, template_ys
            

    def rv_lnprior(self, rvs):
        return -0.5 * np.mean(rvs)**2/1.**2

    def drv_lnprior_dv(self, rvs):
        return np.zeros_like(rvs) - np.mean(rvs)/1.**2/len(rvs)
        
    def d2rv_lnprior_dv2(self, rvs):
        return np.zeros_like(rvs) - 1./1.**2/len(rvs)**2
    
    def calc_pds(self, r, n, rv, role):
        if role == 'star':
            state_star = self.state(rv, self.data_xs[r][n], self.model_xs_star[r])    
            state_t = self.state(self.rvs_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
            dpd_dv = self.dPdotdv(state_star, self.model_ys_star[r])
            
        elif role == 't':
            state_star = self.state(self.rvs_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
            state_t = self.state(rv, self.data_xs[r][n], self.model_xs_t[r])
            dpd_dv = self.airms[n] * self.dPdotdv(state_t, self.model_ys_t[r])
            
        pd_star = self.Pdot(state_star, self.model_ys_star[r])
        pd_t = self.airms[n] * self.Pdot(state_t, self.model_ys_t[r])
        pd = pd_star + pd_t
        return pd, dpd_dv
        
    def lnlike(self, r):
        # lnlike including every prior
        lnlike = 0.
        for n in range(self.N):
            pd, dpd_dv = self.calc_pds(r, n, self.rvs_star[r][n], 'star')
            lnlike += -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])

        lnpost = lnlike + self.rv_lnprior(self.rvs_star[r]) + self.rv_lnprior(self.rvs_t[r]) \
                    + self.model_ys_lnprior(self.model_ys_star[r]) + self.model_ys_lnprior(self.model_ys_t[r])
        return  lnpost
        
    def dlnlike_drv(self, rv, r, n, role):
        pd, dpd_dv = self.calc_pds(r, n, rv, role)
        lnlike = -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
        dlnlike_dv = np.asarray([np.sum((self.data[r][n,:] - pd) * self.ivars[r][n,:] * dpd_dv), ])
        if role == 'star':
            rvs = 1. * self.rvs_star[r]
            rvs[n] = rv
            lnpost = lnlike + self.rv_lnprior(rvs) + self.rv_lnprior(self.rvs_t[r])
            dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(rvs)[n] # HACK
        elif role == 't':
            rvs = 1. * self.rvs_t[r]
            rvs[n] = rv
            lnpost = lnlike + self.rv_lnprior(self.rvs_star[r]) + self.rv_lnprior(rvs)
            dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(rvs)[n] # HACK
        return lnpost, dlnpost_dv
        
    def opposite_dlnlike_drv(self, rv, r, n, role):
        # for scipy.optimize
        lnpost, dlnpost_dv = self.dlnlike_drv(rv, r, n, role)
        return -1.* lnpost, -1.* dlnpost_dv    
               
    def d2lnlike_dv2(self, r, role):
        # returns an N-vector corresponding to the second derivative of lnlike w.r.t v_n (of "star" or of "t")
        if role == 'star':
            rvs = self.rvs_star[r]
        elif role == 't':
            rvs = self.rvs_t[r]
        d2lnlikes = np.zeros_like(rvs)
        for n in range(self.N):
            _, dpd_dv = self.calc_pds(r, n, rvs[n], role)
            d2lnlikes[n] = np.dot(dpd_dv, self.ivars[r][n] * dpd_dv)
        d2lnposts = d2lnlikes # + self.d2rv_lnprior_dv2(rvs)
        return d2lnposts # TODO: do we want d2lnposts or d2lnlikes?

    def model_ys_lnprior(self, w):
        return -0.5 * np.sum(w**2)/100.**2

    def dmodel_ys_lnprior_dw(self, w):
        return -1.*w / 100.**2
        
    def dlnlike_dw(self, r, model, role):
        lnlike = 0.
        Mp = len(model)
        dlnlike_dw = np.zeros(Mp)
        for n in range(self.N):
            state_star = self.state(self.rvs_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
            state_t = self.state(self.rvs_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
            if role =='star':
                pd_star = self.Pdot(state_star, model)
                pd_t = self.airms[n] * self.Pdot(state_t, self.model_ys_t[r])
                state = state_star # for derivative
            elif role == 't':
                pd_star = self.Pdot(state_star, self.model_ys_star[r])
                pd_t = self.airms[n] * self.Pdot(state_t, model)
                state = state_t # for derivative
            pd = pd_star + pd_t
            dp_star = self.dotP(state, (self.data[r][n,:] - pd) * self.ivars[r][n,:])
            lnlike += -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
            dlnlike_dw += dp_star
        lnprior = self.model_ys_lnprior(model)
        dlnprior_dw = self.dmodel_ys_lnprior_dw(model)
        return lnlike + lnprior, dlnlike_dw + dlnprior_dw

    def improve_model(self, r, role, step_scale=1., maxniter=64, tol=0.001):
        """
        This function could be written to be much more computationally efficient.
        """
        if role == 'star':
            w = np.copy(self.model_ys_star[r])
        elif role == 't':
            w = np.copy(self.model_ys_t[r])
        quitc = np.Inf
        i = 0
        while ((quitc > tol) and (i < maxniter)):
            i += 1
            lnlike_o, dlnlike_dw = self.dlnlike_dw(r, w, role)
            stepped = False
            j = 0
            while((not stepped) and (j < 64)): # could go forever in theory
                j += 1
                step = step_scale * dlnlike_dw
                lnlike = self.dlnlike_dw(r, w + step, role)[0]
                dlnlike = lnlike - lnlike_o
                if dlnlike >= 0.0:
                    w += step
                    stepped = True   
                    step_scale *= 2.
                    quitc = lnlike - lnlike_o
                else:
                    step_scale *= 0.5
                    #print "improve_model: reducing step size to", step_scale
        return w
       
    def optimize(self, restart = False, **kwargs):
        """
        Loops over all orders in the e2ds case and optimizes them all as separate spectra.
        Takes the same kwargs as optimize_order.
        """
        if (hasattr(self, 'model_xs_star') == False) or (restart == True):
            self.rvs_star = [np.zeros(self.N) for r in range(self.R)]
            self.rvs_t = [np.zeros(self.N) for r in range(self.R)]
            self.model_ys_star = [None for r in range(self.R)] # not the right shape but whatevs, it's overwritten in the initialization of optimize_order
            self.model_xs_star = [None for r in range(self.R)]
            self.model_ys_t = [None for r in range(self.R)]
            self.model_xs_t = [None for r in range(self.R)]
        
            self.ivars_star = [np.zeros(self.N) for r in range(self.R)]
            self.ivars_t = [np.zeros(self.N) for r in range(self.R)]
        
        for r in range(self.R):
            self.optimize_order(r, restart=restart, **kwargs)
            if (r % 5) == 0:
                self.save_results('state_order{0}.hdf5'.format(r))
                
        self.rvs_star = np.asarray(self.rvs_star)
        self.rvs_t = np.asarray(self.rvs_t)
        self.ivars_star = np.asarray(self.ivars_star)
        self.ivars_t = np.asarray(self.ivars_t)
              
        
    def optimize_order(self, r, niter=5, restart = False, plot=False):
        """
        Optimize the velocities of the telluric spectrum and star as observed from the Earth, as well as
        the data-driven model for the star and telluric features.
        
        Args: 
            niter: The number of iterations to perform on updating the velocities and model spectra. 
                (default: ``5``)
            restart: If an optimization has already been performed, this flag will reject it and start from
                the initial system defaults, rather than continuing to optimize from the previous best fit
                (default: ``False``)
            plot: Display diagnostic plots after each optimization iteration (default: ``False``)
        """
        
        if (self.model_xs_star[r] == 0).all() or (restart == True):
            self.rvs_star[r] = -np.copy(self.drs_rvs)
            self.rvs_star[r] -= np.mean(self.rvs_star[r])
            self.rvs_t[r] = np.zeros(self.N)
            self.initialize_model(r, 'star')
            self.initialize_model(r, 't')
        
        previous_lnlike = self.lnlike(r)
        for iteration in range(niter):
            print "Fitting stellar RVs..."
            for n in range(self.N):
                soln = minimize(self.opposite_dlnlike_drv, self.rvs_star[r][n], args=(r, n, 'star'),
                             method='BFGS', jac=True, options={'disp':False})
                self.rvs_star[r][n] = soln['x']
            new_lnlike = self.lnlike(r)
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                print "self.lnlike = ", self.lnlike(r)
                lnprior_w_star = self.model_ys_lnprior(self.model_ys_star[r])
                lnprior_w_t = self.model_ys_lnprior(self.model_ys_t[r])
                lnprior_rv = self.rv_lnprior(self.rvs_star[r]) + self.rv_lnprior(self.rvs_t[r])
                foo = 0.
                for n in range(self.N):
                    foo += self.opposite_dlnlike_drv(self.rvs_star[r][n], r, n, 'star')[0] + lnprior_rv
                print "self.dlnlike_drv = ", -1. * foo + lnprior_rv + lnprior_w_star + lnprior_w_t
                print "self.dlnlike_dw (star) = ", self.dlnlike_dw(r, self.model_ys_star[r], 'star')[0] + lnprior_rv + lnprior_w_t
                print "self.dlnlike_dw (t) = ", self.dlnlike_dw(r, self.model_ys_t[r], 't')[0] + lnprior_rv + lnprior_w_star
                assert False
            previous_lnlike = new_lnlike 

                
            print "Improving stellar template spectra..."
            self.model_ys_star[r] = self.improve_model(r, 'star')
            new_lnlike = self.lnlike(r)
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                print "self.lnlike = ", self.lnlike(r)
                lnprior_w_star = self.model_ys_lnprior(self.model_ys_star[r])
                lnprior_w_t = self.model_ys_lnprior(self.model_ys_t[r])
                lnprior_rv = self.rv_lnprior(self.rvs_star[r]) + self.rv_lnprior(self.rvs_t[r])
                foo = 0.
                for n in range(self.N):
                    foo += self.opposite_dlnlike_drv(self.rvs_star[r][n], r, n, 'star')[0] + lnprior_rv
                print "self.dlnlike_drv = ", -1. * foo + lnprior_rv + lnprior_w_star + lnprior_w_t
                print "self.dlnlike_dw (star) = ", self.dlnlike_dw(r, self.model_ys_star[r], 'star')[0] + lnprior_rv + lnprior_w_t
                print "self.dlnlike_dw (t) = ", self.dlnlike_dw(r, self.model_ys_t[r], 't')[0] + lnprior_rv + lnprior_w_star
                assert False
            previous_lnlike = new_lnlike 
                
            print "Fitting telluric RVs..."
            for n in range(self.N):
                self.rvs_t[r][n] = minimize(self.opposite_dlnlike_drv, self.rvs_t[r][n], args=(r, n, 't'),
                             method='BFGS', jac=True, options={'disp':False})['x']
            new_lnlike = self.lnlike(r)
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                print self.lnlike(self.rvs_t[r], r, 't')
                assert False
            previous_lnlike = new_lnlike 
                
            print "Improving telluric template spectra..."
            self.model_ys_t[r] = self.improve_model(r, 't')
            new_lnlike = self.lnlike(r)
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                print "self.lnlike = ", self.lnlike(r)
                lnprior_w_star = self.model_ys_lnprior(self.model_ys_star[r])
                lnprior_w_t = self.model_ys_lnprior(self.model_ys_t[r])
                lnprior_rv = self.rv_lnprior(self.rvs_star[r]) + self.rv_lnprior(self.rvs_t[r])
                foo = 0.
                for n in range(self.N):
                    foo += self.opposite_dlnlike_drv(self.rvs_star[r][n], r, n, 'star')[0] + lnprior_rv
                print "self.dlnlike_drv = ", -1. * foo + lnprior_rv + lnprior_w_star + lnprior_w_t
                print "self.dlnlike_dw (star) = ", self.dlnlike_dw(r, self.model_ys_star[r], 'star')[0] + lnprior_rv + lnprior_w_t
                print "self.dlnlike_dw (t) = ", self.dlnlike_dw(r, self.model_ys_t[r], 't')[0] + lnprior_rv + lnprior_w_star
                assert False
            previous_lnlike = new_lnlike 
                
    
            print "order {0}, iter {1}: star std = {2:.2f}, telluric std = {3:.2f}".format(r, iteration, np.std(self.rvs_star[r] + self.bervs), np.std(self.rvs_t[r]))
            print "                       RMS of resids w.r.t. HARPS DRS RVs = {0:.2f}".format(np.std(self.rvs_star[r] + self.drs_rvs))
            if plot:
                plt.figure()
                plt.plot(np.arange(self.N), self.rvs_star[r] + self.bervs - np.mean(self.rvs_star[r] + self.bervs), color='k')
                plt.plot(np.arange(self.N), self.rvs_t[r] - np.mean(self.rvs_t[r]), color='red')
                plt.show()   

            if plot:
                self.plot_models(r,0, filename='order{0}_iter{1}.png'.format(r, iteration))
                
        self.ivars_star[r] = self.d2lnlike_dv2(r, 'star')   
        self.ivars_t[r] = self.d2lnlike_dv2(r, 't')
        
    def plot_models(self, r, n, filepath='../results/plots/', filename=None):
        #calculate models
        state_star = self.state(self.rvs_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
        model_star = self.Pdot(state_star, self.model_ys_star[r])
        state_t = self.state(self.rvs_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
        model_t = self.airms[n] * self.Pdot(state_t, self.model_ys_t[r])
        #plot
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(np.exp(self.data_xs[r][n]), np.exp(self.data[r][n]), 
                    color='blue', alpha=0.5, label='data')
        ax.plot(np.exp(self.data_xs[r][n]), np.exp(model_star), 
                    color='k', alpha=0.5, label='star model')
        ax.plot(np.exp(self.data_xs[r][n]), np.exp(model_t), 
                    color='red', alpha=0.5, label='telluric model')
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel('Normalized Flux')
        ax.set_ylim([0.0, 1.3])
        ax.legend(loc='lower right')
        ax.set_title('Order #{0}, Epoch #{1}'.format(r,n))
        if filename is None:
            filename = 'model_order{0}_epoch{1}.png'.format(r,n)
        plt.savefig(filepath+filename)
        plt.close(fig)
        
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
            
    def load_results(self, filename):
        with h5py.File(filename) as f:
            self.rvs_star = np.copy(f['rvs_star'])
            self.rvs_t = np.copy(f['rvs_t'])
            self.model_xs_star = np.copy(f['model_xs_star']).tolist()
            self.model_ys_star = np.copy(f['model_ys_star']).tolist()
            self.model_xs_t = np.copy(f['model_xs_t']).tolist()
            self.model_ys_t = np.copy(f['model_ys_t']).tolist()
            try:
                self.ivars_star = np.copy(f['ivars_star'])
                self.ivars_t = np.copy(f['ivars_t'])
            except:  # temporary fix for old results files
                self.ivars_star = np.copy(f['ivars_rv_star'])
                self.ivars_t = np.copy(f['ivars_rv_t'])                
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
        print "optimize_sigmas: optimizing..."
        soln_sigmas = minimize(self.opposite_lnlike_sigmas, x0_sigmas, args=(restart), method='BFGS', options={'disp':True})['x'] # HACK
        # save results
        lnlike, rvs_N, rvs_R = self.lnlike_sigmas(soln_sigmas, return_rvs=True)
        self.order_rvs = rvs_R
        self.time_rvs = rvs_N
        self.order_sigmas = soln_sigmas
        
def pack_keplerian_pars(P, K, e, omega, M0, offset):
    return [P, K, e, omega, M0, offset]
    
def unpack_keplerian_pars(pars):
    P, K, e, omega, M0, offset = pars
    return P, K, e, omega, M0, offset

def opposite_lnlike_keplerian(pars, times, rvs, sigs):
    P, K, e, omega, M0, offset = unpack_keplerian_pars(pars)
    t0 = np.min(times)
    ys = cy_rv_from_elements(times, P, K, e, omega, M0, t0, 1e-10, 128)
    if np.any(np.isnan(ys)):
        return np.inf
    ys += offset
    lnlike = -0.5 * np.sum((rvs - ys)**2 / sigs**2)
    return -1.0 * lnlike

def fit_keplerian(pars0, times, rvs, sigs):
    bounds = [(None, None) for par in pars0]
    bounds[2] = (0.0, 1.0)
    pars = minimize(opposite_lnlike_keplerian, pars0, args=(times.astype('<f8'), rvs, sigs), method='L-BFGS-B',
             bounds=bounds, options={'disp':True})['x']
    return pars
    
            
if __name__ == "__main__":
    # temporary code to diagnose issues in results
    starid = 'hip30037'
    a = star(starid+'_e2ds.hdf5', orders=np.arange(72), N=25)    
    
    if False: # optimize
        a.optimize(niter=20, plot=False)
        a.save_results('../results/'+starid+'_results.hdf5')
        
    a.load_results('../results/'+starid+'_results.hdf5')
    
    if False: # make model plots
        for r in range(a.R):
            a.plot_models(r, 0)
        
    N_epochs = 25
    a.rvs_star = np.asarray(a.rvs_star)[:,:N_epochs] # just in case
    a.ivars_star = np.asarray(a.ivars_star)[:,:N_epochs]
    a.bervs = a.bervs[:N_epochs]
    a.N = N_epochs
    
    a.optimize_sigmas()
    print "RV std = {0:.2f} m/s".format(np.std(a.time_rvs + a.bervs))
    print "HARPS pipeline std = {0:.2f} m/s".format(np.std(a.drs_rvs - a.bervs))
    print "std w.r.t HARPS values = {0:.2f} m/s".format(np.std(a.time_rvs + a.drs_rvs))
    
    sigs = np.ones_like(a.time_rvs) # HACK
    P, K, e, omega, M0, offset = 31.6, 4243.8, 0.3, 226. * np.pi / 180., 63. * np.pi / 180., 0.0 # days, m/s, dimensionless, radians, JD
    t0 = np.min(a.dates)
    pars0 = pack_keplerian_pars(P, K, e, omega, M0, offset)
    soln = fit_keplerian(pars0, a.dates, a.time_rvs, sigs)
    P, K, e, omega, M0, offset = unpack_keplerian_pars(soln)
    
    orbit = KeplerOrbit(P=P*u.day, e=e, omega=omega*u.rad, 
                        M0=M0*u.rad, Omega=0*u.deg, i=90*u.deg,
                        t0=t0_time)
    plt.plot(t_grid.jd % P, K*orbit.unscaled_radial_velocity(t_grid) + offset)
    plt.errorbar(a.dates % P, rvs, sigs, marker='o', linestyle='none')
    plt.show()
    
    if False:
        order_stds = np.std(a.rvs_star + np.tile(a.bervs, (a.R,1)), axis=1)
    
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.semilogy(np.arange(a.R), order_stds, marker='o', ls=' ')
        ax.set_ylabel(r'empirical std per order (m s$^{-1}$)')
        ax.set_xlabel('Order #')    
        plt.savefig('../results/plots/order_stds.png')
        plt.close(fig)
    
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.errorbar(np.arange(a.R), a.order_rvs, a.order_sigmas, fmt='o')
        ax.set_xlabel('Order #')
        ax.set_ylabel(r'RV (m s$^{-1}$)')
        plt.savefig('../results/plots/order_rvs.png')
        plt.close(fig)
    
        fig = plt.figure()
        gs = gridspec.GridSpec(2,1,height_ratios=[4,1],hspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.ticklabel_format(useOffset=False)
        plt.setp(ax1.get_xticklabels(), visible=False)    
        ax1.plot(a.dates[:a.N] - 2450000., a.time_rvs + a.bervs - np.mean(a.time_rvs + a.bervs), 'o', color='k', alpha=0.8, label='wobble')
        ax1.plot(a.dates[:a.N] - 2450000., -1. * a.drs_rvs + a.bervs - np.mean(-1. * a.drs_rvs + a.bervs), 'o', color='r', alpha=0.8, label='HARPS DRS')
        ax1.legend(loc='upper left',prop={'size':24})
        ax1.set_ylabel(r'RV (m s$^{-1}$)')
        diff = a.time_rvs + a.drs_rvs - np.mean(a.drs_rvs + a.time_rvs)
        ax2.plot(a.dates[:a.N] - 2450000., diff, 'o', color='k', alpha=0.8)
        ax2.set_ylabel('Diff')
        ax2.set_xlabel('J.D. - 2450000')
        plt.savefig('../results/plots/time_rvs.png')
        plt.close(fig)
    
        fig = plt.figure()
        ax = plt.subplot(111)
        rv_predictions = np.tile(a.order_rvs[:,None], (1,a.N)) + np.tile(a.time_rvs, (a.R,1))
        resids = a.rvs_star - rv_predictions
        all_vars = 1./a.ivars_star**2 + np.tile((a.order_sigmas[:,None])**2, (1,a.N))
        chis = resids / np.median(np.sqrt(all_vars))
        pos = resids >= 0.
        neg = resids < 0.
        sizes = np.abs(chis) * 4.
        ax.scatter(np.tile(np.arange(a.R)[:,None], (1,a.N))[pos], np.tile(np.arange(a.N), (a.R,1))[pos], marker='o', c='k', s=sizes[pos])
        ax.scatter(np.tile(np.arange(a.R)[:,None], (1,a.N))[neg], np.tile(np.arange(a.N), (a.R,1))[neg], marker='o', c='r', s=sizes[neg])
        ax.set_xlabel('Order #')
        ax.set_ylabel('Epoch #')
        plt.savefig('../results/plots/outlier_check.png')
        plt.close(fig)
    
    if False:
        # tests on separating RVs by time and by order
        cmap = matplotlib.cm.get_cmap(name='jet')
        colors_by_epoch = [cmap(1.*i/a.N) for i in range(a.N)]
        colors_by_order = [cmap(1.*i/a.R) for i in range(a.R)]
    
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.loglog([2.,1.e3],[2.,1.e3], c='red')
        ax.loglog(order_stds, a.order_sigmas, marker='o', ls=' ')
        ax.set_ylabel(r'best-fit $\sigma_{order}$ (m s$^{-1}$)')
        ax.set_xlabel(r'empirical std per order (m s$^{-1}$)')    
        plt.savefig('../results/plots/order_stds_against_bestfit.png')
        plt.close(fig)
    
        orders_in_order = np.argsort(order_stds) # order inds from lowest to highest emp. variance
        order_counts = np.arange(2,72)
        time_rvs = np.zeros((len(order_counts), a.N))
        rv_stds = np.zeros(len(order_counts))
        for i,j in enumerate(order_counts):
            print "fitting orders", orders_in_order[:j]
            b = copy.deepcopy(a)
            b.R = j
            b.rvs_star = b.rvs_star[orders_in_order[:j]] # take only the j best orders
            b.ivars_star = b.ivars_star[orders_in_order[:j]]
            b.optimize_sigmas(restart=True)
            time_rvs[i,:] = b.time_rvs + b.bervs
            rv_stds[i] = np.std(b.time_rvs + b.bervs)
            print "RV std = {0:.2f} m/s".format(np.std(b.time_rvs + b.bervs))
        
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.axhline(np.std(a.drs_rvs[:a.N] - a.bervs), color='k', alpha=0.5)
        ax.scatter(order_counts, rv_stds, marker='o')
        ax.set_xlabel('Number of orders used')
        ax.set_ylabel('stdev of time-series RVs')
        plt.savefig('../results/plots/stdev_by_orders_used.png')
        plt.close(fig)
    
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(len(order_counts)):
            ax.plot(np.arange(a.N), time_rvs[i], color=colors_by_order[i], alpha=0.5)
        ax.set_xlabel('Epoch #')
        ax.set_ylabel(r'barycentric RV (m s$^{-1}$)')
        plt.savefig('../results/plots/rvs_by_orders_used.png')
        plt.close(fig)
    
        
    if False:
        # tests on time-series RVs
        fig = plt.figure()
        ax = plt.subplot(111) 
        for n in range(a.N): 
            ax.errorbar(np.arange(a.R), a.rvs_star[:,n] + a.bervs[n], 1./np.sqrt(a.ivars_star[:,n]), 
                color=colors_by_epoch[n], alpha=0.5, fmt='o')
        ax.set_ylabel(r'barycentric RV (m s$^{-1}$)')
        ax.set_xlabel('Order #')
        plt.savefig('../results/plots/rv_per_order.png')
        plt.close(fig) 
    
        fig = plt.figure()
        ax = plt.subplot(111) 
        for r in range(a.R): 
            ax.errorbar(np.arange(a.N), a.rvs_star[r,:] + a.bervs, 1./np.sqrt(a.ivars_star[r,:]), 
                color=colors_by_order[r], alpha=0.5, fmt='o')
        ax.set_ylabel(r'barycentric RV (m s$^{-1}$)')
        ax.set_xlabel('Epoch #')
        plt.savefig('../results/plots/rv_per_epoch.png')
        plt.close(fig) 
        
        rv_predictions = np.tile(a.order_rvs[:,None], (1,a.N)) + np.tile(a.time_rvs, (a.R,1))
        resids = a.rvs_star - rv_predictions
    
        fig = plt.figure()
        ax = plt.subplot(111) 
        for n in range(a.N): 
            ax.errorbar(np.arange(a.R), resids[:,n], 1./np.sqrt(a.ivars_star[:,n]), 
                color=colors_by_epoch[n], alpha=0.5, fmt='o')
        ax.set_ylabel(r'RV - predicted (m s$^{-1}$)')
        ax.set_xlabel('Order #')
        plt.savefig('../results/plots/resids_per_order.png')
        plt.close(fig) 
    
        fig = plt.figure()
        ax = plt.subplot(111) 
        for r in range(a.R): 
            ax.errorbar(np.arange(a.N), resids[r,:], 1./np.sqrt(a.ivars_star[r,:]), 
                color=colors_by_order[r], alpha=0.5, fmt='o')
        ax.set_ylabel(r'RV - predicted (m s$^{-1}$)')
        ax.set_xlabel('Epoch #')
        plt.savefig('../results/plots/resids_per_epoch.png')
        plt.close(fig)
    
    
        
