import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_cg, minimize
import h5py

c = 2.99792458e8   # m/s

class star(object):
    '''
    Example use case:
    import wobble
    a = wobble.star('hip30037.hdf5')
    a.optimize(niter=10)
    '''
    
    def __init__(self, filename, filepath='../data/', wl_lower = 5900, wl_upper = 6000, N=16 *args):
        filename = filepath + filename
        self.N = N
        self.wavelength_lower = wl_lower
        self.wavelength_upper = wl_upper
        
        with h5py.File(filename) as f:
            inds = (f['xs'][:] > self.wavelength_lower) & (f['xs'][:] < self.wavelength_upper)
            self.data = np.copy(f['data'])[:self.N,inds]
            self.data_xs = np.log(np.copy(f['xs'][inds]))
            self.ivars = np.copy(f['ivars'])[:self.N,inds]
            self.true_rvs = np.copy(f['true_rvs'])[:self.N]
            self.bervs = np.copy(f['berv'])[:self.N] * -1.e3
            
            for i in xrange(len(self.data)):
                self.data[i] /= np.median(self.data[i])
    
            self.data = np.log(self.data)
        
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

    def make_template(self, rvs, dx = np.log(6000.01) - np.log(6000.)):
        """
        `all_data`: `[N, M]` array of pixels
        `rvs`: `[N]` array of RVs
        `xs`: `[M]` array of wavelength values
        `dx`: linear spacing desired for template wavelength grid (A)
        """

        (N,M) = np.shape(self.data)
        all_xs = np.empty_like(self.data)
        for i in range(N):
            all_xs[i,:] = self.data_xs - np.log(self.doppler(rvs[i])) # shift to rest frame
        all_data, all_xs = np.ravel(self.data), np.ravel(all_xs)
        tiny = 10.
        template_xs = np.arange(min(all_xs)-tiny*dx, max(all_xs)+tiny*dx, dx)
        template_ys = np.nan + np.zeros_like(template_xs)
        for i,t in enumerate(template_xs):
            ind = (all_xs >= t-dx/2.) & (all_xs < t+dx/2.)
            if np.sum(ind) > 0:
                template_ys[i] = np.nanmedian(all_data[ind])
        ind_nan = np.isnan(template_ys)
        template_ys.flat[ind_nan] = np.interp(template_xs[ind_nan], template_xs[~ind_nan], template_ys[~ind_nan]) #np.interp(template_xs[ind_nan], template_xs[~ind_nan], template_ys[~ind_nan])
        return template_xs, template_ys

    def rv_lnprior(self, rvs):
        return -0.5 * np.mean(rvs)**2/1.**2

    def drv_lnprior_dv(self, rvs):
        return np.zeros_like(rvs) - np.mean(rvs)/1.**2/len(rvs)

    def lnlike_star(self, x0_star):
        try:
            N = len(x0_star)
        except:
            N = 1  
        lnlike = 0.
        dlnlike_dv = np.zeros(N)
        for n in range(N):
            state_star = self.state(x0_star[n], self.data_xs, self.model_xs_star)
            pd_star = self.Pdot(state_star, self.model_ys_star)
            state_t = self.state(self.x0_t[n], self.data_xs, self.model_xs_t)
            pd_t = self.Pdot(state_t, self.model_ys_t)
            pd = pd_star + pd_t
            lnlike += -0.5 * np.sum((self.data[n,:] - pd)**2 * self.ivars[n,:])
            dpd_dv = self.dPdotdv(state_star, self.model_ys_star)
            dlnlike_dv[n] = np.sum((self.data[n,:] - pd) * self.ivars[n,:] * dpd_dv)
        lnpost = lnlike + self.rv_lnprior(x0_star)
        dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0_star)
        return -1 * lnpost, -1 * dlnpost_dv

    def lnlike_t(self, x0_t):
        try:
            N = len(x0_t)
        except:
            N = 1  
        lnlike = 0.
        dlnlike_dv = np.zeros(N)
        for n in range(N):
            state_star = self.state(self.x0_star[n], self.data_xs, self.model_xs_star)
            pd_star = self.Pdot(state_star, self.model_ys_star)
            state_t = self.state(x0_t[n], self.data_xs, self.model_xs_t)
            pd_t = self.Pdot(state_t, self.model_ys_t)
            pd = pd_star + pd_t
            lnlike += -0.5 * np.sum((self.data[n,:] - pd)**2 * self.ivars[n,:])
            dpd_dv = self.dPdotdv(state_t, self.model_ys_t)
            dlnlike_dv[n] = np.sum((self.data[n,:] - pd) * self.ivars[n,:] * dpd_dv)
        lnpost = lnlike + self.rv_lnprior(x0_t)
        dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0_t)
        return -1 * lnpost, -1 * dlnpost_dv

    def model_ys_lnprior(self, w):
        return -0.5 * np.sum(w**2)/100.**2


    def dmodel_ys_lnprior_dw(self, w):
        return -1.*w / 100.**2


    def dlnlike_star_dw_star(self, model_ys_star):
        try:
            N = len(self.x0_star)
        except:
            N = 1  
        lnlike = 0.
        Mp = len(self.model_xs_star)
        dlnlike_dw = np.zeros(Mp)
        for n in range(N):
            state_star = self.state(self.x0_star[n], self.data_xs, self.model_xs_star)
            pd_star = self.Pdot(state_star, model_ys_star)
            state_t = self.state(self.x0_t[n], self.data_xs, self.model_xs_t)
            pd_t = self.Pdot(state_t, self.model_ys_t)
            pd = pd_star + pd_t
            dp_star = self.dotP(state_star, (self.data[n,:] - pd)*self.ivars[n,:]) 
            lnlike += -0.5 * np.sum((self.data[n,:] - pd)**2 * self.ivars[n,:])
            dlnlike_dw += dp_star
        lnprior = self.model_ys_lnprior(model_ys_star)
        dlnprior = self.dmodel_ys_lnprior_dw(model_ys_star)
        return -lnlike - lnprior, -dlnlike_dw - dlnprior

    def dlnlike_t_dw_t(self, model_ys_t):
        try:
            N = len(self.x0_t)
        except:
            N = 1  
        lnlike = 0.
        Mp = len(self.model_xs_t)
        dlnlike_dw = np.zeros(Mp)
        for n in range(N):
            state_star = self.state(self.x0_star[n], self.data_xs, self.model_xs_star)
            pd_star = self.Pdot(state_star, self.model_ys_star)
            state_t = self.state(self.x0_t[n], self.data_xs, self.model_xs_t)
            pd_t = self.Pdot(state_t, model_ys_t)
            pd = pd_star + pd_t
            dp_t = self.dotP(state_t, (self.data[n,:] - pd)*self.ivars[n,:]) 
            lnlike += -0.5 * np.sum((self.data[n,:] - pd)**2 * self.ivars[n,:])
            dlnlike_dw += dp_t
        lnprior = self.model_ys_lnprior(model_ys_t)
        dlnprior = self.dmodel_ys_lnprior_dw(model_ys_t)
        return -lnlike - lnprior, -dlnlike_dw - dlnprior


    def improve_telluric_model(self, step_scale=5e-7):
        w = np.copy(self.model_ys_t)
        for i in range(50):
            lnlike, dlnlike_dw = self.dlnlike_t_dw_t(w)
            w -= step_scale * dlnlike_dw    
        return w

    def improve_star_model(self, step_scale=5e-7):
        w = np.copy(self.model_ys_star)
        for i in range(50):
            lnlike, dlnlike_dw = self.dlnlike_star_dw_star(w)
            w -= step_scale * dlnlike_dw 
        return w
        
    def optimize(self, niter=5, restart = False, plot=False):
        
        if (hasattr(self, 'model_xs_star') == False) or (restart == True):

            self.x0_star = -np.copy(self.true_rvs)
            self.x0_star -= np.mean(self.x0_star)
            self.x0_t = np.zeros(self.N)
            self.model_xs_star, self.model_ys_star = self.make_template(self.x0_star)
            self.model_xs_t, self.model_ys_t = self.make_template(self.x0_t)
        
        for iteration in range(niter):
            print "Fitting stellar RVs..."
            self.soln_star =  minimize(self.lnlike_star, self.x0_star,
                             method='BFGS', jac=True, options={'disp':True, 'gtol':1.e-2, 'eps':1.5e-5})['x']

            self.model_ys_t = self.improve_telluric_model()
            self.model_ys_star = self.improve_star_model()
            print "Star model improved. Fitting telluric RVs..."
            self.soln_t =  minimize(self.lnlike_t, self.x0_t, 
                             method='BFGS', jac=True, options={'disp':True, 'gtol':1.e-2, 'eps':1.5e-5})['x']

            self.model_ys_t = self.improve_telluric_model()
            self.model_ys_star = self.improve_star_model()

            self.x0_star = self.soln_star
            self.x0_t = self.soln_t

            print "iter {0}: star std = {1:.2f}, telluric std = {2:.2f}".format(iteration, np.std(self.soln_star + self.true_rvs), np.std(self.soln_t))
            if plot == True:
                plt.plot(np.arange(self.N), self.soln_star + self.true_rvs - np.mean(self.soln_star + self.true_rvs), color='k')
                plt.plot(np.arange(self.N), self.soln_t - np.mean(self.soln_t), color='red')
                plt.show()
            
    def show_results(self):
        plt.scatter(np.arange(self.N), self.soln_star+self.true_rvs)
        plt.show()
        
        plt.plot(np.arange(self.N), self.soln_star + self.true_rvs - np.mean(self.soln_star + self.true_rvs), color='k')
        plt.plot(np.arange(self.N), self.soln_t - np.mean(self.soln_t), color='red')
        plt.show()
        
        plt.plot(np.arange(self.N), self.soln_star - np.mean(self.soln_star) + self.bervs, 'ko')
        plt.plot(np.arange(self.N), -self.true_rvs + np.mean(self.true_rvs) + self.bervs, 'r.')
        plt.show()
