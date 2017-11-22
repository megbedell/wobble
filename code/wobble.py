import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_cg, minimize
import h5py
from utils import fit_continuum

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
    a = wobble.star('hip54287_e2ds.hdf5', orders=np.arange(0,72), N=40)
    a.optimize(niter=30)

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
                    N = 16, e2ds = True, orders = [30], min_flux = 1.):
        filename = filepath + filename
        self.N = N
        self.R = len(orders) # number of orders to be analyzed
        if e2ds:
            self.orders = orders
            with h5py.File(filename) as f:
                self.data = [f['data'][i][:self.N,:] for i in orders]
                self.data_xs = [np.log(f['xs'][i][:self.N,:]) for i in orders]
                self.ivars = [f['ivars'][i][:self.N,:] for i in orders]
                self.true_rvs = np.copy(f['true_rvs'])[:self.N]
                self.bervs = np.copy(f['berv'])[:self.N] * -1.e3
                self.drifts = np.copy(f['drift'])[:self.N]
                self.airms = np.copy(f['airm'])[:self.N]
        else:
            self.wavelength_lower = wl_lower
            self.wavelength_upper = wl_upper
        
            with h5py.File(filename) as f:
                inds = (f['xs'][:] > self.wavelength_lower) & (f['xs'][:] < self.wavelength_upper)
                N_all, M_all = np.shape(inds)
                data = np.copy(f['data'])[inds]
                self.data = [np.reshape(data, (N_all, -1))[:self.N,:]]
                data_xs = np.log(np.copy(f['xs'][inds]))
                self.data_xs = [np.reshape(data_xs, (N_all, -1))[:self.N,:]]
                ivars = np.copy(f['ivars'])[inds]
                self.ivars = [np.reshape(ivars, (N_all, -1))[:self.N,:]]
                self.true_rvs = np.copy(f['true_rvs'])[:self.N]
                self.bervs = np.copy(f['berv'])[:self.N] * -1.e3
                self.drifts = np.copy(f['drift'])[:self.N]
                self.airms = np.copy(f['airm'])[:self.N]

        for r in range(self.R):
            bad = np.where(self.data[r] < min_flux)
            self.data[r][bad] = min_flux
            self.ivars[r][bad] = 0.
        self.data = np.log(self.data) 
        
        self.continuum_normalize() 
        
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

    def make_template(self, r, rvs, dx = np.log(6000.01) - np.log(6000.)):
        """
        `all_data`: `[N, M]` array of pixels
        `rvs`: `[N]` array of RVs
        `xs`: `[N, M]` array of wavelength values
        `dx`: linear spacing desired for template wavelength grid (A)
        """

        (N,M) = np.shape(self.data[r])
        all_xs = np.empty_like(self.data[r])
        for i in range(N):
            all_xs[i,:] = self.data_xs[r][i,:] - np.log(self.doppler(rvs[i])) # shift to rest frame
        all_data, all_xs = np.ravel(self.data[r]), np.ravel(all_xs)
        tiny = 10.
        template_xs = np.arange(min(all_xs)-tiny*dx, max(all_xs)+tiny*dx, dx)
        template_ys = np.nan + np.zeros_like(template_xs)
        for i,t in enumerate(template_xs):
            ind = (all_xs >= t-dx/2.) & (all_xs < t+dx/2.)
            if np.sum(ind) > 0:
                template_ys[i] = np.nanmedian(all_data[ind])
        ind_nan = np.isnan(template_ys)
        template_ys.flat[ind_nan] = np.interp(template_xs[ind_nan], template_xs[~ind_nan], template_ys[~ind_nan])
        return template_xs, template_ys

    def rv_lnprior(self, rvs):
        return -0.5 * np.mean(rvs)**2/1.**2

    def drv_lnprior_dv(self, rvs):
        return np.zeros_like(rvs) - np.mean(rvs)/1.**2/len(rvs)

    def lnlike_star(self, x0_star, r):
        try:
            N = len(x0_star)
        except:
            N = 1  
        lnlike = 0.
        dlnlike_dv = np.zeros(N)
        for n in range(N):
            state_star = self.state(x0_star[n], self.data_xs[r][n], self.model_xs_star[r])
            pd_star = self.Pdot(state_star, self.model_ys_star[r])
            state_t = self.state(self.x0_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
            pd_t = self.airms[n] * self.Pdot(state_t, self.model_ys_t[r])
            pd = pd_star + pd_t
            lnlike += -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
            dpd_dv = self.dPdotdv(state_star, self.model_ys_star[r])
            dlnlike_dv[n] = np.sum((self.data[r][n,:] - pd) * self.ivars[r][n,:] * dpd_dv)
        lnpost = lnlike + self.rv_lnprior(x0_star) + self.rv_lnprior(self.x0_t[r])
        dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0_star)
        return lnpost, dlnpost_dv

    def lnlike_t(self, x0_t, r):
        try:
            N = len(x0_t)
        except:
            N = 1  
        lnlike = 0.
        dlnlike_dv = np.zeros(N)
        for n in range(N):
            state_star = self.state(self.x0_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
            pd_star = self.Pdot(state_star, self.model_ys_star[r])
            state_t = self.state(x0_t[n], self.data_xs[r][n], self.model_xs_t[r])
            pd_t = self.airms[n] * self.Pdot(state_t, self.model_ys_t[r])
            pd = pd_star + pd_t
            lnlike += -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
            dpd_dv = self.airms[n] * self.dPdotdv(state_t, self.model_ys_t[r])
            dlnlike_dv[n] = np.sum((self.data[r][n,:] - pd) * self.ivars[r][n,:] * dpd_dv)
        lnpost = lnlike  + self.rv_lnprior(self.x0_star[r]) + self.rv_lnprior(x0_t)
        dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0_t)
        return lnpost, dlnpost_dv

    def model_ys_lnprior(self, w):
        return -0.5 * np.sum(w**2)/100.**2


    def dmodel_ys_lnprior_dw(self, w):
        return -1.*w / 100.**2


    def dlnlike_star_dw_star(self, r, model_ys_star):
        try:
            N = len(self.x0_star[r])
        except:
            N = 1  
        lnlike = 0.
        Mp = len(self.model_xs_star[r])
        dlnlike_dw = np.zeros(Mp)
        for n in range(N):
            state_star = self.state(self.x0_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
            pd_star = self.Pdot(state_star, model_ys_star)
            state_t = self.state(self.x0_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
            pd_t = self.airms[n] * self.Pdot(state_t, self.model_ys_t[r])
            pd = pd_star + pd_t
            dp_star = self.dotP(state_star, (self.data[r][n,:] - pd)*self.ivars[r][n,:]) 
            lnlike += -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
            dlnlike_dw += dp_star
        lnprior = self.model_ys_lnprior(model_ys_star[r])
        dlnprior = self.dmodel_ys_lnprior_dw(model_ys_star[r])
        return lnlike + lnprior, dlnlike_dw + dlnprior

    def dlnlike_t_dw_t(self, r, model_ys_t):
        try:
            N = len(self.x0_t[r])
        except:
            N = 1  
        lnlike = 0.
        Mp = len(self.model_xs_t[r])
        dlnlike_dw = np.zeros(Mp)
        for n in range(N):
            state_star = self.state(self.x0_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
            pd_star = self.Pdot(state_star, self.model_ys_star[r])
            state_t = self.state(self.x0_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
            pd_t = self.airms[n] * self.Pdot(state_t, model_ys_t)
            pd = pd_star + pd_t
            dp_t = self.airms[n] * self.dotP(state_t, (self.data[r][n,:] - pd)*self.ivars[r][n,:]) 
            lnlike += -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
            dlnlike_dw += dp_t
        lnprior = self.model_ys_lnprior(model_ys_t[r])
        dlnprior = self.dmodel_ys_lnprior_dw(model_ys_t[r])
        return lnlike + lnprior, dlnlike_dw + dlnprior
    


    def improve_telluric_model(self, r, step_scale=5e-7, maxniter=50):
        w = np.copy(self.model_ys_t[r])
        lnlike_o = -1e10
        quitc = +1e10
        i = 0
        while ((quitc > 1) and (i < maxniter)):
            i += 1 
            lnlike, dlnlike_dw = self.dlnlike_t_dw_t(r, w)
            stepsize = step_scale * dlnlike_dw
            dlnlike = lnlike - lnlike_o
            if dlnlike > 0.0:
                w -= stepsize   
                step_scale *= 1.1
                quitc = lnlike - lnlike_o
                lnlike_o = lnlike + 0.0
            else:
                step_scale *= 0.5
        return w

    def improve_star_model(self, r, step_scale=5e-7, maxniter=50):
        w = np.copy(self.model_ys_star[r])
        lnlike_o = -1e10
        quitc = 1e10
        i = 0
        while ((quitc > 1) and (i < maxniter)):
            i += 1 
            lnlike, dlnlike_dw = self.dlnlike_star_dw_star(r, w)
            stepsize = step_scale * dlnlike_dw 
            dlnlike = lnlike - lnlike_o
            if dlnlike > 0.0:
                w -= stepsize
                step_scale *= 1.1
                quitc = lnlike_o - lnlike
                lnlike_o = lnlike + 0.0
            else:
                step_scale *= 0.5
        return w
        
    def optimize(self, restart = False, **kwargs):
        """
        Loops over all orders in the e2ds case and optimizes them all as separate spectra.
        Takes the same kwargs as optimize_order.
        """
        if (hasattr(self, 'model_xs_star') == False) or (restart == True):
            self.x0_star = [np.zeros(self.N) for r in range(self.R)]
            self.x0_t = [np.zeros(self.N) for r in range(self.R)]
            self.model_ys_star = [np.zeros(self.N) for r in range(self.R)] # not the right shape but whatevs
            self.model_xs_star = [np.zeros(self.N) for r in range(self.R)]
            self.model_ys_t = [np.zeros(self.N) for r in range(self.R)]
            self.model_xs_t = [np.zeros(self.N) for r in range(self.R)]
        
        self.soln_star = [np.zeros(self.N) for r in range(self.R)]
        self.soln_t = [np.zeros(self.N) for r in range(self.R)]
        
        for r in range(self.R):
            self.optimize_order(r, restart=restart, **kwargs)
              
        
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
        
        if (hasattr(self, 'model_xs_star') == False)  or (self.model_xs_star[r] == 0).all() \
                or (restart == True):
            self.x0_star[r] = -np.copy(self.true_rvs)
            self.x0_star[r] -= np.mean(self.x0_star[r])
            self.x0_t[r] = np.zeros(self.N)
            self.model_xs_star[r], self.model_ys_star[r] = self.make_template(r, self.x0_star[r])
            self.model_xs_t[r], self.model_ys_t[r] = self.make_template(r, self.x0_t[r])
            
        
        previous_lnlike = self.lnlike_star(self.x0_star[r], r)[0]
        assert previous_lnlike == self.lnlike_t(self.x0_t[r], r)[0]
        for iteration in range(niter):
            print "Fitting stellar RVs..."
            self.soln_star[r] =  minimize(-1.*self.lnlike_star, self.x0_star[r], args=(r),
                             method='BFGS', jac=True, options={'disp':True, 'gtol':1.e-2, 'eps':1.5e-5})['x']

            self.model_ys_t[r] = self.improve_telluric_model(r)
            self.model_ys_star[r] = self.improve_star_model(r)
            print "Star model improved. Fitting telluric RVs..."
            self.soln_t[r] =  minimize(-1*.self.lnlike_t, self.x0_t[r], args=(r),
                             method='BFGS', jac=True, options={'disp':True, 'gtol':1.e-2, 'eps':1.5e-5})['x']

            self.model_ys_t[r] = self.improve_telluric_model(r)
            self.model_ys_star[r] = self.improve_star_model(r)

            self.x0_star[r] = self.soln_star[r]
            self.x0_t[r] = self.soln_t[r]

            print "order {0}, iter {1}: star std = {2:.2f}, telluric std = {3:.2f}".format(r, iteration, np.std(self.soln_star[r] + self.bervs), np.std(self.soln_t[r]))
            if plot == True:
                alpha = (iteration + 1.) / niter 
                plt.plot(np.arange(self.N), self.soln_star[r] + self.bervs - np.mean(self.soln_star[r] + self.bervs), color='k', alpha=alpha)
                plt.plot(np.arange(self.N), self.soln_t[r] - np.mean(self.soln_t[r]), color='red', alpha=alpha)
                plt.show()
                
            new_lnlike = self.lnlike_star(self.x0_star[r], r)[0]
            if new_lnlike != self.lnlike_t(self.x0_t[r], r)[0]:
                print "new_lnlike for star: {0}, new_lnlike for tellurics: {1}".format(new_lnlike, self.lnlike_t(self.x0_t[r], r)[0])
                assert False
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration. Step-size issues?"
                assert False
            previous_lnlike = new_lnlike
            
    def show_results(self, r):
        """
        Plot three diagnostic plots. In order, the difference between the inferred RVs and those returned by the 
        HARPS pipeline, the same with the inferred telluric velocities at each epoch plotted as well, and a
        plot of the inferred RVs and the HARPS pipeline RVs overplotted (without the barycentric correction 
        removed).
        
        r is the index of the order to be plotted.
        """
        plt.scatter(np.arange(self.N), self.soln_star[r]+self.true_rvs)
        plt.show()
        
        plt.plot(np.arange(self.N), self.soln_star[r] + self.true_rvs - np.mean(self.soln_star[r] + self.true_rvs), color='k')
        plt.plot(np.arange(self.N), self.soln_t[r] - np.mean(self.soln_t[r]), color='red')
        plt.show()
        
        plt.plot(np.arange(self.N), self.soln_star[r] - np.mean(self.soln_star[r]) + self.bervs, 'ko')
        plt.plot(np.arange(self.N), -self.true_rvs + np.mean(self.true_rvs) + self.bervs, 'r.')
        plt.show()
        
    def save_results(self, filename):
        with h5py.File(filename,'w') as f:
            dset = f.create_dataset('model_xs_star', data=self.model_xs_star)
            dset = f.create_dataset('model_ys_star', data=self.model_ys_star)
            dset = f.create_dataset('model_xs_t', data=self.model_xs_t)
            dset = f.create_dataset('model_ys_t', data=self.model_ys_t)
            dset = f.create_dataset('rvs_star', data=self.soln_star)
            dset = f.create_dataset('rvs_t', data=self.soln_t)
