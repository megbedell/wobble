import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.optimize import fmin_cg, minimize
import h5py
from utils import fit_continuum
import copy

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
        self.x0_star = [-np.copy(self.drs_rvs)+np.mean(self.drs_rvs) for r in range(self.R)]
        self.x0_t = [np.zeros(self.N) for r in range(self.R)]
        self.model_ys_star = [np.zeros(self.N) for r in range(self.R)] # not the right shape but whatevs, it's overwritten in the initialization of optimize_order
        self.model_xs_star = [np.zeros(self.N) for r in range(self.R)]
        self.model_ys_t = [np.zeros(self.N) for r in range(self.R)]
        self.model_xs_t = [np.zeros(self.N) for r in range(self.R)]
        self.soln_star = [np.zeros(self.N) for r in range(self.R)]
        self.soln_t = [np.zeros(self.N) for r in range(self.R)]
        self.ivars_star = [np.zeros(self.N) for r in range(self.R)]
        self.ivars_t = [np.zeros(self.N) for r in range(self.R)]
        
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
            rvs = self.x0_star[r]
            if self.model_ys_t[r] is not None:
                for n in range(self.N):
                    pd, dpd_dv = self.calc_pds(r, n, self.x0_t[r][n], 't')
                    resids[n] -= pd
        elif role == 't':
            rvs = self.x0_t[r]
            if self.model_ys_star[r] is not None:
                for n in range(self.N):
                    pd, dpd_dv = self.calc_pds(r, n, self.x0_star[r][n], 'star')
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
    
    def calc_pds(self, r, n, x0, role):
        if role == 'star':
            state_star = self.state(x0, self.data_xs[r][n], self.model_xs_star[r])    
            state_t = self.state(self.x0_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
            dpd_dv = self.dPdotdv(state_star, self.model_ys_star[r])
            
        elif role == 't':
            state_star = self.state(self.x0_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
            state_t = self.state(x0, self.data_xs[r][n], self.model_xs_t[r])
            dpd_dv = self.airms[n] * self.dPdotdv(state_t, self.model_ys_t[r])
            
        pd_star = self.Pdot(state_star, self.model_ys_star[r])
        pd_t = self.airms[n] * self.Pdot(state_t, self.model_ys_t[r])
        pd = pd_star + pd_t
        return pd, dpd_dv
        
    def lnlike(self, x0, r, role):
        lnlike = 0.
        dlnlike_dv = np.zeros(self.N)
        for n in range(self.N):
            pd, dpd_dv = self.calc_pds(r, n, x0[n], role)
            lnlike += -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
            dlnlike_dv[n] = np.sum((self.data[r][n,:] - pd) * self.ivars[r][n,:] * dpd_dv)
        if role == 'star':
            lnpost = lnlike + self.rv_lnprior(x0) + self.rv_lnprior(self.x0_t[r])
            dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0)
        elif role == 't':
            lnpost = lnlike + self.rv_lnprior(self.x0_star[r]) + self.rv_lnprior(x0)
            dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0)            
        return  lnpost,  dlnpost_dv
        
    def lnlike_dumb(self, rv, r, n, role):
        pd, dpd_dv = self.calc_pds(r, n, rv, role)
        lnlike = -0.5 * np.sum((self.data[r][n,:] - pd)**2 * self.ivars[r][n,:])
        dlnlike_dv = np.asarray([np.sum((self.data[r][n,:] - pd) * self.ivars[r][n,:] * dpd_dv), ])
        if role == 'star':
            x0 = 1. * self.x0_star[r]
            x0[n] = rv
            lnpost = lnlike + self.rv_lnprior(x0) + self.rv_lnprior(self.x0_t[r])
            dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0)[n] # HACK
        elif role == 't':
            x0 = 1. * self.x0_t[r]
            x0[n] = rv
            lnpost = lnlike + self.rv_lnprior(self.x0_star[r]) + self.rv_lnprior(x0)
            dlnpost_dv = dlnlike_dv + self.drv_lnprior_dv(x0)[n] # HACK
        return lnpost, dlnpost_dv
        
    def opposite_lnlike(self, x0, r, role):
        # for scipy.optimize
        lnpost, dlnpost_dv = self.lnlike(x0, r, role)
        return -1.* lnpost, -1.* dlnpost_dv
        
    def opposite_lnlike_dumb(self, rv, r, n, role):
        # for scipy.optimize
        lnpost, dlnpost_dv = self.lnlike_dumb(rv, r, n, role)
        return -1.* lnpost, -1.* dlnpost_dv
        
    
               
    def d2lnlike_dv2(self, r, role):
        # returns an N-vector corresponding to the second derivative of lnlike w.r.t v_n (of "star" or of "t")
        if role == 'star':
            x0 = self.x0_star[r]
        elif role == 't':
            x0 = self.x0_t[r]
        d2lnlikes = np.zeros_like(x0)
        for n in range(self.N):
            _, dpd_dv = self.calc_pds(r, n, x0[n], role)
            d2lnlikes[n] = np.dot(dpd_dv, self.ivars[r][n] * dpd_dv)
        d2lnposts = d2lnlikes # + self.d2rv_lnprior_dv2(x0)
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
            state_star = self.state(self.x0_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
            state_t = self.state(self.x0_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
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

    def improve_model(self, r, role, step_scale=5e-7, maxniter=100):
        if role == 'star':
            w = np.copy(self.model_ys_star[r])
        elif role == 't':
            w = np.copy(self.model_ys_t[r])
        lnlike_o = -np.Inf
        quitc = np.Inf
        i = 0
        while ((quitc > 0.001) and (i < maxniter)):
            i += 1 
            lnlike, dlnlike_dw = self.dlnlike_dw(r, w, role)
            stepsize = step_scale * dlnlike_dw
            dlnlike = lnlike - lnlike_o
            if dlnlike > 0.0:
                w += stepsize   
                step_scale *= 1.1
                quitc = lnlike - lnlike_o
                lnlike_o = lnlike + 0.0
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
            self.x0_star = [np.zeros(self.N) for r in range(self.R)]
            self.x0_t = [np.zeros(self.N) for r in range(self.R)]
            self.model_ys_star = [None for r in range(self.R)] # not the right shape but whatevs, it's overwritten in the initialization of optimize_order
            self.model_xs_star = [None for r in range(self.R)]
            self.model_ys_t = [None for r in range(self.R)]
            self.model_xs_t = [None for r in range(self.R)]
        
            self.soln_star = [np.zeros(self.N) for r in range(self.R)]
            self.soln_t = [np.zeros(self.N) for r in range(self.R)]
            self.ivars_star = [np.zeros(self.N) for r in range(self.R)]
            self.ivars_t = [np.zeros(self.N) for r in range(self.R)]
        
        for r in range(self.R):
            self.optimize_order(r, restart=restart, **kwargs)
            if (r % 5) == 0:
                self.save_results('state_order{0}.hdf5'.format(r))
                
        self.soln_star = np.asarray(self.soln_star)
        self.soln_t = np.asarray(self.soln_t)
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
            self.x0_star[r] = -np.copy(self.drs_rvs)
            self.x0_star[r] -= np.mean(self.x0_star[r])
            self.x0_t[r] = np.zeros(self.N)
            self.initialize_model(r, 'star')
            self.initialize_model(r, 't')
        
        previous_lnlike = self.lnlike(self.x0_star[r], r, 'star')[0]
        for iteration in range(niter):
            print "Fitting stellar RVs..."
            print self.x0_star[r]
            for n in range(self.N):
                self.soln_star[r][n] = minimize(self.opposite_lnlike_dumb, self.x0_star[r][n], args=(r, n, 'star'),
                             method='BFGS', jac=True, options={'disp':False})['x']
            self.x0_star[r] = self.soln_star[r]
            print self.x0_star[r]
            new_lnlike = self.lnlike(self.x0_star[r], r, 'star')[0]
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                print self.lnlike(self.x0_star[r], r, 'star')
                assert False
            previous_lnlike = new_lnlike 

                
            print "Improving stellar template spectra..."
            self.model_ys_star[r] = self.improve_model(r, 'star')
            new_lnlike = self.lnlike(self.x0_star[r], r, 'star')[0]
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                assert False
            previous_lnlike = new_lnlike 
                
            print "Fitting telluric RVs..."
            print self.x0_t[r]
            for n in range(self.N):
                self.soln_t[r][n] = minimize(self.opposite_lnlike_dumb, self.x0_t[r][n], args=(r, n, 't'),
                             method='BFGS', jac=True, options={'disp':False})['x']
            self.x0_t[r] = self.soln_t[r]
            print self.x0_t[r]
            new_lnlike = self.lnlike(self.x0_star[r], r, 'star')[0]
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                print self.lnlike(self.x0_t[r], r, 't')
                assert False
            previous_lnlike = new_lnlike 
                
            print "Improving telluric template spectra..."
            self.model_ys_t[r] = self.improve_model(r, 't')
            new_lnlike = self.lnlike(self.x0_star[r], r, 'star')[0]
            if new_lnlike < previous_lnlike:
                print "likelihood got worse this iteration."
                print new_lnlike, previous_lnlike, previous_lnlike - new_lnlike
                assert False
            previous_lnlike = new_lnlike 
                
    
            print "order {0}, iter {1}: star std = {2:.2f}, telluric std = {3:.2f}".format(r, iteration, np.std(self.soln_star[r] + self.bervs), np.std(self.soln_t[r]))
            print "                       RMS of resids w.r.t. HARPS DRS RVs = {0:.2f}".format(np.std(self.soln_star[r] + self.drs_rvs))
            if plot:
                plt.plot(np.arange(self.N), self.x0_star[r] + self.bervs - np.mean(self.x0_star[r] + self.bervs), color='k')
                plt.plot(np.arange(self.N), self.x0_t[r] - np.mean(self.x0_t[r]), color='red')
                plt.show()

            if plot:
                self.plot_models(r,0, filename='order{0}_iter{1}.png'.format(r, iteration))
        self.ivars_star[r] = self.d2lnlike_dv2(r, 'star')   
        self.ivars_t[r] = self.d2lnlike_dv2(r, 't') 
        
    def plot_models(self, r, n, filepath='../results/plots/', filename=None):

        #calculate models
        state_star = self.state(self.soln_star[r][n], self.data_xs[r][n], self.model_xs_star[r])
        model_star = self.Pdot(state_star, self.model_ys_star[r])
        state_t = self.state(self.soln_t[r][n], self.data_xs[r][n], self.model_xs_t[r])
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
        ax.set_ylim([0.0, 1.2])
        ax.legend(loc='lower right')
        ax.set_title('Order #{0}, Epoch #{1}'.format(r,n))
        if filename is None:
            filename = 'model_order{0}_epoch{1}.png'.format(r,n)
        plt.savefig(filepath+filename)
        plt.close(fig)          
            
    def show_results(self, r):
        """
        Plot three diagnostic plots. In order, the difference between the inferred RVs and those returned by the 
        HARPS pipeline, the same with the inferred telluric velocities at each epoch plotted as well, and a
        plot of the inferred RVs and the HARPS pipeline RVs overplotted (without the barycentric correction 
        removed).
        
        r is the index of the order to be plotted.
        """
        plt.scatter(np.arange(self.N), self.soln_star[r]+self.drs_rvs)
        plt.show()
        
        plt.plot(np.arange(self.N), self.soln_star[r] + self.drs_rvs - np.mean(self.soln_star[r] + self.drs_rvs), color='k')
        plt.plot(np.arange(self.N), self.soln_t[r] - np.mean(self.soln_t[r]), color='red')
        plt.show()
        
        plt.plot(np.arange(self.N), self.soln_star[r] - np.mean(self.soln_star[r]) + self.bervs, 'ko')
        plt.plot(np.arange(self.N), -self.drs_rvs + np.mean(self.drs_rvs) + self.bervs, 'r.')
        plt.show()
        
    def save_results(self, filename):
        max_len = np.max([len(x) for x in self.model_xs_star])
        for r in range(self.R): # resize to make rectangular arrays bc h5py is infuriating
            self.model_xs_star[r] = np.append(self.model_xs_star[r], np.zeros(max_len - len(self.model_xs_star[r])))
            self.model_ys_star[r] = np.append(self.model_ys_star[r], np.zeros(max_len - len(self.model_ys_star[r])))
            self.model_xs_t[r] = np.append(self.model_xs_t[r], np.zeros(max_len - len(self.model_xs_t[r])))
            self.model_ys_t[r] = np.append(self.model_ys_t[r], np.zeros(max_len - len(self.model_ys_t[r])))
        with h5py.File(filename,'w') as f:
            dset = f.create_dataset('rvs_star', data=self.soln_star)
            dset = f.create_dataset('ivars_star', data=self.ivars_star)
            dset = f.create_dataset('rvs_t', data=self.soln_t)
            dset = f.create_dataset('ivars_t', data=self.ivars_t)
            dset = f.create_dataset('model_xs_star', data=self.model_xs_star)
            dset = f.create_dataset('model_ys_star', data=self.model_ys_star)
            dset = f.create_dataset('model_xs_t', data=self.model_xs_t)
            dset = f.create_dataset('model_ys_t', data=self.model_ys_t)
            
    def load_results(self, filename):
        with h5py.File(filename) as f:
            self.soln_star = np.copy(f['rvs_star'])
            self.soln_t = np.copy(f['rvs_t'])
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
            
    def unpack_rv_pars(self, rv_pars):
        self.order_rvs = np.copy(rv_pars[:self.R])
        self.time_rvs = np.copy(rv_pars[self.R:self.R + self.N])
        self.order_vars = np.exp(rv_pars[self.R + self.N:])
                
    def lnlike_rvs(self, rv_pars):
        self.unpack_rv_pars(rv_pars)
        rv_predictions = np.tile(self.order_rvs[:,None], (1,self.N)) + np.tile(self.time_rvs, (self.R,1))
        resids = self.soln_star - rv_predictions
        all_vars = 1./self.ivars_star**2 + np.tile(self.order_vars[:,None], (1,self.N))
        lnlike = -0.5 * np.sum(resids**2 / all_vars + np.log(2. * np.pi * all_vars))
        dlnlike_drv_pars = np.zeros_like(rv_pars)
        dlnlike_drv_pars[:self.R] = np.sum(resids / all_vars, axis=1)
        dlnlike_drv_pars[self.R:self.R + self.N] = np.sum(resids / all_vars, axis=0)
        dlnlike_drv_pars[self.R + self.N:] = np.sum((0.5 * resids**2 / all_vars**2 - 0.5 / all_vars), axis=1)
        return lnlike, dlnlike_drv_pars
                
    def opposite_lnlike_rvs(self, rv_pars):
        lnlike, dlnlike_drv_pars = self.lnlike_rvs(rv_pars)
        return -1. * lnlike, -1. * dlnlike_drv_pars
        
    def combine_rvs(self):
        # first guesses:
        x0_order_rvs = np.median(self.soln_star, axis=1)
        x0_time_rvs = np.median(self.soln_star - np.tile(x0_order_rvs[:,None], (1, self.N)), axis=0)
        x0_rvs = np.append(x0_order_rvs, x0_time_rvs)
        rv_predictions = np.tile(x0_order_rvs[:,None], (1,self.N)) + np.tile(x0_time_rvs, (self.R,1))
        x0_rv_pars = np.append(x0_rvs, np.log(np.var(self.soln_star - rv_predictions, axis=1)))
        self.unpack_rv_pars(x0_rv_pars)
        if self.R < 2:
            return
        # optimize:
        soln_rv_pars = minimize(self.opposite_lnlike_rvs, x0_rv_pars,
                             method='BFGS', jac=True, options={'disp':True})['x']
        self.unpack_rv_pars(soln_rv_pars)
            
if __name__ == "__main__":
    # temporary code to diagnose issues in results
    starid = 'hip54287'
    a = star(starid+'_e2ds.hdf5', orders=np.arange(72), N=25)    
    
    if True: # optimize
        a.optimize_order(1, niter=20) # just to test
        a.optimize(niter=20, plot=True)
        a.save_results('../results/'+starid+'_results.hdf5')
        
    a.load_results('../results/'+starid+'_results.hdf5')
    
    if True: # make model plots
        for r in range(a.R):
            a.plot_models(r, 0)
        
    N_epochs = 35
    a.soln_star = np.asarray(a.soln_star)[:,:N_epochs] # just in case
    a.ivars_star = np.asarray(a.ivars_star)[:,:N_epochs]
    a.bervs = a.bervs[:N_epochs]
    a.N = N_epochs
    a.combine_rvs()
    print "RV std = {0:.2f} m/s".format(np.std(a.time_rvs + a.bervs))
    
    order_stds = np.std(a.soln_star + np.tile(a.bervs, (a.R,1)), axis=1)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.semilogy(np.arange(a.R), order_stds, marker='o', ls=' ')
    ax.set_ylabel(r'empirical std per order (m s$^{-1}$)')
    ax.set_xlabel('Order #')    
    plt.savefig('../results/plots/order_stds.png')
    plt.close(fig)
    
    
    cmap = matplotlib.cm.get_cmap(name='jet')
    colors_by_epoch = [cmap(1.*i/a.N) for i in range(a.N)]
    colors_by_order = [cmap(1.*i/a.R) for i in range(a.R)]
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.loglog([2.,1.e3],[2.,1.e3], c='red')
    ax.loglog(order_stds, np.sqrt(a.order_vars), marker='o', ls=' ')
    ax.set_ylabel(r'best-fit $\sigma_{order}$ (m s$^{-1}$)')
    ax.set_xlabel(r'empirical std per order (m s$^{-1}$)')    
    plt.savefig('../results/plots/order_stds_against_bestfit.png')
    plt.close(fig)
    
    orders_in_order = np.argsort(order_stds) # order inds from lowest to highest emp. variance
    order_counts = np.arange(1,72)
    time_rvs = np.zeros((len(order_counts), a.N))
    rv_stds = np.zeros(len(order_counts))
    for i,j in enumerate(order_counts):
        print "fitting orders", orders_in_order[:j]
        b = copy.deepcopy(a)
        b.R = j
        b.soln_star = b.soln_star[orders_in_order[:j]] # take only the j best orders
        b.ivars_star = b.ivars_star[orders_in_order[:j]]
        b.combine_rvs()
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
        fig = plt.figure()
        ax = plt.subplot(111) 
        for n in range(a.N): 
            ax.errorbar(np.arange(a.R), a.soln_star[:,n] + a.bervs[n], 1./np.sqrt(a.ivars_star[:,n]), 
                color=colors_by_epoch[n], alpha=0.5, fmt='o')
        ax.set_ylabel(r'barycentric RV (m s$^{-1}$)')
        ax.set_xlabel('Order #')
        plt.savefig('../results/plots/rv_per_order.png')
        plt.close(fig) 
    
        fig = plt.figure()
        ax = plt.subplot(111) 
        for r in range(a.R): 
            ax.errorbar(np.arange(a.N), a.soln_star[r,:] + a.bervs, 1./np.sqrt(a.ivars_star[r,:]), 
                color=colors_by_order[r], alpha=0.5, fmt='o')
        ax.set_ylabel(r'barycentric RV (m s$^{-1}$)')
        ax.set_xlabel('Epoch #')
        plt.savefig('../results/plots/rv_per_epoch.png')
        plt.close(fig) 
        
        rv_predictions = np.tile(a.order_rvs[:,None], (1,a.N)) + np.tile(a.time_rvs, (a.R,1))
        resids = a.soln_star - rv_predictions
    
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
    
    
        
