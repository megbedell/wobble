import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from .utils import get_session

class History(object):
    """
    Information about optimization history of a single order stored in numpy arrays/lists.
    """   
    def __init__(self, model, niter):
        self.r = model.r
        self.order = model.order
        self.niter = niter
        self.data = model.data
        self.nll_history = np.empty(niter)
        self.synth_history = np.empty(np.append(niter, np.shape(self.data.ys[self.r])))
        self.rvs_history = [np.empty((niter, self.data.N)) for c in model.components]
        template_Ms = [int(c.template_ys.shape[0]) for c in model.components] # number of elements in templates
        self.template_history = [np.empty((niter, template_Ms[i])) for i,c in enumerate(model.components)]
        self.basis_vectors_history = [np.empty((niter, c.K, template_Ms[i])) for i,c in enumerate(model.components)]
        self.basis_weights_history = [np.empty((niter, self.data.N, c.K)) for c in model.components]

        
    def save_iter(self, model, i):
        """
        Save all necessary information after optimization number i
        """
        if not hasattr(model, 'nll'):
            print("ERROR: History: Model must be set up before save_iter() can be called.")
        session = get_session()
        self.nll_history[i] = session.run(model.nll)
        self.synth_history[i] = session.run(model.synth)
        for j,c in enumerate(model.components):
            self.template_history[j][i,:] = np.copy(session.run(c.template_ys))
            self.rvs_history[j][i,:] = np.copy(session.run(c.rvs))  
            if c.K > 0:
                self.basis_vectors_history[j][i,:,:] = np.copy(session.run(c.basis_vectors)) 
                self.basis_weights_history[j][i,:,:] = np.copy(session.run(c.basis_weights))
        if i == 0:
            self.template_xs = [session.run(c.template_xs) for c in model.components]
            
                
        
    def animfunc(self, i, xs, ys, xlims, ylims, ax, driver, xlabel, ylabel):
        """
        Produces each frame; called by History.plot()
        """
        ax.cla()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title('Optimization step #{0}'.format(i))
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        s = driver(xs, ys[i,:], alpha=0.8)
        
    def plot(self, xs, ys, linestyle='line', nframes=None, ylims=None, xlabel='', ylabel=''):
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
                    fargs=(xs, ys, xlims, ylims, ax, driver, xlabel, ylabel), interval=150)
        plt.close(fig)
        return ani  
                         
    def plot_rvs(self, ind, compare_to_pipeline=True, **kwargs):
        """
        Generate a matplotlib animation of RVs vs. time
        ind: index of component in model to be plotted
        compare_to_pipeline keyword subtracts off the HARPS DRS values (useful for removing BERVs)
        """
        xs = self.data.dates
        ys = self.rvs_history[ind]
        if compare_to_pipeline:
            ys -= np.repeat([self.data.pipeline_rvs], self.niter, axis=0)    
            ylabel = 'RV Residuals to HARPS Pipeline (m/s)'
        else:
            ylabel = 'RV (m/s)'
        return self.plot(xs, ys, linestyle='scatter', ylabel=ylabel, xlabel='JD', **kwargs)     
    
    def plot_template(self, ind, **kwargs):
        """
        Generate a matplotlib animation of the template inferred from data
        ind: index of component in model to be plotted
        """
        xs = np.exp(self.template_xs[ind])
        ys = np.exp(self.template_history[ind])
        # set up ylims if not otherwise specified:
        y_pad = 0.1
        ylims = (np.min(ys)-y_pad, min([np.max(ys)+y_pad, 10.])) # hard upper limit to prevent inf
        kwargs['ylims'] = kwargs.get('ylims', ylims)
        return self.plot(xs, ys, linestyle='line', ylabel='Normalized Flux', xlabel=r'Wavelength ($\AA$)', **kwargs)
        
    def animfunc_synth(self, i, xs, synths, data, data_mask, resids, xlims, ylims, ylims2, ax, ax2):
        ax.cla()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims2)
        ax.set_title('Optimization step #{0}'.format(i))
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)
        ax.set_ylabel('Normalized Flux', fontsize=14)
        ax2.set_ylabel('Resids', fontsize=14)
        l = ax.plot(xs, synths[i,:], alpha=0.8)
        s = ax.scatter(xs, data, marker=".", alpha=0.5, c='k', s=40)
        sm = ax.scatter(xs[data_mask], data[data_mask], marker=".", alpha=1., c='white', s=20)
        s2 = ax2.scatter(xs, resids[i,:], marker=".", alpha=0.5, c='k', s=40)
        sm2 = ax2.scatter(xs[data_mask], resids[i,:][data_mask], marker=".", alpha=1., c='white', s=20)
        
    def plot_synth(self, e, nframes=None, ylims=None, ylims2=None, **kwargs):
        """
        Generate a matplotlib animation of the data + model prediction for a given epoch
        e: index of epoch to be plotted
        """
        xs = np.exp(self.data.xs[self.r][e])
        synths = np.exp(self.synth_history[:,e,:])
        data = np.exp(self.data.ys[self.r][e])
        data_mask = self.data.ivars[self.r][e] <= 1.e-8
        resids =  np.exp(np.repeat([self.data.ys[self.r][e]], self.niter, axis=0)) - np.exp(self.synth_history[:,e,:])
        # set up ylims if not otherwise specified:
        y_pad = 0.1
        yl = (np.min(synths)-y_pad, min([np.max(synths)+y_pad, 10.])) # hard upper limit to prevent inf
        ylims = kwargs.get('ylims', yl)
        ylims2 = kwargs.get('ylims2', [-0.25, 0.25])
        if nframes is None:
            nframes = self.niter
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(12,5))
        x_pad = (np.max(xs) - np.min(xs)) * 0.1
        xlims = (np.min(xs)-x_pad, np.max(xs)+x_pad)
        ani = animation.FuncAnimation(fig, self.animfunc_synth, np.linspace(0, self.niter-1, nframes, dtype=int), 
                    fargs=(xs, synths, data, data_mask, resids, xlims, ylims, ylims2, ax, ax2), interval=150)
        plt.close(fig)
        return ani  
        
                
    def save_plots(self, basename, epochs_to_plot=[0,50], movies=True):
        plt.scatter(np.arange(len(self.nll_history)), self.nll_history)
        ax = plt.gca()
        ax.set_yscale('log')
        plt.savefig(basename+'_order{0}_nll.png'.format(self.order))   
        plt.clf()        
        if movies:
            rvs_ani_star = self.plot_rvs(0, compare_to_pipeline=True)
            rvs_ani_star.save(basename+'_order{0}_rvs_star.mp4'.format(self.order), fps=30, extra_args=['-vcodec', 'libx264'])    
            template_ani_star = self.plot_template(0, nframes=50)
            template_ani_star.save(basename+'_order{0}_template_star.mp4'.format(self.order), fps=30, extra_args=['-vcodec', 'libx264'])
            template_ani_t = self.plot_template(1, nframes=50)
            template_ani_t.save(basename+'_order{0}_template_t.mp4'.format(self.order), fps=30, extra_args=['-vcodec', 'libx264'])
            for e in epochs:
                synth_ani = self.plot_synth(e)
                synth_ani.save(basename+'_order{0}_epoch{1}.mp4'.format(self.order, e), fps=30, extra_args=['-vcodec', 'libx264'])