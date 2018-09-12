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
        self.ivars_history = [np.empty((niter, self.data.N)) for c in model.components]
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
            self.ivars_history[j][i,:] = np.copy(session.run(c.ivars))  
            if c.K > 0:
                self.basis_vectors_history[j][i,:,:] = np.copy(session.run(c.basis_vectors)) 
                self.basis_weights_history[j][i,:,:] = np.copy(session.run(c.basis_weights))
        if i == 0:
            self.template_xs = [session.run(c.template_xs) for c in model.components]
            
                
        
    def animfunc(self, i, xs, ys, ys2, xlims, ylims, ax, driver):
        """
        Produces each frame; called by History.plot()
        """
        ax.cla()
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title('Optimization step #{0}'.format(i))
        if ys2 is not None:
            s2 = driver(xs, ys2, 'k', alpha=0.8)
        s = driver(xs, ys[i,:], alpha=0.8)
        
    def plot(self, xs, ys, ys2=None, linestyle='line', nframes=None, ylims=None):
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
                    fargs=(xs, ys, ys2, xlims, ylims, ax, driver), interval=150)
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
        return self.plot(xs, ys, linestyle='scatter', **kwargs)     
    
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
        return self.plot(xs, ys, linestyle='line', **kwargs)
        
    def plot_synth(self, e, **kwargs):
        """
        Generate a matplotlib animation of the data + model prediction for a given epoch
        e: index of epoch to be plotted
        """
        xs = np.exp(self.data.xs[self.r][e])
        ys = np.exp(self.synth_history[:,e,:])
        ys2 = np.exp(self.data.ys[self.r][e])
        # set up ylims if not otherwise specified:
        y_pad = 0.1
        ylims = (np.min(ys)-y_pad, min([np.max(ys)+y_pad, 10.])) # hard upper limit to prevent inf
        kwargs['ylims'] = kwargs.get('ylims', ylims)
        return self.plot(xs, ys, ys2=ys2, linestyle='line', **kwargs)
        
    def save_plots(self, basename, epochs=[0,50]):
        plt.scatter(np.arange(len(self.nll_history)), self.nll_history)
        ax = plt.gca()
        ax.set_yscale('log')
        plt.savefig(basename+'_order{0}_nll.png'.format(self.order))   
        plt.clf()
        
        rvs_ani_star = self.plot_rvs(0, compare_to_pipeline=True)
        rvs_ani_star.save(basename+'_order{0}_rvs_star.mp4'.format(self.order), fps=30, extra_args=['-vcodec', 'libx264'])
        
        template_ani_star = self.plot_template(0, nframes=50)
        template_ani_star.save(basename+'_order{0}_template_star.mp4'.format(self.order), fps=30, extra_args=['-vcodec', 'libx264'])
        template_ani_t = self.plot_template(1, nframes=50)
        template_ani_t.save(basename+'_order{0}_template_t.mp4'.format(self.order), fps=30, extra_args=['-vcodec', 'libx264'])

        for e in epochs:
            synth_ani = self.plot_synth(e)
            synth_ani.save(basename+'_order{0}_synth_epoch{1}.mp4'.format(self.order, e), fps=30, extra_args=['-vcodec', 'libx264'])