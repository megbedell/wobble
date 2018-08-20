import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import sys
import h5py
import copy
import tensorflow as tf
T = tf.float64

from .utils import get_session

class History(object):
    """
    Information about optimization history of a single order stored in numpy arrays/lists
    """   
    def __init__(self, model, data, r, niter, filename=None):
        for c in model.components:
            assert c.template_exists[r], "ERROR: Cannot initialize History() until templates are initialized."
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