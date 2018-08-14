import numpy as np
import matplotlib.pyplot as plt
import copy
import wobble
import tensorflow as tf
from tqdm import tqdm
from time import time
import pickle

class Parameters(object):
    def __init__(self, R, filename=None):
        if filename is not None:
            self.load(filename)
        else:
            self.L1_template = [0. for r in range(R)]
            self.L2_template = [0. for r in range(R)]
            self.L1_basis_vectors = [0. for r in range(R)]
            self.L2_basis_vectors = [0. for r in range(R)]
            self.L2_basis_weights = [1. for r in range(R)]  # these will never be changed
        
    def update(self, r, c):
        """
        Update order r parameters to current values in model component c.
        **NOTE** r refers ONLY to the order index in Parameters() record; 
        c is assumed to be single-order!
        """
        for attr in ['L1_template', 'L2_template', 'L1_basis_vectors', 'L2_basis_vectors']:
            getattr(self, attr)[r] = np.copy(getattr(c, attr)[0])
            
    def copy_to_model(self, r, c):
        """
        Update regularization parameters in model component c to current values for order r.
        **NOTE** r refers ONLY to the order index in Parameters() record; 
        c is assumed to be single-order!
        """
        for attr in ['L1_template', 'L2_template', 'L1_basis_vectors', 'L2_basis_vectors']:
             getattr(c, attr)[0] = np.copy(getattr(self, attr)[r])       
            
    def set_order_to_previous(self, r):
        """
        Set order r parameters to those of order r-1.
        """
        try:
            for attr in ['L1_template', 'L2_template', 'L1_basis_vectors', 'L2_basis_vectors']:
                getattr(self, attr)[r] = np.copy(getattr(self, attr)[r-1]) 
        except:
            print('ERROR: cannot access previous order')
            
    def save(self, filename):
        """
        Save to pickle file.
        """
        pickle.dump(self, open(filename, 'wb'))
        
    def load(self, filename):
        """
        Load from pickle file; will overwrite entire Parameters() object.
        """
        self = pickle.load(open(filename, 'rb'))
             
def fit_rvs_only(model, data, r, niter=80):
    synth = model.synthesize(r)
    nll = 0.5*tf.reduce_sum(tf.square(tf.boolean_mask(data.ys[r], data.epoch_mask) 
                                      - tf.boolean_mask(synth, data.epoch_mask)) 
                            * tf.boolean_mask(data.ivars[r], data.epoch_mask)) 
    for c in model.components:
        if c.K > 0:
            nll += c.L2_basis_weights[r] * tf.reduce_sum(tf.square(c.basis_weights[r]))
    
    # set up optimizers: 
    session = wobble.get_session()

    for c in model.components:
        if not c.rvs_fixed:
            c.gradients_rvs = tf.gradients(nll, c.rvs_block[r])
            optimizer = tf.train.AdamOptimizer(c.learning_rate_rvs)
            c.opt_rvs = optimizer.minimize(nll, 
                            var_list=[c.rvs_block[r]])
            session.run(tf.variables_initializer(optimizer.variables()))
        if c.K > 0:
            c.gradients_basis = tf.gradients(nll, c.basis_weights[r])
            optimizer = tf.train.AdamOptimizer(c.learning_rate_basis)
            c.opt_basis = optimizer.minimize(nll, 
                            var_list=c.basis_weights[r])
            session.run(tf.variables_initializer(optimizer.variables()))
    
    results = wobble.Results(model=model, data=data)

    # optimize:
    for i in tqdm(range(niter), total=niter, miniters=int(niter/10)):         
        for c in model.components:
            if not c.rvs_fixed:            
                session.run(c.opt_rvs) # optimize RVs
            if c.K > 0:
                session.run(c.opt_basis) # optimize variable components
    results.copy_model(model) # update
    return results
    
def improve_order_regularization(model, data, r, verbose=True, plot=False, basename='', L1=True, L2=True): 
    """
    Use a validation scheme to determine the best regularization parameters for 
    all model components in a given order r.
    """
    validation_epochs = np.random.choice(data.N, data.N//10, replace=False)
    training_epochs = np.delete(np.arange(data.N), validation_epochs)
    
    training_data = copy.copy(data)
    training_data.epoch_mask = np.isin(np.arange(data.N), training_epochs)
    validation_mask = np.isin(np.arange(data.N), validation_epochs)
    validation_data = copy.copy(data)
    validation_data.epoch_mask = validation_mask 
    
    if L2:
        for c in model.components:
            improve_parameter('L2_template', c, model, training_data, validation_data, r, 
                            verbose=verbose, plot=plot, basename=basename+'_{0}'.format(c.name))
            if c.K > 0:
                improve_parameter('L2_basis_vectors', c, model, training_data, validation_data, r, 
                            verbose=verbose, plot=plot, basename=basename+'_{0}'.format(c.name))
    
    if L1:
        for c in model.components:
            improve_parameter('L1_template', c, model, training_data, validation_data, r, 
                            verbose=verbose, plot=plot, basename=basename+'_{0}'.format(c.name))
            if c.K > 0:
                improve_parameter('L1_basis_vectors', c, model, training_data, validation_data, r, 
                            verbose=verbose, plot=plot, basename=basename+'_{0}'.format(c.name))  
    
    if verbose:                       
        print('---- ORDER COMPLETE ----')
        print('star component:')
        for attr in ['L1_template', 'L2_template']:
            print('{0}: {1:.0e}'.format(attr, getattr(model.components[0], attr)[0]))
        print('tellurics component:')
        for attr in ['L1_template', 'L2_template', 'L1_basis_vectors', 'L2_basis_vectors']:
            print('{0}: {1:.0e}'.format(attr, getattr(model.components[1], attr)[0]))
        print('---------------')      
    
    
   
def improve_parameter(name, c, model, training_data, validation_data, r, verbose=True, plot=False, basename=''):
    """
    Perform a grid search to set the value of regularization parameter `name` in component `c`.
    Requires training data and validation data to evaluate goodness-of-fit for each parameter value.
    """
    current_value = getattr(c, name)[r]
    grid = np.logspace(-1.0, 1.0, num=3) * current_value
    chisqs_grid = np.zeros_like(grid)
    for i,val in enumerate(grid):
        chisqs_grid[i] = test_regularization_value(val, name, c, model, training_data, validation_data, r, 
                                        verbose=verbose, plot=plot, basename=basename)
        

    # ensure that the minimum isn't on a grid edge:
    best_ind = np.argmin(chisqs_grid)
    while best_ind == 0:
        val = grid[0]/10.
        chisq = test_regularization_value(val, name, c, model, training_data, validation_data, r, 
                                                verbose=verbose, plot=plot, basename=basename)
        if np.abs(chisq - chisqs_grid[0]) < 1.:
            break  # prevent runaway minimization
        grid = np.append(val, grid)
        chisqs_grid = np.append(chisq, chisqs_grid)
        best_ind = np.argmin(chisqs_grid)
        
    while best_ind == len(grid) - 1:
        val = grid[-1]*10.
        chisq = test_regularization_value(val, name, c, model, training_data, validation_data, r, 
                                                verbose=verbose, plot=plot, basename=basename)
        grid = np.append(grid, val)
        chisqs_grid = np.append(chisqs_grid, chisq)
        best_ind = np.argmin(chisqs_grid)
        
    # adopt best value:
    getattr(c, name)[r] = grid[best_ind]
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(grid, chisqs_grid)
        ax.set_xscale('log')
        #plt.yscale('log')
        ax.set_xlabel('{0} values'.format(name))
        ax.set_ylabel(r'$\chi^2$')
        plt.savefig('{0}_{1}_chis.png'.format(basename, name))
        plt.close(fig)
    if verbose:
        print("{0} optimized; setting to {1:.0e}".format(name, grid[best_ind]))
    
def test_regularization_value(val, name, c, model, training_data, validation_data, r, 
                                verbose=True, plot=False, basename=''):
    getattr(c, name)[r] = val
    
    for co in model.components:
        co.template_exists[r] = False # force reinitialization at each iteration
        
    results_train = wobble.optimize_order(model, training_data, r, niter=50)
    
    results = fit_rvs_only(model, validation_data, r)
    
    chisqs = (results.ys[r][validation_data.epoch_mask] 
              - results.ys_predicted[r][validation_data.epoch_mask])**2 * (results.ivars[r][validation_data.epoch_mask])
              
    if plot:
        validation_epochs = np.arange(validation_data.N)[validation_data.epoch_mask]
        e = validation_epochs[0] # random epoch
        xs = np.exp(results.xs[0][e])
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]})
        ax.plot(xs, np.exp(results.star_ys_predicted[0][e]), label='star model', lw=1.5, alpha=0.7)
        ax.plot(xs, np.exp(results.tellurics_ys_predicted[0][e]), label='tellurics model', lw=1.5, alpha=0.7)
        ax.scatter(xs, np.exp(results.ys[0][e]), marker=".", alpha=0.5, c='k', label='data')
        ax.set_xticklabels([])
        ax.set_ylabel('Normalized Flux', fontsize=14)
        ax2.scatter(xs, np.exp(results.ys[0][e]) - np.exp(results.ys_predicted[0][e]), marker=".", alpha=0.5, c='k')
        ax2.set_ylim([-0.05, 0.05])
        ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)
        ax2.set_ylabel('Resids', fontsize=14)
        
        ax.legend(fontsize=12)
        ax.set_title('{0}: value {1:.0e}, chisq {2:.0f}'.format(name, val, np.sum(chisqs)), 
             fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.savefig('{0}_{1}_val{2:.0e}.png'.format(basename, name, val))
        
        xlim = [np.percentile(xs, 20) - 7.5, np.percentile(xs, 20) + 7.5] # 15A near-ish the edge of the order
        ax.set_xlim(xlim)
        ax.set_xticklabels([])
        ax2.set_xlim(xlim)
        plt.savefig('{0}_{1}_val{2:.0e}_zoom.png'.format(basename, name, val))
        
        plt.close(fig)
        
    if verbose:
        print('{0}: value {1:.0e}, chisq {2:.0f}'.format(name, val, np.sum(chisqs)))
        
    return np.sum(chisqs)
    

        
if __name__ == "__main__":
    starname = '51peg'
    R = 72
    K = 3

    
    # initialize star regularization:
    print("initializing star regularization parameters...")
    star_parameters = Parameters(R)
    star_filename = 'wobble/regularization/{0}_star.p'.format(starname)

    
    # initialize tellurics regularization
    print("initializing telluric regularization parameters...")
    telluric_parameters = Parameters(R)
    telluric_filename = 'wobble/regularization/{0}_t_K{1}.p'.format(starname, K)

    
    start_time = time()
    
    for r in range(R):
        print("starting order {0}...".format(r))
        # make new data and model objects - this avoids wasting time with a giant results file
        data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=[r])
        model = wobble.Model(data)
        model.add_star('star')
        model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K)
        if r==0:
            star_parameters.L1_template[0] = 1.e1
            star_parameters.L2_template[0] = 1.e0
            telluric_parameters.L1_template[0] = 1.e0
            telluric_parameters.L2_template[0] = 1.e4
            telluric_parameters.L1_basis_vectors[0] = 1.e0
            telluric_parameters.L2_basis_vectors[0] = 1.e8
        else:  # initialize from previous order
            star_parameters.set_order_to_previous(r)
            telluric_parameters.set_order_to_previous(r)
        
        # initialize model parameters    
        star_parameters.copy_to_model(r, model.components[0])   
        telluric_parameters.copy_to_model(r, model.components[1])     
    
        improve_order_regularization(model, data, 0, verbose=True, 
                                    plot=False, basename='regularization/o{0}'.format(r))
        
        time2 = time()
        print('regularization for order {1} completed: time elapsed: {0:.2f} min'.format((time2 - start_time)/60., r))
        
        # save results to Parameters() objects
        star_parameters.update(r, model.components[0])
        telluric_parameters.update(r, model.components[1])
        if (r % 10) == 0:
            print("saving progress to {0} and {1}".format(star_filename, telluric_filename))
            star_parameters.save(star_filename)
            telluric_parameters.save(telluric_filename)
            
    # final save to disk
    star_parameters.save(star_filename)
    telluric_parameters.save(telluric_filename)
