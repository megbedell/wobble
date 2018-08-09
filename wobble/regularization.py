import numpy as np
import matplotlib.pyplot as plt
import copy
import wobble
import tensorflow as tf
from tqdm import tqdm
from time import time

def fit_rvs_only(model, data, r, niter=80):
    synth = model.synthesize(r)
    nll = 0.5*tf.reduce_sum(tf.square(tf.boolean_mask(data.ys[r], data.epoch_mask) 
                                      - tf.boolean_mask(synth, data.epoch_mask)) 
                            * tf.boolean_mask(data.ivars[r], data.epoch_mask)) 
    
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
    for i in tqdm(range(niter)):         
        for c in model.components:
            if not c.rvs_fixed:            
                session.run(c.opt_rvs) # optimize RVs
            if c.K > 0:
                session.run(c.opt_basis) # optimize variable components
    results.copy_model(model) # update
    return results
    
def improve_order_regularization(model, data, r, verbose=True, plot=False, basename='', L1=True, L2=True): 
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
        print('---- ORDER #{0} ----'.format(r))
        print('star component:')
        for attr in ['L1_template', 'L2_template']:
            print('{0}: {1:.0e}'.format(attr, getattr(model.components[0], attr)))
        print('tellurics component:')
        for attr in ['L1_template', 'L2_template', 'L1_basis_vectors', 'L2_basis_vectors']:
            print('{0}: {1:.0e}'.format(attr, getattr(model.components[1], attr)))
        print('---------------')      
    
    
   
def improve_parameter(name, c, model, training_data, validation_data, r, verbose=True, plot=False, basename=''):
    """
    Perform a grid search to set the value of regularization parameter `name` in component `c`.
    Requires training data and validation data to evaluate goodness-of-fit for each parameter value.
    """
    current_value = getattr(c, name)
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
    setattr(c, name, grid[best_ind])
    
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
    setattr(c, name, val)
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
    o = 57
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=[o])
    model = wobble.Model(data)
    model.add_star('star')
    K = 3
    model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K)

    # from hand-tuning
    model.components[0].L1_template = 1.e4
    model.components[0].L2_template = 1.e3
    model.components[1].L1_template = 1.e3
    model.components[1].L2_template = 1.e3
    model.components[1].L1_basis_vectors = 1.e3
    model.components[1].L2_basis_vectors = 1.e5
    
    start_time = time()
    
    r = 0
    improve_order_regularization(model, data, r, verbose=True, 
                                    plot=True, basename='../regularization/o{0}'.format(o))
    time2 = time()
    print('regularization for order {1} took {0:.2f} min'.format((time2 - start_time)/60., o))

