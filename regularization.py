import numpy as np
import matplotlib.pyplot as plt
import wobble
import tensorflow as tf
from tqdm import tqdm
import h5py
import pdb

__all__ = ["improve_order_regularization", "improve_parameter", "test_regularization_value"]

def get_name_from_tensor(tensor):
    # hacky method to get rid of characters TF adds to the variable names
    # NOTE - does not handle '_2' type additions!
    return str.split(tensor.name, ':')[0]

def improve_order_regularization(r, best_regularization_par,
                                 training_data, training_results,
                                 validation_data, validation_results,
                                 verbose=True, plot=False, basename='', 
                                 K_star=0, K_t=0, L1=True, L2=True): 
    """
    Use a validation scheme to determine the best regularization parameters for 
    all model components in a given order r.
    Update dict best_regularization_parameters and return it.
    """
    
    training_model = wobble.Model(training_data, training_results, r)
    training_model.add_star('star', variable_bases=K_star)
    training_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t)
    training_model.setup()
    training_model.optimize(niter=0)
    
    validation_model = wobble.Model(validation_data, validation_results, r)
    validation_model.add_star('star', variable_bases=K_star, 
                          template_xs=training_results.star_template_xs[r]) # ensure templates are same size
    validation_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t,
                              template_xs=training_results.tellurics_template_xs[r])
    validation_model.setup()
    
    # the order in which these are defined will determine the order in which they are optimized:
    tensors_to_tune = [training_model.components[0].L2_template_tensor, training_model.components[1].L2_template_tensor,
                       training_model.components[0].L1_template_tensor, training_model.components[1].L1_template_tensor]
    tensor_names = ['L2_template_star', 'L2_template_tellurics', 'L1_template_star', 'L1_template_tellurics'] # HACK
    if K_star > 0:
        tensors_to_tune = np.append(tensors_to_tune, [training_model.components[0].L1_basis_vectors_tensor, 
                                    training_model.components[0].L2_basis_vectors_tensor])
    if K_t > 0:
        tensors_to_tune = np.append(tensors_to_tune, [training_model.components[1].L1_basis_vectors_tensor, 
                                    training_model.components[1].L2_basis_vectors_tensor])
    
    regularization_dict = {}
    r_init = max(0, r-1) # initialize from previous order, or if r=0 use defaults
    for (tensor,name) in zip(tensors_to_tune, tensor_names):
        regularization_dict[tensor] = np.copy(best_regularization_par[name][r_init])

    for (tensor,name) in zip(tensors_to_tune, tensor_names):
        if (name[0:2] == "L1" and L1) or (name[0:2] == "L2" and L2):
            regularization_dict[tensor] = improve_parameter(tensor, training_model, validation_model, 
                                                         regularization_dict, verbose=verbose,
                                                         plot=plot, basename=basename)
            best_regularization_par[name][r] = np.copy(regularization_dict[tensor])
    
    
    if verbose:                       
        print('---- ORDER {0} COMPLETE ----'.format(r)) 
    return best_regularization_par
    
    
def improve_parameter(par, training_model, validation_model, regularization_dict, 
                      plot=False, verbose=True, basename=''):
    """
    Perform a grid search to set the value of regularization parameter `par`.
    Requires training data and validation data to evaluate goodness-of-fit for each parameter value.
    Returns optimal parameter value.
    """
    current_value = np.copy(regularization_dict[par])
    name = str.split(par.name, ':')[0] # chop off TF's ID #
    grid = np.logspace(-1.0, 1.0, num=3) * current_value
    nll_grid = np.zeros_like(grid)
    for i,val in enumerate(grid):
        nll_grid[i] = test_regularization_value(par, val, training_model, 
                                                validation_model, regularization_dict, 
                                                plot=plot, verbose=verbose, basename=basename)


    # ensure that the minimum isn't on a grid edge:
    best_ind = np.argmin(nll_grid)
    while best_ind == 0:
        val = grid[0]/10.
        new_nll = test_regularization_value(par, val, training_model, 
                                                validation_model, regularization_dict, 
                                                plot=plot, verbose=verbose, basename=basename)
        grid = np.append(val, grid)
        nll_grid = np.append(new_nll, nll_grid)
        best_ind = np.argmin(nll_grid)
        if val <= 1.e-8:
            break  # prevent runaway minimization
        
    while best_ind == len(grid) - 1:
        val = grid[-1]*10.
        new_nll = test_regularization_value(par, val, training_model, 
                                            validation_model, regularization_dict,                                                               
                                            plot=plot, verbose=verbose, basename=basename)

        grid = np.append(grid, val)
        nll_grid = np.append(nll_grid, new_nll)
        best_ind = np.argmin(nll_grid)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(grid, nll_grid)
        ax.axvline(grid[best_ind], c='k', alpha=0.7, ls='dashed', lw=2)
        ax.set_xlim([grid[0]*0.5, grid[-1]*2.])
        ax.set_xscale('log')
        ax.set_xlabel('{0} values'.format(name))
        ax.set_ylabel('NLL')
        fig.tight_layout()
        plt.savefig('{0}_{1}_nll.png'.format(basename, name))
        plt.close(fig)
    if verbose:
        print("{0} optimized to {1:.0e}".format(name, grid[best_ind]))
        
    return grid[best_ind]
    
def test_regularization_value(par, val, training_model, validation_model, regularization_dict, 
                              plot=False, verbose=True, basename=''):
    '''
    Try setting regularization parameter `par` to value `val`; return goodness metric `nll`.
    '''
    r = training_model.r
    regularization_dict[par] = val
    session = wobble.utils.get_session()
    session.run(tf.global_variables_initializer()) # reset both models
    
    training_model.optimize(niter=80, feed_dict=regularization_dict)
    validation_dict = {**regularization_dict}
    for c in validation_model.components:
        validation_dict[getattr(c, 'template_xs')] = getattr(training_model.results, 
                                                             c.name+'_template_xs')[r]
        validation_dict[getattr(c, 'template_ys')] = getattr(training_model.results, 
                                                             c.name+'_template_ys')[r]
        if c.K > 0:
            validation_dict[getattr(c, 'basis_vectors')] = getattr(training_model.results, 
                                                                   c.name+'_basis_vectors')[r]
    session = wobble.utils.get_session()
    for i in tqdm(range(80)):
        for c in validation_model.components:
            if not c.rvs_fixed:
                session.run(c.opt_rvs, feed_dict=validation_dict) # HACK
            if c.K > 0:
                session.run(c.opt_basis_weights, feed_dict=validation_dict)
                
    for c in validation_model.components:
        validation_model.results.update(c, feed_dict=validation_dict)
                
    zero_regularization_dict = {**regularization_dict} # for final chi-sq eval
    for key in zero_regularization_dict:
        zero_regularization_dict[key] = 0.0
    for c in validation_model.components:
        zero_regularization_dict[getattr(c, 'template_xs')] = getattr(training_model.results, 
                                                             c.name+'_template_xs')[r]
        zero_regularization_dict[getattr(c, 'template_ys')] = getattr(training_model.results, 
                                                             c.name+'_template_ys')[r]
        if not c.rvs_fixed:
            zero_regularization_dict[getattr(c, 'rvs')] = getattr(validation_model.results, 
                                                             c.name+'_rvs')[r]
        if c.K > 0:
            zero_regularization_dict[getattr(c, 'basis_vectors')] = getattr(training_model.results, 
                                                                   c.name+'_basis_vectors')[r]
            zero_regularization_dict[getattr(c, 'basis_weights')] = getattr(validation_model.results, 
                                                                   c.name+'_basis_weights')[r]


    if plot:
        name = get_name_from_tensor(par) # chop off TF's ID #
        n = 0
        xs = np.exp(validation_data.xs[r][n])
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(12,5))
        ax.scatter(xs, np.exp(validation_data.ys[r][n]), marker=".", alpha=0.5, c='k', label='data')
        ax.plot(xs, 
                np.exp(validation_results.star_ys_predicted[r][n]), 
                color='r', label='star model', lw=1.5, alpha=0.7)
        ax.plot(xs, 
                np.exp(validation_results.tellurics_ys_predicted[r][n]), 
                color='b', label='tellurics model', lw=1.5, alpha=0.7)
        ax.set_xticklabels([])
        ax.set_ylabel('Normalized Flux', fontsize=14)
        resids = np.exp(validation_data.ys[r][n]) - np.exp(validation_results.star_ys_predicted[r][n] 
                            + validation_results.tellurics_ys_predicted[r][n])
        ax2.scatter(xs, resids, marker=".", alpha=0.5, c='k')
        ax2.set_ylim([-0.1, 0.1])
        ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)
        ax2.set_ylabel('Resids', fontsize=14)
        
        ax.legend(fontsize=12)
        ax.set_title('{0}: value {1:.0e}'.format(name, val), fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.savefig('{0}_{1}_val{2:.0e}.png'.format(basename, name, val))
        
        xlim = [np.percentile(xs, 20) - 7.5, np.percentile(xs, 20) + 7.5] # 15A near-ish the edge of the order
        ax.set_xlim(xlim)
        ax.set_xticklabels([])
        ax2.set_xlim(xlim)
        plt.savefig('{0}_{1}_val{2:.0e}_zoom.png'.format(basename, name, val))
        plt.close(fig)

    nll = session.run(validation_model.nll, feed_dict=zero_regularization_dict)
    if verbose:
        print('{0}, value {1:.0e}: nll {2:.4e}'.format(name, val, nll))
    return nll

        
if __name__ == "__main__":
    starname = '51peg'
    R = 4
    K_star = 0
    K_t = 0
    plot = True
    
    # set up best_regularization_par dict:
    best_regularization_par = {}
    keys = ['L1_template_star', 'L2_template_star', 'L1_template_tellurics', 'L2_template_tellurics', 
            'L1_basis_vectors_star', 'L2_basis_vectors_star', 'L1_basis_vectors_tellurics', 
            'L2_basis_vectors_tellurics']
    for key in keys:
        best_regularization_par[key] = np.zeros(R)
    best_regularization_par['L2_basis_weights'] = np.ones(R) # never tuned, just need to pass to wobble
        
    # HAND-TUNE INITIALIZATION:
    best_regularization_par['L1_template_star'][0] = 1.e-2
    best_regularization_par['L2_template_star'][0] = 1.e3
    best_regularization_par['L1_template_tellurics'][0] = 1.e1
    best_regularization_par['L2_template_tellurics'][0] = 1.e5
    best_regularization_par['L1_basis_vectors_star'][0] = 1.e5
    best_regularization_par['L2_basis_vectors_star'][0] = 1.e6
    best_regularization_par['L1_basis_vectors_tellurics'][0] = 1.e5
    best_regularization_par['L2_basis_vectors_tellurics'][0] = 1.e6

    # set up training & validation data sets:
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(R)) # to get N_epochs    
    validation_epochs = np.random.choice(data.N, data.N//10, replace=False) # 10% of epochs will be validation set
    training_epochs = np.delete(np.arange(data.N), validation_epochs)
    training_data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(R), 
                        epochs=training_epochs)
    training_results = wobble.Results(training_data)
    validation_data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(R), 
                          epochs=validation_epochs)
    validation_results = wobble.Results(validation_data)
    
    # improve each order's regularization:
    for r in range(R):
        print('---- STARTING ORDER {0} ----'.format(r))
        best_regularization_par = improve_order_regularization(r, best_regularization_par,
                                         training_data, training_results,
                                         validation_data, validation_results,
                                         verbose=True, plot=plot, 
                                         basename='regularization/{0}_Kstar{1}_Kt{2}_o{3}_'.format(starname, K_star, K_t, r), 
                                         K_star=K_star, K_t=K_t, L1=True, L2=True)
        

    # save in format to be read by wobble:
    if R==72: # avoid overwriting with files that don't contain all orders
        regularization_par = ['L1_template', 'L2_template', 
                              'L1_basis_vectors', 'L2_basis_vectors', 'L2_basis_weights']
        star_filename = 'wobble/regularization/{0}_star_K{0}.p'.format(starname, K_star)
        with h5py.File(star_filename,'w') as f:
            for par in regularization_par:
                f.create_dataset(par, data=best_regularization_par[par+'_star'])
        tellurics_filename = 'wobble/regularization/{0}_t_K{1}.p'.format(starname, K_t)            
        with h5py.File(tellurics_filename,'w') as f:
            for par in regularization_par:
                f.create_dataset(par, data=best_regularization_par[par+'_tellurics'])
