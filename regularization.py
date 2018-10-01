import numpy as np
import matplotlib.pyplot as plt
import wobble
import tensorflow as tf
from tqdm import tqdm
import h5py
import os

__all__ = ["improve_order_regularization", "improve_parameter", "test_regularization_value", "plot_pars_from_file"]

def get_name_from_tensor(tensor):
    # hacky method to get rid of characters TF adds to the variable names
    # NOTE - does not handle '_2' type additions!
    return str.split(tensor.name, ':')[0]

def improve_order_regularization(r, o, star_filename, tellurics_filename,
                                 training_data, training_results,
                                 validation_data, validation_results,
                                 verbose=True, plot=False, basename='', 
                                 K_star=0, K_t=0, L1=True, L2=True): 
    """
    Use a validation scheme to determine the best regularization parameters for 
    all model components in a given order r.
    Update files at star_filename, tellurics_filename and return it.
    """
    
    training_model = wobble.Model(training_data, training_results, r)
    training_model.add_star('star', variable_bases=K_star)
    training_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t)
    training_model.setup()
    training_model.optimize(niter=0, verbose=verbose, uncertainties=False)
    
    if plot:
        n = 0 # epoch to plot
        title = 'Initialization'
        filename = '{0}_init'.format(basename)
        plot_fit(r, n, training_data, training_results, title=title, basename=filename)

    
    validation_model = wobble.Model(validation_data, validation_results, r)
    validation_model.add_star('star', variable_bases=K_star, 
                          template_xs=training_results.star_template_xs[r]) # ensure templates are same size
    validation_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t,
                              template_xs=training_results.tellurics_template_xs[r])
    validation_model.setup()
    
    # the order in which these are defined will determine the order in which they are optimized:
    tensors_to_tune = [training_model.components[1].L2_template_tensor, training_model.components[0].L2_template_tensor,
                       training_model.components[1].L1_template_tensor, training_model.components[0].L1_template_tensor]
    tensor_names = ['L2_template', 'L2_template', 'L1_template',
                     'L1_template'] # HACK - this is needed bc TF appends garbage to the end of the tensor name
    tensor_components = ['tellurics', 'star', 'tellurics', 'star'] # HACK
    if K_star > 0:
        tensors_to_tune = np.append(tensors_to_tune, [training_model.components[0].L2_basis_vectors_tensor, 
                                                    training_model.components[0].L1_basis_vectors_tensor])
        tensor_names = np.append(tensor_names, ['L2_basis_vectors', 'L1_basis_vectors'])
        tensor_components = np.append(tensor_components, ['star', 'star'])
    if K_t > 0:
        tensors_to_tune = np.append(tensors_to_tune, [training_model.components[1].L2_basis_vectors_tensor, 
                                                training_model.components[1].L1_basis_vectors_tensor])
        tensor_names = np.append(tensor_names, ['L2_basis_vectors', 'L1_basis_vectors'])
        tensor_components = np.append(tensor_components, ['tellurics', 'tellurics'])
    
    regularization_dict = {}
    #o_init = max(0, o-1) # initialize from previous order, or if o=0 use defaults
    o_init = o # always initialize from starting guess (TODO: decide which init is better)
    for i,tensor in enumerate(tensors_to_tune):
        if tensor_components[i] == 'star':
            filename = star_filename
        elif tensor_components[i] == 'tellurics':
            filename = tellurics_filename
        else:
            print("something has gone wrong.")
            assert False
        with h5py.File(filename, 'r') as f:                
                regularization_dict[tensor] = np.copy(f[tensor_names[i]][o_init])

    i = 0 # track order in which parameters are improved
    for component,(tensor,name) in zip(tensor_components, zip(tensors_to_tune, tensor_names)):
        if (name[0:2] == "L1" and L1) or (name[0:2] == "L2" and L2):
            i += 1
            regularization_dict[tensor] = improve_parameter(tensor, training_model, validation_model, 
                                                         regularization_dict, validation_data, validation_results, 
                                                         verbose=verbose,
                                                         plot=plot, basename=basename+'_par{0}'.format(i))
            if component == 'star':
                filename = star_filename
            elif component == 'tellurics':
                filename = tellurics_filename
            else:
                print("something has gone wrong.")
                assert False
            with h5py.File(filename, 'r+') as f:
                    f[name][o] = np.copy(regularization_dict[tensor])   
                    
    if plot:
        test_regularization_value(tensor, regularization_dict[tensor],
                                  training_model, validation_model, regularization_dict,
                                  validation_data, validation_results, plot=False, verbose=False) # hack to update results
        title = 'Final'
        filename = '{0}_final'.format(basename)
        plot_fit(r, n, validation_data, validation_results, title=title, basename=filename)       
    
    
def improve_parameter(par, training_model, validation_model, regularization_dict, 
                      validation_data, validation_results, 
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
                                                validation_data, validation_results, 
                                                plot=plot, verbose=verbose, basename=basename)


    # ensure that the minimum isn't on a grid edge:
    best_ind = np.argmin(nll_grid)
    while (best_ind == 0 and val >= 1.e-2): # prevent runaway minimization
        val = grid[0]/10.
        new_nll = test_regularization_value(par, val, training_model, 
                                                validation_model, regularization_dict, 
                                                validation_data, validation_results, 
                                                plot=plot, verbose=verbose, basename=basename)
        grid = np.append(val, grid)
        nll_grid = np.append(new_nll, nll_grid)
        best_ind = np.argmin(nll_grid)
        
    while best_ind == len(grid) - 1:
        val = grid[-1]*10.
        new_nll = test_regularization_value(par, val, training_model, 
                                            validation_model, regularization_dict,  
                                            validation_data, validation_results,                                                              
                                            plot=plot, verbose=verbose, basename=basename)

        grid = np.append(grid, val)
        nll_grid = np.append(nll_grid, new_nll)
        best_ind = np.argmin(nll_grid)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(grid, nll_grid, color='r')
        ax.plot(grid, nll_grid, c='r', ls='dashed', lw=1)
        ax.axvline(grid[best_ind], c='k', alpha=0.7, ls='dashed', lw=2)
        ax.set_ylim([nll_grid[best_ind]-10., nll_grid[best_ind]+100.])
        ax.set_xlim([grid[0]*0.5, grid[-1]*2.])
        ax.set_xscale('log')
        ax.set_xlabel('{0} values'.format(name))
        ax.set_ylabel('NLL')
        fig.tight_layout()
        plt.savefig('{0}_nll.png'.format(basename))
        plt.close(fig)
    if verbose:
        print("{0} optimized to {1:.0e}".format(name, grid[best_ind]))
        
    return grid[best_ind]
    
def test_regularization_value(par, val, training_model, validation_model, regularization_dict, 
                              validation_data, validation_results, 
                              plot=False, verbose=True, basename=''):
    '''
    Try setting regularization parameter `par` to value `val`; return goodness metric `nll`.
    '''
    r = training_model.r
    regularization_dict[par] = val
    name = get_name_from_tensor(par) # chop off TF's ID #
    session = wobble.utils.get_session()
    session.run(tf.global_variables_initializer()) # reset both models
    
    training_model.optimize(niter=80, feed_dict=regularization_dict, verbose=verbose, uncertainties=False)
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
    if verbose:
        iterator = tqdm(range(100))
    else:
        iterator = range(100)
    for i in iterator:
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
        n = 0 # epoch to plot
        title = '{0}: value {1:.0e}'.format(name, val)
        filename = '{0}_val{1:.0e}'.format(basename, val)
        plot_fit(r, n, validation_data, validation_results, title=title, basename=filename)

    nll = session.run(validation_model.nll, feed_dict=zero_regularization_dict)
    if verbose:
        print('{0}, value {1:.0e}: nll {2:.4e}'.format(name, val, nll))
    return nll
    
def plot_fit(r, n, data, results, title='', basename=''):
    fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(12,5))
    xs = np.exp(data.xs[r][n])
    ax.scatter(xs, np.exp(data.ys[r][n]), marker=".", alpha=0.5, c='k', label='data', s=16)
    mask = data.ivars[r][n] <= 1.e-8
    ax.scatter(xs[mask], np.exp(data.ys[r][n,mask]), marker=".", alpha=1., c='white', s=8)
    ax.plot(xs, 
            np.exp(results.star_ys_predicted[r][n]), 
            color='r', label='star model', lw=1.5, alpha=0.7)
    ax.plot(xs, 
            np.exp(results.tellurics_ys_predicted[r][n]), 
            color='b', label='tellurics model', lw=1.5, alpha=0.7)
    ax.set_xticklabels([])
    ax.set_ylabel('Normalized Flux', fontsize=14)
    resids = np.exp(data.ys[r][n]) - np.exp(results.star_ys_predicted[r][n] 
                        + results.tellurics_ys_predicted[r][n])
    ax2.scatter(xs, resids, marker=".", alpha=0.5, c='k')
    ax2.set_ylim([-0.1, 0.1])
    ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)
    ax2.set_ylabel('Resids', fontsize=14)
    
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.savefig('{0}.png'.format(basename))
    
    xlim = [np.percentile(xs, 20) - 7.5, np.percentile(xs, 20) + 7.5] # 15A near-ish the edge of the order
    ax.set_xlim(xlim)
    ax.set_xticklabels([])
    ax2.set_xlim(xlim)
    plt.savefig('{0}_zoom.png'.format(basename))
    plt.close(fig) 
    
def plot_pars_from_file(filename, basename, orders=np.arange(72)):
    """Takes an HDF5 file and automatically creates overview plots of regularization amplitudes"""
    with h5py.File(filename, 'r') as f:
        for key in list(f.keys()):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_yscale('log')
            ax.plot(orders, np.array(f[key])[orders], 'o')
            ax.set_xlabel('Order #')
            ax.set_ylabel('Regularization Amplitude')
            ax.set_title(key)
            ax.set_xlim([-3,75])
            fig.tight_layout()
            plt.savefig(basename+'_{0}.png'.format(key))
            plt.close(fig)      

        
if __name__ == "__main__":
    starname = 'barnards'
    orders = np.arange(72)
    K_star = 0
    K_t = 0
    plot = True
    verbose = False
    plot_dir = 'regularization/{0}_Kstar{1}_Kt{2}/'.format(starname, K_star, K_t)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    R = len(orders)
    
    # create HDF5 files if they don't exist:
    regularization_par = ['L1_template', 'L2_template', 
                          'L1_basis_vectors', 'L2_basis_vectors', 'L2_basis_weights']
    star_filename = 'wobble/regularization/{0}_star_K{1}.hdf5'.format(starname, K_star)
    if not os.path.isfile(star_filename):
        with h5py.File(star_filename,'w') as f:
            f.create_dataset('L1_template', data=np.zeros(R)+1.e-2)
            f.create_dataset('L2_template', data=np.zeros(R)+1.e-2)
            if K_star > 0:
                f.create_dataset('L1_basis_vectors', data=np.zeros(R)+1.e5)
                f.create_dataset('L2_basis_vectors', data=np.zeros(R)+1.e6)
                f.create_dataset('L2_basis_weights', data=np.ones(R)) # never tuned, just need to pass to wobble
                

    tellurics_filename = 'wobble/regularization/{0}_t_K{1}.hdf5'.format(starname, K_t)
    if not os.path.isfile(tellurics_filename):                
        with h5py.File(tellurics_filename,'w') as f:
            f.create_dataset('L1_template', data=np.zeros(R)+1.e5)
            f.create_dataset('L2_template', data=np.zeros(R)+1.e5)
            if K_t > 0:
                f.create_dataset('L1_basis_vectors', data=np.zeros(R)+1.e3)
                f.create_dataset('L2_basis_vectors', data=np.zeros(R)+1.e9)
                f.create_dataset('L2_basis_weights', data=np.ones(R)) # never tuned, just need to pass to wobble

    # set up training & validation data sets:
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=orders) # to get N_epochs    
    validation_epochs = np.random.choice(data.N, data.N//10, replace=False) # 10% of epochs will be validation set
    training_epochs = np.delete(np.arange(data.N), validation_epochs)
    training_data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=orders, 
                        epochs=training_epochs)
    training_results = wobble.Results(training_data)
    validation_data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=training_data.orders, 
                          epochs=validation_epochs)
    validation_results = wobble.Results(validation_data)
    assert len(training_data.orders) == len(validation_data.orders), "Number of orders used is not the same between training and validation data."
    orders = training_data.orders
    
    
    # improve each order's regularization:
    for r,o in enumerate(orders): # r is an index into the (cleaned) data. o is an index into the 72 orders (and the file tracking them).
        print('---- STARTING ORDER {0} ----'.format(o))
        improve_order_regularization(r, o, star_filename, tellurics_filename,
                                         training_data, training_results,
                                         validation_data, validation_results,
                                         verbose=verbose, plot=plot, 
                                         basename='{0}o{1}'.format(plot_dir, o), 
                                         K_star=K_star, K_t=K_t, L1=True, L2=True)
                                         
        print('---- ORDER {0} COMPLETE ({1}/{2}) ----'.format(o,r,len(orders)))
        print("best values:")
        print("star:")
        with h5py.File(star_filename, 'r') as f:
            for key in list(f.keys()):
                print("{0}: {1:.0e}".format(key, f[key][o]))
        print("tellurics:")
        with h5py.File(tellurics_filename, 'r') as f:
            for key in list(f.keys()):
                print("{0}: {1:.0e}".format(key, f[key][o]))        

    plot_pars_from_file(star_filename, 'regularization/{1}_star_Kstar{2}_Kt{3}'.format(starname, K_star, K_t), orders=orders)
    plot_pars_from_file(tellurics_filename, 'regularization/{1}_tellurics_Kstar{2}_Kt{3}'.format(starname, K_star, K_t), orders=orders)          
