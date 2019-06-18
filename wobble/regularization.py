import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import h5py
import copy

from .data import Data
from .model import Model
from .results import Results
from .utils import get_session

def generate_regularization_file(filename, R, type='star'):
    """
    Create a regularization parameter file with default values.
    
    Parameters
    ----------
    filename : str
        Name of file to be made.
    R : int
        Number of echelle orders.
    type : str, optional
        Type of object; sets which default values to use. 
        Acceptable values are 'star' (default) or 'telluric'.
    """
    regularization_par = ['L1_template', 'L2_template', 
                              'L1_basis_vectors', 'L2_basis_vectors', 'L2_basis_weights']
    star_defaults = [1.e-2, 1.e2, 1.e5, 1.e6, 1.]
    telluric_defaults = [1.e4, 1.e6, 1.e3, 1.e8, 1.]
    if type=='star':
        defaults = star_defaults
    elif type=='telluric':
        defaults = telluric_defaults
    else:
        assert False, "ERROR: type not recognized."
    with h5py.File(filename,'w') as f:
        for par,val in zip(regularization_par, defaults):
            f.create_dataset(par, data=np.zeros(R)+val)

def get_name_from_tensor(tensor):
    # hacky method to get rid of characters TF adds to the variable names
    # NOTE - does not handle '_2' type additions!
    # also won't work if you put colons in your variable names but why would you do that?
    return str.split(tensor.name, ':')[0]
    
def setup_for_order(r, data, validation_epochs):
    """
    Set up necessary training & validation datasets and results objects
    for tuning the regularization on a given order from a given dataset.
                                 
    Parameters
    ----------
    r : int
        Index into `data.ys` to retrieve desired order.
    data : wobble.Data object
        The data used to tune regularization.
    validation_epochs : list of ints
        List of indices marking the epochs in `data` that
        will be set aside for validation. All other epochs will 
        make up the training set.
        
    Returns
    -------
    training_data : wobble.Data
    training_results : wobble.Results
    validation_data : wobble.Data
    validation_results : wobble.Results
    """
    # load up training and validation data for order r & set up results
    assert np.all(validation_epochs < data.N), "Invalid epoch index given."
    training_data = copy.copy(data)
    orders_to_cut = np.arange(data.R) != r
    training_data.delete_orders(orders_to_cut)
    validation_data = Data()
    validation_epochs[::-1].sort() # reverse sort in-place
    for v in validation_epochs:
        validation_data.append(training_data.pop(v))
    training_results = Results(training_data)
    validation_results = Results(validation_data)
    return training_data, training_results, validation_data, validation_results

def improve_order_regularization(o, star_filename, tellurics_filename,
                                 training_data, training_results,
                                 validation_data, validation_results,
                                 verbose=True, plot=False, plot_minimal=False,
                                 basename='', 
                                 K_star=0, K_t=0, L1=True, L2=True,
                                 tellurics_template_fixed=False): 
    """
    Use a validation scheme to determine the best regularization parameters for 
    all model components in a given order.
    Update files at star_filename, tellurics_filename with the best parameters.
                                 
    By default, this tunes in the following order: 
            tellurics L2, star L2, tellurics L1, star L1.
                                 
    Parameters
    ----------
    o : int
        Index into `star_filename` and `telluric_filename` to retrieve desired order.
    star_filename : str
        Filename containing regularization amplitudes for the star.
    tellurics_filename : str
        Filename containing regularization amplitudes for the tellurics.
    training_data : wobble.Data object
        Data to train template on (should be the majority of available data).
    training_results : wobble.Results object
        Results object corresponding to `training_data`.
    validation_data : wobble.Data object
        Data to use in assessing goodness-of-fit for template 
        (should be a representative minority of the available data).
    validation_results : wobble.Results object
        Results object corresponding to `validation_data`.
    verbose : bool (default `True`)
        Toggle print statements and progress bars.
    plot : bool (default `False`)
        Generate and save plots of fits to validation data including all variations
        of regularization amplitudes tried. (This will be a lot of plots!)
    plot_minimal : bool (default `False`)
        Generate and save only the before/after plots of fits to validation data.
    basename : str (default ``)
        String to append to the beginning of saved plots (file path and base).
    K_star : int (default `0`)
        Number of variable basis vectors for the star.
    K_t : int (default `0`)
        Number of variable basis vectors for the tellurics.
    L1 : bool (default `True`)
        Whether to tune L1 amplitudes.
    L2 : bool (default `True`)
        Whether to tune L2 amplitudes.
    """
    assert (training_data.R == 1) & (validation_data.R == 1)
    assert training_data.orders == validation_data.orders

    training_model = Model(training_data, training_results, 0)
    training_model.add_star('star', variable_bases=K_star)
    training_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t)
    training_model.setup()
    training_model.optimize(niter=0, verbose=verbose, rv_uncertainties=False)
    
    validation_model = Model(validation_data, validation_results, 0)
    validation_model.add_star('star', variable_bases=K_star, 
                          template_xs=training_results.star_template_xs[0]) # ensure templates are same size
    validation_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t,
                          template_xs=training_results.tellurics_template_xs[0])
    validation_model.setup()
    
    # the order in which these are defined will determine the order in which they are optimized:
    tensors_to_tune = [training_model.components[1].L2_template_tensor, training_model.components[0].L2_template_tensor,
                       training_model.components[1].L1_template_tensor, training_model.components[0].L1_template_tensor]
    tensor_names = ['L2_template', 'L2_template', 'L1_template',
                     'L1_template'] # this is only needed bc TF appends garbage to the end of the tensor name
    tensor_components = ['tellurics', 'star', 'tellurics', 'star'] # ^ same
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
                
    if plot or plot_minimal:
        test_regularization_value(tensor, 
                                  regularization_dict[tensor],
                                  training_model, validation_model, regularization_dict,
                                  validation_data, validation_results, plot=False, verbose=False) # hack to initialize validation results
        n = 0 # epoch to plot
        title = 'Initialization'
        filename = '{0}_init'.format(basename)
        plot_fit(0, n, validation_data, validation_results, title=title, basename=filename)

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
                    
    if plot or plot_minimal:
        test_regularization_value(tensor, regularization_dict[tensor],
                                  training_model, validation_model, regularization_dict,
                                  validation_data, validation_results, plot=False, verbose=False) # hack to update results
        title = 'Final'
        filename = '{0}_final'.format(basename)
        plot_fit(0, n, validation_data, validation_results, title=title, basename=filename)    
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        val_rvs = validation_results.star_rvs[0] + validation_results.bervs
        train_rvs = training_results.star_rvs[0] + training_results.bervs
        ax.plot(validation_results.dates, val_rvs - np.mean(val_rvs), 'r.', 
                label='validation set\n(may have RV offset)')
        ax.plot(training_results.dates, train_rvs - np.mean(train_rvs), 'k.', alpha=0.5,
                label='training set') 
        if not np.all(training_results.pipeline_rvs == 0.): # if pipeline RVs exist, plot them
            all_dates = np.append(validation_results.dates, training_results.dates)
            all_rvs = np.append(validation_results.pipeline_rvs + validation_results.bervs,
                                training_results.pipeline_rvs + training_results.bervs)
            ax.plot(all_dates, all_rvs - np.mean(all_rvs), 'b.', alpha=0.8, label='expected values')  
        ax.set_ylabel('RV (m/s)', fontsize=14)
        ax.set_xlabel('JD', fontsize=14)
        ax.legend(fontsize=12)
        fig.tight_layout()
        plt.savefig(basename+'_final_rvs.png')
        plt.close(fig)
        
    
    
def improve_parameter(par, training_model, validation_model, regularization_dict, 
                      validation_data, validation_results, 
                      plot=False, verbose=True, basename=''):
    """
    Perform a grid search to set the value of regularization parameter `par`.
    Requires training data and validation data to evaluate goodness-of-fit for each parameter value.
    Returns optimal parameter value.
    """
    current_value = np.copy(regularization_dict[par])
    if current_value == 0: # can't be scaled
        return 0
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
                              plot=False, verbose=True, basename='', 
                              training_niter=200, validation_niter=1000):
    '''
    Try setting regularization parameter `par` to value `val`; return goodness metric `nll`.
    '''
    r = training_model.r
    regularization_dict[par] = val
    name = get_name_from_tensor(par) # chop off TF's ID #
    session = get_session()
    session.run(tf.global_variables_initializer()) # reset both models
    
    training_model.optimize(niter=training_niter, feed_dict=regularization_dict, verbose=verbose, rv_uncertainties=False)
    validation_dict = {**regularization_dict}
    for c in validation_model.components:
        validation_dict[getattr(c, 'template_xs')] = getattr(training_model.results, 
                                                             c.name+'_template_xs')[r]
        validation_dict[getattr(c, 'template_ys')] = getattr(training_model.results, 
                                                             c.name+'_template_ys')[r]
        if c.K > 0:
            validation_dict[getattr(c, 'basis_vectors')] = getattr(training_model.results, 
                                                                   c.name+'_basis_vectors')[r]
    session = get_session()
    if verbose:
        iterator = tqdm(range(validation_niter))
    else:
        iterator = range(validation_niter)
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
    """Plots full-order and zoomed-in versions of fits & residuals for order `r`, epoch `n`."""
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
    """Takes an HDF5 file and automatically creates overview plots of regularization amplitudes."""
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
