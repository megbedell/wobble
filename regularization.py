import numpy as np
import matplotlib.pyplot as plt
import copy
import wobble
import tensorflow as tf
from tqdm import tqdm
from time import time
import pickle
import pdb

__all__ = ["Parameters", "fit_rvs_only", "improve_order_regularization", "improve_parameter", "test_regularization_value"]

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
    
def improve_order_regularization(r, training_data, training_results,
                                 validation_data, validation_results,
                                 verbose=True, plot=False, basename='', 
                                 K_star=0, K_t=0, L1=True, L2=True): 
    """
    Use a validation scheme to determine the best regularization parameters for 
    all model components in a given order r.
    """
    
    training_model = wobble.Model(training_data, training_results, r)
    training_model.add_star('star', variable_bases=K_star)
    training_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t)
    training_model.setup()
    training_model.optimize(niter=0)
    
    validation_model = wobble.Model(validation_data, validation_results, r)
    validation_model.add_star('star', variable_bases=K_star, 
                          template_xs=training_results.star_template_xs[0])
    validation_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t,
                              template_xs=training_results.tellurics_template_xs[0])
    validation_model.setup()
    
    regularization_dict = {training_model.components[0].L1_template_tensor: 0.,
                      training_model.components[0].L2_template_tensor: 0.,
                      training_model.components[1].L1_template_tensor: 0.,
                      training_model.components[1].L2_template_tensor: 0.,
                      training_model.components[1].L1_basis_vectors_tensor: 0.,
                      training_model.components[1].L2_basis_vectors_tensor: 0.,
                      training_model.components[1].L2_basis_weights_tensor: 1.0}
    # INITIALIZE THIS PROPERLY

    for key in regularization_dict:
        name = key.name
        if (name[0:2] == "L1" and L1) or (name[0:2] == "L2" and L2):
            regularization_dict[key] = improve_parameter(key, training_model, validation_model, 
                                                         regularization_dict, verbose=verbose,
                                                         plot=plot, basename=basename)
    
    if verbose:                       
        print('---- ORDER COMPLETE ----')  
    
    
def improve_parameter(par, training_model, validation_model, regularization_dict, 
                      plot=False, verbose=True, basename=''):
    """
    Perform a grid search to set the value of regularization parameter `par`.
    Requires training data and validation data to evaluate goodness-of-fit for each parameter value.
    Returns optimal parameter value.
    """
    current_value = np.copy(regularization_dict[par])
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
        if np.abs(new_nll - nll_grid[0]) < 1.:
            break  # prevent runaway minimization
        grid = np.append(val, grid)
        chisqs_grid = np.append(new_nll, nll_grid)
        best_ind = np.argmin(nll_grid)
        
    while best_ind == len(grid) - 1:
        val = grid[-1]*10.
        new_nll = test_regularization_value(par, val, training_model, 
                                            validation_model, regularization_dict,                                                                           plot=plot, verbose=verbose, basename=basename)

        grid = np.append(grid, val)
        nll_grid = np.append(nll_grid, new_nll)
        best_ind = np.argmin(nll_grid)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(grid, nll_grid)
        ax.set_xscale('log')
        ax.set_xlabel('{0} values'.format(par.name))
        ax.set_ylabel('NLL')
        plt.savefig('{0}_{1}_nll.png'.format(basename, par.name))
        plt.close(fig)
    if verbose:
        print("{0} optimized to {1:.0e}".format(par.name, grid[best_ind]))
        
    return grid[best_ind]
    
def test_regularization_value(par, val, training_model, validation_model, regularization_dict, 
                              plot=False, verbose=True, basename=''):
    '''
    Try setting regularization parameter `par` to value `val`; return goodness metric `nll`.
    '''
    regularization_dict[par] = val
    training_model.optimize(niter=60, feed_dict=regularization_dict)
    validation_dict = {**regularization_dict}
    for c in validation_model.components:
        validation_dict[getattr(c, 'template_xs')] = getattr(training_model.results, 
                                                             c.name+'_template_xs')[training_model.r]
        validation_dict[getattr(c, 'template_ys')] = getattr(training_model.results, 
                                                             c.name+'_template_ys')[training_model.r]
        if c.K > 0:
            validation_dict[getattr(c, 'basis_vectors')] = getattr(training_model.results, 
                                                                   c.name+'_basis_vectors')[training_model.r]
    session = wobble.utils.get_session()
    for i in tqdm(range(60)):
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
                                                             c.name+'_template_xs')[training_model.r]
        zero_regularization_dict[getattr(c, 'template_ys')] = getattr(training_model.results, 
                                                             c.name+'_template_ys')[training_model.r]
        if not c.rvs_fixed:
            zero_regularization_dict[getattr(c, 'rvs')] = getattr(validation_model.results, 
                                                             c.name+'_rvs')[training_model.r]
        if c.K > 0:
            zero_regularization_dict[getattr(c, 'basis_vectors')] = getattr(training_model.results, 
                                                                   c.name+'_basis_vectors')[training_model.r]
            zero_regularization_dict[getattr(c, 'basis_weights')] = getattr(validation_model.results, 
                                                                   c.name+'_basis_weights')[training_model.r]


    if plot:
        n = 0
        xs = np.exp(validation_data.xs[r][n])
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]})
        ax.scatter(xs, np.exp(validation_data.ys[r][n]), marker=".", alpha=0.5, c='k', label='data')
        ax.plot(xs, 
                np.exp(validation_results.star_ys_predicted[r][n]), 
                color='r', label='star model', lw=1.5, alpha=0.7)
        ax.plot(xs, 
                np.exp(validation_results.tellurics_ys_predicted[r][n]), 
                color='b', label='tellurics model', lw=1.5, alpha=0.7)
        ax.set_xticklabels([])
        ax.set_ylabel('Normalized Flux', fontsize=14)
        resids = np.exp(validation_data.ys[r][n]) - np.exp(validation_results.star_ys_predicted[r][n]) \
                    - np.exp(validation_results.tellurics_ys_predicted[r][n])
        ax2.scatter(xs, resids, marker=".", alpha=0.5, c='k')
        ax2.set_ylim([-0.05, 0.05])
        ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=14)
        ax2.set_ylabel('Resids', fontsize=14)
        
        ax.legend(fontsize=12)
        ax.set_title('{0}: value {1:.0e}'.format(par.name, val), fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.savefig('{0}_{1}_val{2:.0e}.png'.format(basename, par.name, val))
        
        xlim = [np.percentile(xs, 20) - 7.5, np.percentile(xs, 20) + 7.5] # 15A near-ish the edge of the order
        ax.set_xlim(xlim)
        ax.set_xticklabels([])
        ax2.set_xlim(xlim)
        plt.savefig('{0}_{1}_val{2:.0e}_zoom.png'.format(basename, par.name, val))
        plt.close(fig)

    nll = session.run(validation_model.nll, feed_dict=zero_regularization_dict)
    if verbose:
        print('{0}, value {1:.0e}: nll {2:.2e}'.format(par.name, val, nll))
    return nll
    

        
if __name__ == "__main__":
    starname = '51peg'
    R = 72
    K_star = 0
    K_t = 3
    
    if True:
        R = 2
        data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(R))
        results = wobble.Results(data=data)
        
        validation_epochs = np.random.choice(data.N, data.N//10, replace=False)
        training_epochs = np.delete(np.arange(data.N), validation_epochs)
    
        training_data = wobble.Data('51peg_e2ds.hdf5', filepath='data/', orders=np.arange(R), 
                            epochs=training_epochs)
        training_results = wobble.Results(training_data)

        validation_data = wobble.Data('51peg_e2ds.hdf5', filepath='data/', orders=np.arange(R), 
                              epochs=validation_epochs)
        validation_results = wobble.Results(validation_data)
        
        r = 0
        training_model = wobble.Model(training_data, training_results, r)
        training_model.add_star('star', variable_bases=K_star)
        training_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t)
        training_model.setup()  
        training_model.optimize(niter=0)
        validation_model = wobble.Model(validation_data, validation_results, r)
        validation_model.add_star('star', variable_bases=0, 
                                  template_xs=training_results.star_template_xs[0])
        validation_model.add_telluric('tellurics', rvs_fixed=True, variable_bases=0,
                                      template_xs=training_results.tellurics_template_xs[0])
        validation_model.setup()
        
        regularization_dict = {training_model.components[0].L1_template_tensor: 0.,
                      training_model.components[0].L2_template_tensor: 1.e-1,
                      training_model.components[1].L1_template_tensor: 1.e1,
                      training_model.components[1].L2_template_tensor: 1.e5,
                      training_model.components[1].L1_basis_vectors_tensor: 0.,
                      training_model.components[1].L2_basis_vectors_tensor: 0.,
                      training_model.components[1].L2_basis_weights_tensor: 1.0}
        
        par = training_model.components[1].L2_template_tensor
        best_val = improve_parameter(par, training_model, validation_model, regularization_dict)

    if False:
        # initialize star regularization:
        print("initializing star regularization parameters...")
        star_parameters = Parameters(R)
        star_filename = 'wobble/regularization/{0}_star.p'.format(starname)
        #star_parameters.save(star_filename)


        # initialize tellurics regularization
        print("initializing telluric regularization parameters...")
        telluric_parameters = Parameters(R)
        telluric_filename = 'wobble/regularization/{0}_t_K{1}.p'.format(starname, K)



        start_time = time()

        data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(R))
        results = wobble.Results(data=data)

        validation_epochs = np.random.choice(data.N, data.N//10, replace=False)
        training_epochs = np.delete(np.arange(data.N), validation_epochs)

        training_data = wobble.Data('51peg_e2ds.hdf5', filepath='data/', orders=o, 
                                epochs=training_epochs)
        training_results = wobble.Results(training_data)

        validation_data = wobble.Data('51peg_e2ds.hdf5', filepath='data/', orders=o, 
                                  epochs=validation_epochs)
        validation_results = wobble.Results(validation_data)

        for r in range(R):
            print("starting order {0}...".format(r))

            '''
            if r==0:
                star_parameters.L1_template[0] = 1.e1
                star_parameters.L2_template[0] = 1.e1
                telluric_parameters.L1_template[0] = 1.e1
                telluric_parameters.L2_template[0] = 1.e5
                telluric_parameters.L1_basis_vectors[0] = 1.e5
                telluric_parameters.L2_basis_vectors[0] = 1.e6
            else:  # initialize from previous order
                star_parameters.set_order_to_previous(r)
                telluric_parameters.set_order_to_previous(r)

            # initialize model parameters    
            star_parameters.copy_to_model(r, model.components[0])   
            telluric_parameters.copy_to_model(r, model.components[1]) 
            '''


            improve_order_regularization(r, training_data, training_results,
                                         validation_data, validation_results, verbose=True, 
                                         plot=True, basename='regularization/o{0}'.format(r),
                                         K_star=K_star, K_t=K_t)

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
