import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wobble
from time import time
import h5py
import os

if __name__ == "__main__":
    starname = 'barnards'
    K_star = 0
    K_t = 0    
    niter = 150 # for optimization
    plots = True
    epochs = [0, 50] # to plot
    movies = False
    
    star_reg_file = '../wobble/regularization/{0}_star_K{1}.hdf5'.format(starname, K_star)
    tellurics_reg_file = '../wobble/regularization/{0}_t_K{1}.hdf5'.format(starname, K_t)
    plot_dir = '../results/plots_{0}_Kstar{1}_Kt{2}/'.format(starname, K_star, K_t)
    
    print("running wobble on star {0} with K_star = {1}, K_t = {2}".format(starname, K_star, K_t))
    start_time = time()
    orders = np.arange(72)
    data = wobble.Data(filename='../data/'+starname+'_e2ds.hdf5', orders=orders)
    if True: # reload data and remove all post-upgrade spectra
        upgrade = 2457174.5 # June 2015
        e = data.epochs[data.dates < upgrade]
        data = wobble.Data(filename='../data/'+starname+'_e2ds.hdf5', orders=orders, epochs=e)
    data.drop_bad_orders()
    data.drop_bad_epochs()
    orders = np.copy(data.orders)
    results = wobble.Results(data=data)
    
    print("data loaded")
    print("time elapsed: {0:.2f} min".format((time() - start_time)/60.0))
    elapsed_time = time() - start_time
    

    if plots:
        print("plots will be saved under directory: {0}".format(plot_dir))
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    star_learning_rate = 0.1
    telluric_learning_rate = 0.01
    for r,o in enumerate(orders):
        model = wobble.Model(data, results, r)
        model.add_star('star', variable_bases=K_star, 
                        regularization_par_file=star_reg_file, 
                        learning_rate_template=star_learning_rate)
        model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t, 
                            regularization_par_file=tellurics_reg_file, 
                            learning_rate_template=telluric_learning_rate)
        print("--- ORDER {0} ---".format(o))
        if plots:
            wobble.optimize_order(model, niter=niter, save_history=True, 
                                  basename=plot_dir+'history', movies=movies, epochs_to_plot=epochs) 
            fig, ax = plt.subplots(1, 1, figsize=(8,5))
            ax.plot(data.dates, results.star_rvs[r] + data.bervs - data.drifts - np.mean(results.star_rvs[r] + data.bervs), 
                    'k.', alpha=0.8, ms=4)
            ax.plot(data.dates, data.pipeline_rvs + data.bervs - np.mean(data.pipeline_rvs + data.bervs), 
                    'r.', alpha=0.5, ms=4)   
            ax.set_ylabel('RV (m/s)', fontsize=14)     
            ax.set_xlabel('BJD', fontsize=14)   
            plt.savefig(plot_dir+'results_rvs_o{0}.png'.format(o))
            plt.close(fig)           
            for e in epochs:
                results.plot_spectrum(r, e, data, plot_dir+'results_synth_o{0}_e{1}.png'.format(o, e))
        else:
            wobble.optimize_order(model, niter=niter)
        del model # not sure if this does anything
        print("order {1} optimization finished. time elapsed: {0:.2f} min".format((time() - start_time)/60.0, o))
        print("this order took {0:.2f} min".format((time() - start_time - elapsed_time)/60.0))
        elapsed_time = time() - start_time
    
    print("all orders optimized.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
    
    # do post-processing:
    results.combine_orders('star')
    results.apply_drifts('star')
    results.apply_bervs('star')
    
    if plots:
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
        ax.scatter(data.dates, data.pipeline_rvs - np.mean(data.pipeline_rvs), 
                    c='r', label='DRS', alpha=0.7, s=12)
        ax.scatter(data.dates, results.star_time_rvs - np.mean(results.star_time_rvs), 
                    c='k', label='wobble', alpha=0.7, s=12)
        ax.legend()
        ax.set_xticklabels([])
        ax2.scatter(data.dates, results.star_time_rvs - data.pipeline_rvs, c='k', s=12)
        ax2.set_ylabel('JD')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.savefig(plot_dir+'results_rvs.png')
        plt.close(fig)
    
    print("final RVs calculated.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
        
    # save output:
    results_file = '../results/results_{0}_Kstar{1}_Kt{2}.hdf5'.format(starname, K_star, K_t)
    results.write(results_file)
    star_rvs_file = '../results/rvs_{0}_Kstar{1}_Kt{2}.txt'.format(starname, K_star, K_t)
    results.write_rvs('star', star_rvs_file, all_orders=True)
    
        
    print("results saved as: {0} & {1}".format(results_file, star_rvs_file))
    print("-----------------------------")    
    print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs)))
    print("wobble std = {0:.3f} m/s".format(np.std(results.star_time_rvs)))        
    print("total runtime:{0:.2f} minutes".format((time() - start_time)/60.0))
