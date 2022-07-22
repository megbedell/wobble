import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wobble
from time import time
import h5py
import os

if __name__ == "__main__":
    starname = 'HD189733'
    K_star = 0
    K_t = 0    
    niter = 100 # for optimization
    plots = True
    epochs = [0, 20] # to plot
    movies = False
    
    plot_dir = '../results/plots_{0}_Kstar{1}_Kt{2}/'.format(starname, K_star, K_t)
    
    print("running wobble on star {0} with K_star = {1}, K_t = {2}".format(starname, K_star, K_t))
    start_time = time()
    orders = np.arange(72)
    '''
    e = [ 0,  1,  6,  7,  9, 17, 18, 19, 21, 23, 24, 26, 30, 33, 34, 35, 36,
       37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 56, 61,
       66, 69, 70, 72, 73, 75] # night of August 28, 2007
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='../data/', orders=orders, epochs=e)
    '''
    data = wobble.Data(filename='../data/'+starname+'_e2ds.hdf5', orders=orders)
    orders = np.copy(data.orders)
    results = wobble.Results(data=data)
    
    results_51peg = wobble.Results(filename='/Users/mbedell/python/wobble/results/results_51peg_Kstar0_Kt0.hdf5')
    
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
                        regularization_par_file=None, 
                        learning_rate_template=star_learning_rate)
        model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t, 
                            learning_rate_template=telluric_learning_rate,
                            template_fixed=True, template_xs=results_51peg.tellurics_template_xs[o],
                            template_ys=results_51peg.tellurics_template_ys[o]) # assumes all orders are there for 51 Peg
        print("--- ORDER {0} ---".format(o))
        if plots:
            wobble.optimize_order(model, niter=niter, save_history=True, 
                                  basename=plot_dir+'history', epochs_to_plot=epochs, movies=movies)
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
       
    results.combine_orders('star')
    
    print("final RVs calculated.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
        
    results_file = '../results/results_{0}_Kstar{1}_Kt{2}.hdf5'.format(starname, K_star, K_t)
    results.write(results_file)
        
    print("results saved as: {0}".format(results_file))
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
        
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
        ax.scatter(data.dates % 2.21857312, data.pipeline_rvs - np.mean(data.pipeline_rvs), 
                    c='r', label='DRS', alpha=0.7, s=12)
        ax.scatter(data.dates % 2.21857312, results.star_time_rvs - np.mean(results.star_time_rvs), 
                    c='k', label='wobble', alpha=0.7, s=12)
        ax.legend()
        ax.set_xticklabels([])
        ax2.scatter(data.dates, results.star_time_rvs - data.pipeline_rvs, c='k', s=12)
        ax2.set_ylabel('JD')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.savefig(plot_dir+'results_rvs_phased.png')
        plt.close(fig)
    
    print("final RVs calculated.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
        
    # save output:
    results_file = 'results/results_{0}_Kstar{1}_Kt{2}.hdf5'.format(starname, K_star, K_t)
    results.write(results_file)
    star_rvs_file = 'results/rvs_{0}_Kstar{1}_Kt{2}.hdf5'.format(starname, K_star, K_t)
    results.write_rvs('star', star_rvs_file, all_orders=True)
    
        
    print("results saved as: {0} & {1}".format(results_file, star_rvs_file))
    print("-----------------------------")         
    print("total runtime:{0:.2f} minutes".format((time() - start_time)/60.0))
