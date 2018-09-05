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
    K_t = 3
    
    star_reg_file = 'wobble/regularization/{0}_star_K{1}.hdf5'.format(starname, K_star)
    tellurics_reg_file = 'wobble/regularization/{0}_t_K{1}.hdf5'.format(starname, K_t)
    
    if False:
        # quick test on two orders
        data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=[30,56])
        results = wobble.Results(data=data)
        for r in range(data.R):
            model = wobble.Model(data, results, r)
            model.add_star('star', variable_bases=K_star, 
                            regularization_par_file=star_reg_file)
            model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t, 
                                regularization_par_file=tellurics_reg_file)
            wobble.optimize_order(model, niter=80, save_history=True, basename='results/plots_{0}/history'.format(starname))
        assert False
    
    
    start_time = time()
    orders = np.arange(10,72)
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=orders)
    results = wobble.Results(data=data)
    
    print("data loaded")
    print("time elapsed: {0:.2f} s".format(time() - start_time))

    plot_dir = 'results/plots_{0}_Kstar{1}_Kt{2}/'.format(starname, K_star, K_t)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    niter = 80
    plots = True
    for r,o in enumerate(orders):
        model = wobble.Model(data, results, r)
        model.add_star('star', variable_bases=K_star, 
                        regularization_par_file=star_reg_file)
        model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t, 
                            regularization_par_file=tellurics_reg_file)
        print("--- ORDER {0} ---".format(o))
        if plots:
            wobble.optimize_order(model, niter=80, save_history=True, 
                                  basename=plot_dir+'history')
        else:
            wobble.optimize_order(model, niter=80)
        del model # not sure if this does anything
        print("order {1} optimization finished. time elapsed: {0:.2f} s".format(time() - start_time, o))
    
    print("all orders optimized.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
       
    results.combine_orders('star')
    
    print("final RVs calculated.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
        
    results.write('results/results_{0}_Kstar{1}_Kt{2}.hdf5'.format(starname, K_star, K_t))
        
    print("results saved.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
    
    star_rvs = np.copy(results.star_time_rvs)
    if starname=='hip54287':  # cut off post-upgrade epochs
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs[:-5] + data.bervs[:-5])))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs[:-5] + data.bervs[:-5])))
    else:
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs + data.bervs)))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs + data.bervs)))
        
    fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
    ax.scatter(data.dates, data.pipeline_rvs + data.bervs, c='r', label='DRS', alpha=0.7)
    ax.scatter(data.dates, results.star_time_rvs + data.bervs, c='k', label='wobble', alpha=0.7)
    ax.legend()
    ax.set_xticklabels([])
    ax2.scatter(data.dates, results.star_time_rvs - data.pipeline_rvs, c='k')
    ax2.set_ylabel('JD')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt.savefig(plot_dir+'results_rvs.png')
    plt.close(fig)
    
    epochs = [0, 50] # to plot
    for r,o in enumerate(orders):
        for e in epochs:
            fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(12,5))
            ax.scatter(np.exp(data.xs[r][e]), np.exp(data.ys[r][e]), c='k', alpha=0.6)
            ax.plot(np.exp(data.xs[r][e]), np.exp(results.star_ys_predicted[r][e]), c='r', alpha=0.8)
            ax.plot(np.exp(data.xs[r][e]), np.exp(results.tellurics_ys_predicted[r][e]), c='b', alpha=0.8)
            ax2.scatter(np.exp(data.xs[r][e]), np.exp(data.ys[r][e]) - np.exp(results.star_ys_predicted[r][e]
                                                        + results.tellurics_ys_predicted[r][e]), 
                        c='k', alpha=0.6)
            ax.set_ylim([0.0,1.3])
            ax2.set_ylim([-0.08,0.08])
            ax.set_xticklabels([])
            fig.tight_layout()
            fig.subplots_adjust(hspace=0.05)
            plt.savefig(plot_dir+'results_synth_o{0}_e{1}.png'.format(o, e))
            plt.close(fig)
    
    
    print("plots saved.")    
    print("total runtime:{0:.2f} minutes".format((time() - start_time)/60.0))
