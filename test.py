import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wobble
from time import time
import h5py

if __name__ == "__main__":
    starname = '51peg'
    K_star = 0
    K_t = 3
    
    star_reg_file = 'wobble/regularization/{0}_star_K{1}.hdf5'.format(starname, K_star)
    tellurics_reg_file = 'wobble/regularization/{0}_t_K{1}.hdf5'.format(starname, K_t)
    
    if False:
        # quick test on two orders
        data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=[30,56])
        results = wobble.Results(data=data)
        model = wobble.Model(data, results, r)
        model.add_star('star', variable_bases=K_star, 
                            regularization_par_file=star_reg_file)
        model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t, 
                                regularization_par_file=tellurics_reg_file)
        wobble.optimize_order(model, niter=80)
        assert False
    
    
    start_time = time()
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(72))
    results = wobble.Results(data=data)
    
    print("data loaded")
    print("time elapsed: {0:.2f} s".format(time() - start_time))

    plot_dir = 'results/plots/'
    niter = 80
    if True: # no plots
        for r in range(data.R):
            model = wobble.Model(data, results, r)
            model.add_star('star', variable_bases=K_star, 
                            regularization_par_file=star_reg_file)
            model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K_t, 
                                regularization_par_file=tellurics_reg_file)
            print("--- ORDER {0} ---".format(r))
            wobble.optimize_order(model, niter=80)
            del model # not sure if this does anything
    else: # plots out the wazoo  
    # NOT WORKING YET      
        for r in range(data.R):
            results = wobble.optimize_order(model, 
                data, r, niter=niter, save_history=True, basename=starname)
            history = wobble.History(model, data, r, niter, filename=starname+'_o{0}_history.hdf5'.format(r))      
             
            plt.scatter(np.arange(len(history.nll_history)), history.nll_history)
            ax = plt.gca()
            ax.set_yscale('log')
            plt.savefig(plot_dir+'nll_order{0}.png'.format(r))   
            plt.clf()
            
            rvs_ani_star = history.plot_rvs(0, model, data, compare_to_pipeline=True)
            rvs_ani_star.save(plot_dir+'rvs_star_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            rvs_ani_t = history.plot_rvs(1, model, data)
            rvs_ani_t.save(plot_dir+'rvs_t_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('RVs animations saved')
            
            template_ani_star = history.plot_template(0, model, data, nframes=50)
            template_ani_star.save(plot_dir+'template_star_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            template_ani_t = history.plot_template(1, model, data, nframes=50)
            template_ani_t.save(plot_dir+'template_t_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('template animations saved')
            
            chis_ani = history.plot_chis(0, model, data, nframes=50)
            chis_ani.save(plot_dir+'chis_order{0}_epoch0.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('chis animation saved')
            
            session = wobble.get_session()
            epochs = [0, 5, 10, 15, 20] # random epochs to plot
            c = model.components[1]
            if c.K > 0:
                for e in epochs:
                    plt.plot(np.exp(results.xs[r][e,:]), np.exp(results.tellurics_ys_predicted[r][e,:]), label='epoch #{0}'.format(e), alpha=0.8)
                plt.ylim(0.6, 1.4)
                plt.legend(fontsize=14)
                plt.savefig(plot_dir+'variable_tellurics_order{0}.png'.format(r))
            print("order {1} optimization finished. time elapsed: {0:.2f} s".format(time() - start_time, r))
        
    print("all orders optimized.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
       
    results.combine_orders('star')
    
    print("final RVs calculated.")
    print("time elapsed: {0:.2f} minutes".format((time() - start_time)/60.0))
        
    if K_t > 0:
        results.write('results/'+starname+'_results_variablet.hdf5')
    else:
        results.write('results/'+starname+'_results_fixedt.hdf5')
        
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
    plt.savefig('results/{0}_Kstar{1}_Kt{2}_rvs.png'.format(starname, K_star, K_t))
    plt.close(fig)
    
    for o in np.arange(R):
        e = 0
        fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(12,5))
        ax.scatter(np.exp(data.xs[o][e]), np.exp(data.ys[o][e]), c='k', alpha=0.6)
        ax.plot(np.exp(data.xs[o][e]), np.exp(results.star_ys_predicted[o][e]), c='r', alpha=0.8)
        ax.plot(np.exp(data.xs[o][e]), np.exp(results.tellurics_ys_predicted[o][e]), c='b', alpha=0.8)
        ax2.scatter(np.exp(data.xs[o][e]), np.exp(data.ys[o][e]) - np.exp(results.star_ys_predicted[o][e]
                                                    + results.tellurics_ys_predicted[o][e]), 
                    c='k', alpha=0.6)
        ax.set_ylim([0.0,1.3])
        ax2.set_ylim([-0.08,0.08])
        ax.set_xticklabels([])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.savefig('results/{0}_Kstar{1}_Kt{2}_o{3}_e{4}.png'.format(starname, K_star, K_t, o, e))
        plt.close(fig)
    
    
    print("plots saved.")    
    print("total runtime:{0:.2f} minutes".format((time() - start_time)/60.0))
