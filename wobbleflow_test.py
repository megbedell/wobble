import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wobble
from time import time
import h5py

if __name__ == "__main__":
    starname = '51peg'
    
    if False:
        # quick single-order test
        data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=[56])
        model = wobble.Model(data)
        model.add_star(starname)
        model.add_telluric('tellurics', rvs_fixed=True)
        wobble.optimize_order(model, data, 0, 
                niter=50, save_history=True, basename=starname) 
        history = wobble.History(model, data, 0, 50, filename=starname+'_o0_history.hdf5')      
        assert False
    
    
    start_time = time()
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(72))
    
    print("data loaded")
    print("time elapsed: {0:.2f} s".format(time() - start_time))
    
    model = wobble.Model(data)
    model.add_star(starname)
    K = 3
    model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K)
    print(model)
    print("time elapsed: {0:.2f} s".format(time() - start_time))
    
    #telluric_basis_vectors = np.zeros((data.R, K, 4096))
    #telluric_basis_weights = np.zeros((data.R, data.N, K))

    plot_dir = 'results/plots/'
    niter = 80
    for r in range(data.R):
        if False: # no plots
            results = wobble.optimize_order(model, data, r, 
                        niter=niter, save_history=False)
        else: # plots out the wazoo
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
    
    '''''
    data.wobble_obj.optimize_sigmas()
    star_rvs = np.copy(data.wobble_obj.time_rvs)
    if starname=='hip54287':  # cut off post-upgrade epochs
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs[:-5] - data.bervs[:-5])))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs[:-5] - data.bervs[:-5])))
    else:
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs - data.bervs)))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs - data.bervs)))
    
    data.wobble_obj.save_results(starname+'_wobbleflow.hdf5')
    '''
    results.write(starname+'_results_variablet.hdf5')
    print("total runtime:{0:.2f} minutes".format((time() - start_time)/60.0))