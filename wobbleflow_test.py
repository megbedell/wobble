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
        histories = wobble.optimize_order(model, data, 0, 
                niter=50, output_history=True)
        (nll_history, rvs_history, template_history, chis_history, \
            basis_vectors_history, basis_weights_history) = histories        
        assert False
    
    
    start_time = time()
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(72))
    
    print("data loaded")
    print("time elapsed: {0:.2f} s".format(time() - start_time))
    
    model = wobble.Model(data)
    model.add_star(starname)
    K = 2
    model.add_telluric('tellurics', rvs_fixed=True, variable_bases=K)
    print(model)
    print("time elapsed: {0:.2f} s".format(time() - start_time))
    
    telluric_basis_vectors = np.zeros((data.R, K, 4096))
    telluric_basis_weights = np.zeros((data.R, data.N, K))

    plot_dir = 'results/plots/'
    niter = 80
    for r in range(data.R):
        if True: # no plots
            wobble.optimize_order(model, data, r, 
                        niter=niter, output_history=False)
        else: # plots out the wazoo
            nll_history, rvs_history, template_history, chis_history, \
                basis_vectors_history, basis_weights_history = wobble.optimize_order(model, 
                data, r, niter=niter, output_history=True) 
            plt.scatter(np.arange(len(nll_history)), nll_history)
            ax = plt.gca()
            ax.set_yscale('log')
            plt.savefig(plot_dir+'nll_order{0}.png'.format(r))   
            plt.clf()
            rvs_ani_star = wobble.plot_rv_history(data, rvs_history[0], niter, 50, compare_to_pipeline=True)
            rvs_ani_star.save(plot_dir+'rvs_star_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            rvs_ani_t = wobble.plot_rv_history(data, rvs_history[1], niter, 50, compare_to_pipeline=False)
            rvs_ani_t.save(plot_dir+'rvs_t_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('RVs animations saved')
            session = wobble.get_session()
            template_xs = session.run(model.components[0].template_xs[r])
            template_ani_star = wobble.plot_template_history(template_xs, template_history[0], niter, 50)
            template_ani_star.save(plot_dir+'template_star_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            template_xs = session.run(model.components[1].template_xs[r])
            template_ani_t = wobble.plot_template_history(template_xs, template_history[1], niter, 50)
            template_ani_t.save(plot_dir+'template_t_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('template animations saved')
            data_xs = session.run(data.xs[r])
            chis_ani = wobble.plot_chis_history(0, data_xs, chis_history, niter, 50)
            chis_ani.save(plot_dir+'chis_order{0}_epoch0.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('chis animation saved')
            epochs = [0, 5, 10, 15, 20] # random epochs to plot
            c = model.components[1]
            t_synth = session.run(tf.matmul(c.basis_weights[r], c.basis_vectors[r]))
            for e in epochs:
                plt.plot(np.exp(data_xs[e,:]), np.exp(t_synth[e,:]), label='epoch #{0}'.format(e), alpha=0.8)
            plt.ylim(0.6, 1.4)
            plt.legend(fontsize=14)
            plt.savefig(plot_dir+'variable_tellurics_order{0}.png'.format(r))
        print("order {1} optimization finished. time elapsed: {0:.2f} s".format(time() - start_time, r))
        
        # save telluric variability:
        session = wobble.get_session()
        if K > 0:
            c = model.components[1]
            telluric_basis_vectors[r,:,:] = np.copy(session.run(c.basis_vectors[r]))
            telluric_basis_weights[r,:,:] = np.copy(session.run(c.basis_weights[r]))
                    
        # HACK to save everything else:    
        data.wobble_obj.rvs_star[r] = session.run(model.components[0].rvs_block[r])          
        data.wobble_obj.rvs_t[r] = session.run(model.components[1].rvs_block[r])
        data.wobble_obj.model_xs_star[r] = session.run(model.components[0].template_xs[r])
        data.wobble_obj.model_ys_star[r] = session.run(model.components[0].template_ys[r])
        data.wobble_obj.model_xs_t[r] = session.run(model.components[1].template_xs[r])
        data.wobble_obj.model_ys_t[r] = session.run(model.components[1].template_ys[r])
        session.close()
        print("order {1} saved. time elapsed: {0:.2f} s".format(time() - start_time, r))
                
    # save telluric variability:
    if K > 0:    
        with h5py.File(starname+'_variable_tellurics.hdf5','w') as f:
            dset = f.create_dataset('telluric_basis_vectors', data=telluric_basis_vectors)
            dset = f.create_dataset('telluric_basis_weights', data=telluric_basis_weights)
    
    # hackety hack hack
    data.wobble_obj.rvs_star = np.asarray(data.wobble_obj.rvs_star)
    data.wobble_obj.rvs_t = np.asarray(data.wobble_obj.rvs_t)
    data.wobble_obj.ivars_star = np.ones_like(data.wobble_obj.rvs_star) # HACK
    data.wobble_obj.ivars_t = np.ones_like(data.wobble_obj.rvs_t) # HACK
    
    
    data.wobble_obj.optimize_sigmas()
    star_rvs = np.copy(data.wobble_obj.time_rvs)
    if starname=='hip54287':  # cut off post-upgrade epochs
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs[:-5] - data.bervs[:-5])))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs[:-5] - data.bervs[:-5])))
    else:
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs - data.bervs)))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs - data.bervs)))
    
    data.wobble_obj.save_results(starname+'_wobbleflow.hdf5')
    print("total runtime:{0:.2f} minutes".format((time() - start_time)/60.0))