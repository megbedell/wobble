import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wobble
from time import time

if __name__ == "__main__":
    starname = '51peg'
    
    start_time = time()
    data = wobble.Data(starname+'_e2ds.hdf5', filepath='data/', orders=np.arange(72))
    print("data loaded")
    print("time elapsed: {0:.2f} s".format(time() - start_time))
    
    model = wobble.Model(data)
    model.add_star(starname)
    model.add_telluric('tellurics', rvs_fixed=True)
    print(model)
    print("time elapsed: {0:.2f} s".format(time() - start_time))
    

    plot_dir = 'results/plots/'
    for r in range(data.R):
        niter = 80
        if True: # no plots
            wobble.optimize_order(model, data, r, 
                        niter=niter, output_history=False)
        else: # plots out the wazoo
            nll_history, rvs_history, model_history, chis_history = wobble.optimize_order(model, data, r, 
                    niter=niter, output_history=True) 
            plt.scatter(np.arange(len(nll_history)), nll_history)
            ax = plt.gca()
            ax.set_yscale('log')
            plt.savefig(plot_dir+'nll_order{0}.png'.format(r))   
            plt.clf()
            rvs_ani = wobble.plot_rv_history(data, rvs_history, niter, 50)
            rvs_ani.save(plot_dir+'rvs_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('RVs animation saved')
            session = wobble.get_session()
            model_xs = session.run(model.components[0].model_xs[r])
            model_ani = wobble.plot_model_history(model_xs, model_history, niter, 50)
            model_ani.save(plot_dir+'model_order{0}.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('model animation saved')
            data_xs = session.run(data.xs[r])
            chis_ani = wobble.plot_chis_history(0, data_xs, chis_history, niter, 50)
            chis_ani.save(plot_dir+'chis_order{0}_epoch0.mp4'.format(r), fps=30, extra_args=['-vcodec', 'libx264'])
            print('chis animation saved')
        print("order {1} optimization finished. time elapsed: {0:.2f} s".format(time() - start_time, r))
        
        # HACK:    
        session = wobble.get_session()  
        data.wobble_obj.rvs_star[r] = session.run(model.components[0].rvs_block[r])          
        data.wobble_obj.rvs_t[r] = session.run(model.components[1].rvs_block[r])
        data.wobble_obj.model_xs_star[r] = session.run(model.components[0].model_xs[r])
        data.wobble_obj.model_ys_star[r] = session.run(model.components[0].model_ys[r])
        data.wobble_obj.model_xs_t[r] = session.run(model.components[1].model_xs[r])
        data.wobble_obj.model_ys_t[r] = session.run(model.components[1].model_ys[r])
        session.close()
        print("order {1} saved. time elapsed: {0:.2f} s".format(time() - start_time, r))
                
    # hackety hack hack
    data.wobble_obj.rvs_star = np.asarray(data.wobble_obj.rvs_star)
    data.wobble_obj.rvs_t = np.asarray(data.wobble_obj.rvs_t)
    data.wobble_obj.ivars_star = np.ones_like(data.wobble_obj.rvs_star) # HACK
    
    data.wobble_obj.optimize_sigmas()
    star_rvs = np.copy(data.wobble_obj.time_rvs)
    if starname=='hip54287':  # cut off post-upgrade epochs
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs[:-5] - data.bervs[:-5])))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs[:-5] - data.bervs[:-5])))
    else:
        print("HARPS pipeline std = {0:.3f} m/s".format(np.std(data.pipeline_rvs - data.bervs)))
        print("wobble std = {0:.3f} m/s".format(np.std(star_rvs - data.bervs)))
    
    data.wobble_obj.save_results(starname+'_wobbleflow_fixedt.hdf5')
    print("total runtime:{0:.2f} minutes".format((time() - start_time)/60.0))