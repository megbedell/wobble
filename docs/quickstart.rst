Quickstart
==========

Installation
------------

*wobble* is currently under development and not yet available through pip. However, you can install the current developer version from GitHub:

.. code-block:: bash

	git clone https://github.com/megbedell/wobble.git
	cd wobble
	python setup.py develop

If you are running macOS 10.14 (Mojave) and run into problems building the wobble C++ extensions, :doc:`try this <mojave>`.

Input Data
----------

Running *wobble* on a generic data set will require some restructuring of the data. If these data are in the standard HARPS *e2ds* format, they can be restructured using the :doc:`make_data.py script <scripts>`.

If you just want some example data for testing purposes, try grabbing one of these premade data sets: `51 Peg <https://www.dropbox.com/s/w1bgbnk80g0l1o9/51peg_e2ds.hdf5?dl=0>`_, `Barnard's Star <https://www.dropbox.com/s/mc7ahjsg0nkexx7/barnards_e2ds.hdf5?dl=0>`_, `HD 189733 <https://www.dropbox.com/s/pnmz9iq1alih3qj/HD189733_e2ds.hdf5?dl=0>`_.

Once the data are saved as an HDF5 file of the correct format, they are loaded as:

.. code-block:: python

	import wobble
	data = wobble.Data('filename.hdf5')

See the :doc:`API <api>` for full documentation of data loading options.

Running *wobble*
----------------

Once the data are loaded, you must create a wobble.Results object in which to store the calculation outputs. You must also construct one or more wobble.Model object(s). Data and Results objects span all echelle orders for the input spectra (or at least as many echelle orders as you specified in the *orders* keyword when loading Data), but a Model is specific to a single order. Thus if you want to combine RV constraints from multiple echelle orders, you'll need to define a model for each one.

A newly created wobble.Model object must be populated with one or more spectral components. Model.add_star() and Model.add_telluric() are convenience functions for this purpose. Here we'll construct a model that includes a single star and a tellurics component.

.. code-block:: python

	results = wobble.Results(data=data)
	for r in range(len(data.orders)):
		model = wobble.Model(data, results, r)
		model.add_star('star')
		model.add_telluric('tellurics')
		wobble.optimize_order(model)
		
Note that in the above code, we're overwriting the `model` variable at each order. This is fine because all parameters are automatically stored inside the `results` object at the end of each optimization.

Once all echelle orders have been modeled and optimized, we can automatically combine the RV constraints from each to get a net RV time series:

.. code-block:: python

	results.combine_orders('star')
	results.write('results.hdf5')

Accessing *wobble* Outputs
--------------------------

All of the outputs from *wobble* are stored in the wobble.Results object. You can download example results files corresponding to the above data files here: `51 Peg <https://www.dropbox.com/s/em4irz97zxqopx4/results_51peg_Kstar0_Kt3.hdf5?dl=0>`_, `Barnard's Star <https://www.dropbox.com/s/ymcu2awo1v05rps/results_barnards_Kstar0_Kt0.hdf5?dl=0>`_, `HD 189733 <https://www.dropbox.com/s/wz4ij56sfvwa037/results_HD189733_Kstar0_Kt0.hdf5?dl=0>`_.

A saved wobble.Results object can be loaded up from disk:

.. code-block:: python

	results = wobble.Results(filename='results.hdf5')
	print(results.component_names)
	
The names of the components are needed to access the associated attributes of each component. For example, let's say that two components are called 'star' and 'tellurics,' as in the example above. We can plot the mean templates for the two components in order `r` as follows:

.. code-block:: python

	import matplotlib.pyplot as plt
	plt.plot(np.exp(results.star_template_xs[r]), np.exp(results.star_template_ys[r]), 
			 label='star')
	plt.plot(np.exp(results.tellurics_template_xs[r]), np.exp(results.tellurics_template_ys[r]),
		 	label='tellurics')
	plt.xlabel('Wavelength (Ang)')
	plt.ylabel('Normalized Flux')
	plt.legend()
	plt.show()
	
And we can plot the RV time series for the star as follows:

.. code-block:: python

	plt.errorbar(results.dates, results.star_time_rvs + results.bervs, 
				 results.star_time_sigmas, 'k.')
	plt.xlabel('RV (m/s)')
	plt.ylabel('JD')
	plt.show()
	
Note that in the above example we had to explicitly add in barycentric corrections to the RVs. *wobble* does not do this within the Results object by default, as the true measured RV from the data is a velocity in the observatory rest frame.

Other useful quantities stored in the Results object include `results.ys_predicted`, which is an order R by epoch N by pixel M array of `y'` model predictions in the data space, and `results.[component name]_ys_predicted`, which is a same-sized array storing the contribution of a given component to the model prediction.

See the `notebook used to generate figures for the paper <https://github.com/megbedell/wobble/blob/master/paper/figures/make_figures.ipynb>`_ for further examples.