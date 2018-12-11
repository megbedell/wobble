Quickstart
==========

Installation
------------

*wobble* is currently under development and not yet available through pip. However, you can install the current developer version from GitHub:

.. code-block:: bash

	git clone https://github.com/megbedell/wobble.git
	cd wobble
	python setup.py install

Input Data
----------

Running *wobble* on a generic data set will require some restructuring of the data. If these data are in the standard HARPS *e2ds* format, they can be restructured using the :doc:`make_data.py script <scripts>`.

If you just want some example data for testing purposes, try grabbing one of these premade data sets: `51 Peg <https://www.dropbox.com/s/a9hxhlr8gxrt9hc/51peg_e2ds.hdf5?dl=0>`_, `Barnard's Star <https://www.dropbox.com/s/mc7ahjsg0nkexx7/barnards_e2ds.hdf5?dl=0>`_, `HD 189733 <https://www.dropbox.com/s/pnmz9iq1alih3qj/HD189733_e2ds.hdf5?dl=0>`_.

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

	for r in range(len(data.orders)):
		model = wobble.Model(data, results, r)
		model.add_star('star')
		model.add_tellurics('tellurics')
		wobble.optimize_order(model)
		
Note that in the above code, we're overwriting the `model` variable at each order. This is fine because all parameters are automatically stored inside the `results` object at the end of each optimization.

Once all echelle orders have been modeled and optimized, we can automatically combine the RV constraints from each to get a net RV time series:

.. code-block:: python

	results.combine_orders('star')
	results.write('results.hdf5')

Accessing *wobble* Outputs
--------------------------