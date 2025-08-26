.. _recipes:

=======
Recipes
=======

Save model as a ``zarr`` store using ``xarray``
###############################################

.. include:: ./zarr-to-s3-obstore.py
   :literal:

Use ``xarray`` to predict tides from a ``zarr`` store
#####################################################

.. include:: ./xarray-predict.py
   :literal:

Use ``s3fs`` to setup a ``zarr`` store 
######################################

The above examples use ``obstore`` to access a ``zarr`` store on S3.
The ``s3fs`` package is an alternative method that creates a ``fsspec`` based store.

.. include:: ./setup-s3fs.py
   :literal:
