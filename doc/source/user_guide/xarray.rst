xarray Integration
==================

Running these example recipes require an `aws installation <../getting_started/Install.html>`_ of the ``pyTMD`` package to include some of the optional dependencies.

Save model to a zarr store
##########################

.. include:: ./zarr-to-s3-obstore.py
    :literal:

Predict tides from model hosted on s3
#####################################

.. include:: ./xarray-predict.py
    :literal:

Use s3fs to setup a zarr store 
##############################

The above examples use ``obstore`` to access a ``zarr`` store on AWS s3.
The ``s3fs`` package is an alternative method that uses a ``fsspec`` based store.

.. include:: ./setup-s3fs.py
    :literal:
