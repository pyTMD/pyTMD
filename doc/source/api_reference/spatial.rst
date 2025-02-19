=======
spatial
=======

- Spatial transformation routines
- Gravitational and ellipsoidal parameters :cite:p:`HofmannWellenhof:2006hy` :cite:p:`Petit:2010tp`

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/spatial.py

General Methods
===============

.. autofunction:: pyTMD.spatial.data_type

.. autoclass:: pyTMD.spatial.datum
   :members:

.. autofunction:: pyTMD.spatial.convert_ellipsoid

.. autofunction:: pyTMD.spatial.compute_delta_h

.. autofunction:: pyTMD.spatial.wrap_longitudes

.. autofunction:: pyTMD.spatial.to_dms

.. autofunction:: pyTMD.spatial.from_dms

.. autofunction:: pyTMD.spatial.to_cartesian

.. autofunction:: pyTMD.spatial.to_sphere

.. autofunction:: pyTMD.spatial.to_geodetic

.. autofunction:: pyTMD.spatial._moritz_iterative

.. autofunction:: pyTMD.spatial._bowring_iterative

.. autofunction:: pyTMD.spatial._zhu_closed_form

.. autofunction:: pyTMD.spatial.to_ENU

.. autofunction:: pyTMD.spatial.from_ENU

.. autofunction:: pyTMD.spatial.to_horizontal

.. autofunction:: pyTMD.spatial.to_zenith

.. autofunction:: pyTMD.spatial.scale_factors
