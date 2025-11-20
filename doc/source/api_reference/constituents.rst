============
constituents
============

- Calculates constituent parameters and nodal arguments

.. note::
    Originally based on Richard Ray's ``ARGUMENTS`` fortran subroutine

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.constituents
    pu,pf,G = pyTMD.constituents.arguments(MJD, constituents,
        deltat=DELTAT, corrections=CORRECTIONS)

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/constituents.py

.. autofunction:: pyTMD.constituents.arguments

.. autofunction:: pyTMD.constituents.minor_arguments

.. autofunction:: pyTMD.constituents.coefficients_table

.. autofunction:: pyTMD.constituents.doodson_number

.. autofunction:: pyTMD.constituents.nodal_modulation

.. autofunction:: pyTMD.constituents.group_modulation

.. autofunction:: pyTMD.constituents.frequency

.. autofunction:: pyTMD.constituents.aliasing_period

.. autofunction:: pyTMD.constituents._arguments_table

.. autofunction:: pyTMD.constituents._minor_table

.. autofunction:: pyTMD.constituents._constituent_parameters

.. autofunction:: pyTMD.constituents._love_numbers

.. autofunction:: pyTMD.constituents._complex_love_numbers

.. autofunction:: pyTMD.constituents._parse_tide_potential_table

.. autofunction:: pyTMD.constituents._to_constituent_id

.. autofunction:: pyTMD.constituents._to_doodson_number

.. autofunction:: pyTMD.constituents._to_extended_doodson

.. autofunction:: pyTMD.constituents._from_doodson_number

.. autofunction:: pyTMD.constituents._from_extended_doodson
