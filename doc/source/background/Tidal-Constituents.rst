:orphan:

========================
Major Tidal Constituents
========================

.. _tab-constituents:

.. csv-table:: Tidal Constituents
   :file: ../_assets/constituents.csv
   :header-rows: 1
   :width: 100%


From :cite:t:`Doodson:1921kt,Cartwright:1971iz,Cartwright:1973em`

Compound Constituents
=====================

Two or more constituents can interact harmonically in shallow-water to form overtides or compound constituents. 
The properties of these compound constituents can be derived from the properties of their parent constituents.

.. _constituent-notation:

Constituent Notations
=====================

Every tidal constituent corresponds to a specific combination of astronomical cycles [see :ref:`astronomical-arguments`], and several notation systems exist for encoding that combination compactly.

``pyTMD`` supports three (interchangeable) formalisms:

- **Cartwright numbers**: stores the multipliers as signed integers in an ordered list :cite:p:`Cartwright:1971iz`.
- **Doodson numbers**: compact decimal representation designed for human-readable identification of constituents :cite:p:`Doodson:1921kt`.
- **Extended Doodson numbers (XDO)**: compact and human-readable representation used by the UK Hydrographic Office (UKHO)

Cartwright Numbers
------------------

Cartwright numbers are an ordered list of signed integers for the multipliers of the astronomical arguments [see :ref:`Equation 1.2 <eq:1.2>`]:

.. math::
    \text{Cartwright numbers} = [d_1, d_2, d_3, d_4, d_5, d_6]

- :math:`d_1`: multiples of the :ref:`spherical harmonic <spherical-harmonics>` dependence (:math:`\tau`)
- :math:`d_2`: multiples of the mean longitude of the Moon (:math:`S`)
- :math:`d_3`: multiples of the mean longitude of the Sun (:math:`H`)
- :math:`d_4`: multiples of the mean longitude of the perigee of the Moon (:math:`P`)
- :math:`d_5`: multiples of the mean longitude of the node of the Moon (:math:`N`)
- :math:`d_6`: multiples of the mean longitude of the perigee of the Sun (:math:`Ps`)

.. _doodson-numbers:

Doodson Numbers
---------------

Doodson numbers are an unsigned notion where the second through sixth multipliers (:math:`d_{2-6}`) are encoded by adding 5 (the first digit :math:`d_1` is not offset).
This offset maps the range of multipliers :math:`[-5, +4]` into :math:`[0, 9]`.
This encoding can be extended by mapping :math:`+5` to :math:`\mathrm{X}`, :math:`+6` to :math:`\mathrm{E}`, and :math:`+7` to :math:`\mathrm{T}` (which would map to :math:`10`, :math:`11` and :math:`12` in the standard notation).

.. math::
    \text{Doodson number} = \{d_1\}\{d_2+5\}\{d_3+5\}.\{d_4+5\}\{d_5+5\}\{d_6+5\}

Extended Doodson Numbers
------------------------

The UKHO Extended Doodson Number (XDO) system was designed to address two limitations of the standard Doodson notation:

1. **Range**: the +5 digit offset only covers multipliers in :math:`[-5, +4]`. Tidal catalogs larger than the original from :cite:t:`Doodson:1921kt` can have constituents with multipliers outside that range
2. **Disambiguation**: the format carries a seventh character encoding the index :math:`k` which resolves ambiguities when constituents share the same Doodson number

The XDO system maps :math:`0` to :math:`\mathrm{Z}`, the range :math:`[1,15]` to :math:`[\mathrm{A},\mathrm{P}]`, and the range :math:`[-8,-1]` to :math:`[\mathrm{R},\mathrm{Y}]`.

Tidal Classifications
---------------------

.. list-table:: Doodson Number Classification
    :header-rows: 1
    :align: center

    * - Classification
      - Description
    * - :term:`Species`
      -  shared :math:`d_1` 
    * - Group
      -  shared :math:`d_1` and :math:`d_2`
    * - Subgroup
      -  shared :math:`d_1`, :math:`d_2` and :math:`d_3`
