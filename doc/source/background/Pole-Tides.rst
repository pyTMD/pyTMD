Pole Tides
##########

The Earth's rotation axis is inclined at an angle of 23.5 degrees to the celestial pole, and rotates about it once every 26,000 years :cite:p:`Kantha:2000vo`.
Superimposed on this long-term :term:`Precession`, the rotation axis of the Earth shifts with respect to its mean pole location due to :term:`Nutation`, :term:`Chandler Wobble`, annual variations, and other processes :cite:p:`Wahr:1985gr,Desai:2002ev,Agnew:2015kw`.
Load and ocean pole tides are driven by these variations, the corresponding elastic response, and for the case of ocean pole tides the centripetal effects of :term:`Polar Motion` on the ocean :cite:p:`Desai:2002ev,Desai:2015jr`.
These variations are centimeter scale in both the vertical and horizontal, and should be taken into account when comparing observations over periods longer than two months.

Methods
-------

The formalism for estimating the pole tides within ``pyTMD`` is also based upon `IERS Conventions <https://iers-conventions.obspm.fr/>`_.
For ocean pole tides, :func:`pyTMD.predict.ocean_pole_tide` uses the equilibrium response model from :cite:t:`Desai:2002ev` as recommended by IERS Conventions :cite:p:`Petit:2010tp`.
``pyTMD`` uses the ``timescale`` library for reading the Earth Orientation Parameters (EOPs) necessary for computing load pole and ocean pole tide variations.
The currently accepted formalism for estimating the reference position of the Earth's figure axis at a given date is the `IERS 2018 secular pole model <https://iers-conventions.obspm.fr/chapter7.php>`_:

.. math::
    :label: 3.1
    :name: eq:3.1

    \bar{x}_s(t) &= 0.055 + 0.001677(t - 2000.0)\\
    \bar{y}_s(t) &= 0.3205 + 0.00346(t - 2000.0)


The time-dependent offsets from the reference rotation pole position, also known as wobble parameters (:math:`m_1` and :math:`m_2`), are then calculated using instantaneous values of the Earth Orientation Parameters :cite:p:`Petit:2010tp,Urban:2013vl`.


.. math::
    :label: 3.2
    :name: eq:3.2

    m_1(t) &= x_p(t) - \bar{x}_s(t)\\
    m_2(t) &= -(y_p(t) - \bar{y}_s(t))

.. plot:: ./background/polar-motion.py
    :show-source-link: False
    :caption: Polar motion estimates from the IERS
    :align: center
