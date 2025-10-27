Time
####

``pyTMD`` uses the ``timescale`` library to manage conversions between time scales.
A uniform time scale expresses dates as the count of time elapsed since a reference :term:`Epoch`.
The Julian Day (JD) is one such scale, and is the continuous count of days from noon on January 1, 4713 B.C (-4712-01-01T12:00:00).
The JD time system simplifies the calculation of the number of days between two epochs, and thus is convenient for astronomical purposes.
The Modified Julian Day (MJD) differs from the JD by 1) beginning at midnight and 2) reducing the total number of digits for modern periods.
The start of the MJD calendar is 1858-11-17T00:00:00, and a given MJD can be calculated from the JD by

.. math::
    :label: 3.1
    :name: eq:3.1

    MJD = JD - 2400000.5

Julian centuries (36525 days consisting of 86400 SI seconds) are used for modern-day celestial calculations, and are set relative to the J2000 epoch (2000-01-01T12:00:00).

.. math::
    :label: 3.2
    :name: eq:3.2

    T = \frac{JD - 2451545.0}{36525}

Standards
---------

``timescale`` and ``pyTMD`` are both reliant on the International Earth Rotation Service (IERS) for the schedule of leap seconds and estimates of delta times.
`Tables of leap seconds <https://github.com/pyTMD/timescale/blob/main/timescale/data/leap-seconds.list>`_ are used to convert from GPS, LORAN and TAI times.

- TAI time: International Atomic Time uses SI seconds and is computed as the weighted average of several hundred atomic clocks.
- GPS time: Atomic timing system for the Global Positioning System constellation of satellites monitored by the United States Naval Observatory (USNO). GPS time and UTC time were equal on January 6, 1980. TAI time is ahead of GPS time by 19 seconds.
- LORAN time: Atomic timing system for the Loran-C chain transmitter sites used in terrestrial radionavigation. LORAN time and UTC time were equal on January 1, 1958. TAI time is ahead of LORAN time by 10 seconds.
- UTC time: Coordinated Universal Time also uses SI seconds, but is `periodically adjusted <https://www.nist.gov/pml/time-and-frequency-division/leap-seconds-faqs>`_ to account for the difference between the definition of the second and the rotation of Earth. UTC is based off of atomic clocks and 1 day is exactly 86,400 seconds.

Solar time is based on the position of the sun in the sky and is used for civil timekeeping.
Universal Time (UT1) is effectively the mean solar time and is based on the true, irregular rotation of the Earth :cite:p:`Kaplan:2005kj`.
The Earth's rate of rotation is unpredictable and is measured through astronomical observations, predominantly from very long baseline interferometry (VLBI).
`Coordinated Universal Time (UTC) <https://crf.usno.navy.mil/ut1-utc>`_ is based on International Atomic Time (TAI) with leap seconds added to keep it within 0.9 seconds of UT1.

Dynamical Time
--------------

Dynamical time is the independent variable in the equations of motion for bodies in the solar system.
Terrestrial Time (TT) is a uniform, monotonically increasing time standard based on atomic clocks that is used for the accurate calculation of celestial mechanics, orbits and ephemerides.
It is the standard time reference for geocentric ephemerides, and is currently ahead of TAI time by 32.184 seconds :cite:p:`Urban:2013vl`.
Barycentric Dynamical Time (TDB) is also used to describe the motion of the planets, sun and moon, but is with respect to the solar system barycenter (SSB) :cite:p:`Kaplan:2005kj`.
TDB and TT are both *dynamical timescales*, with differences between them owing to relativistic effects.
Delta times (TT - UT1) can be added to estimates of Universal Time (UT1) to convert to Terrestrial Time (TT) :cite:p:`Meeus:1991vh`.


.. plot:: ./background/deltatime.py
    :show-source-link: False
    :caption: Delta times between Terrestrial Time (TT) and Universal Time (UT1)
    :align: center

Sidereal Time
-------------

While the Earth rotates about its axis, it also moves in its orbit around the Sun.
Sidereal time is based on the rotation of the Earth with respect to distant celestial objects.
This timescale is slightly different from Universal Time (UT1), which is calculated based on the Earth's rotation with respect to the sun.
As the Earth rotates 360 degrees in approximately 23 hours, 56 minutes and 4 seconds, the difference between a mean sidereal day and solar day is about 3 minutes and 56 seconds.
Because the rotation rate of Earth is variable, both sidereal time and Universal Time are irregular with respect to atomic time :cite:p:`Urban:2013vl`.
For every meridian, there is a local sidereal time, which is calculated with respect to an :term:`Equinox`. 

Greenwich Mean Sidereal Time (GMST) is the angle between the Greenwich meridian and the average position of the :term:`Vernal Equinox`.
GMST is calculated in ``pyTMD`` using the revised IAU 2000 precession model :cite:p:`Capitaine:2005hw,Urban:2013vl`.
Greenwich Apparent Sidereal Time (GAST) takes into account the apparent short term motions of the Vernal Equinox due to :term:`Nutation` using the "equation of the equinoxes" (:math:`E_e`).

.. math::
    :label: 3.3
    :name: eq:3.3

    GAST = GMST + E_e

The "equation of the equinoxes" describes the difference between the positions of the true (:math:`\Upsilon_T`) and mean (:math:`\Upsilon_M`) equinoxes of date, and is calculated using the following equations:

.. math::
    :label: 3.4
    :name: eq:3.4

    E_e &= \Upsilon_T - \Upsilon_M \\
    &= \Delta\psi\cos{\varepsilon} + \sum_k (C'_k \sin{A_k} + S'_k \cos{A_k})

where :math:`\Delta\psi` is the nutation in longitude, :math:`\varepsilon` is the obliquity of the ecliptic, and the series expansion of "complementary terms" describe the combined effects of precession and nutation :cite:p:`Kaplan:2005kj,Petit:2010tp,Urban:2013vl` .

Local Mean Sidereal Time (LMST) is similar to GMST, but takes into account longitudinal position in degrees East from the Greenwich meridian.

