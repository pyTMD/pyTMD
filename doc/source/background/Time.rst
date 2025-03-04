Time
####

Time can be measured using a variety of different scales.
``pyTMD`` uses the ``timescale`` library to manage the conversions between these different time scales.
A uniform time scales expresses dates as the count of time elapsed since a reference epoch.
The Julian Day (JD) is the continuous count of days starting at noon on January 1, 4713 B.C (-4712-01-01T12:00:00).
It is a time system that allows a simple computation for the number of days between two epochs.
The Modified Julian Day (MJD) differs from the Julian Day by reducing the number of digits for modern periods, and by beginning at midnight.
The start of the MJD calendar is 1858-11-17T00:00:00, and a given MJD can be calculated from the Julian Day by

.. math::
    :label: 3.1
    :name: eq:3.1

    MJD = JD - 2400000.5

Julian centuries (36525 days) are used for celestial calculations, and are fixed relative to the J2000 epoch (2000-01-01T12:00:00).

.. math::
    :label: 3.2
    :name: eq:3.2

    T = \frac{JD - 2451545.0}{36525}

Standards
---------

``timescale`` and ``pyTMD`` are reliant on the International Earth Rotation Service (IERS) for the schedule of leap seconds and estimates of delta times.
`Tables of leap seconds <https://github.com/pyTMD/timescale/blob/main/timescale/data/leap-seconds.list>`_ are used to convert between GPS, LORAN and TAI times.

- TAI time: International Atomic Time uses SI seconds and is computed as the weighted average of several hundred atomic clocks.
- UTC time: Coordinated Universal Time also uses SI seconds, but is `periodically adjusted <https://www.nist.gov/pml/time-and-frequency-division/leap-seconds-faqs>`_ to account for the difference between the definition of the second and the rotation of Earth. UTC is based off of atomic clocks and 1 day is exactly 86,400 seconds.
- GPS time: Atomic timing system for the Global Positioning System constellation of satellites monitored by the United States Naval Observatory (USNO). GPS time and UTC time were equal on January 6, 1980. TAI time is ahead of GPS time by 19 seconds.
- LORAN time: Atomic timing system for the Loran-C chain transmitter sites used in terrestrial radionavigation. LORAN time and UTC time were equal on January 1, 1958. TAI time is ahead of LORAN time by 10 seconds.

`Tables of delta times <https://github.com/pyTMD/timescale/blob/main/timescale/data/merged_deltat.data>`_ are used to convert between dynamic terrestrial (TT) and universal (UT1) times :cite:p:`Meeus:1991vh`.
Universal Time (UT1) is effectively the mean solar time and is based on the true, irregular rotation of the Earth :cite:p:`Kaplan:2005kj`.
The Earth's rate of rotation is unpredictable and is measured through astronomical observations, predominantly from very long baseline interferometry (VLBI).
`Coordinated Universal Time (UTC) <https://crf.usno.navy.mil/ut1-utc>`_ is based on International Atomic Time (TAI) with leap seconds added to keep it within 0.9 seconds of UT1.

Terrestrial Time (TT) is a uniform, monotonically increasing time standard based on atomic clocks that is used for the accurate calculation of celestial mechanics, orbits and ephemerides.
Barycentric Dynamical Time (TDB) is also used to describe the motion of the planets, sun and moon, but is with respect to the solar system barycenter (SSB) :cite:p:`Kaplan:2005kj`.
TDB and TT are both dynamic timescales, with differences owing to relativistic effects.
Delta times (TT - UT1) can be added to estimates of Universal Time (UT1) to convert to Dynamic Time (TT).

.. plot:: ./background/deltatime.py
    :show-source-link: False
    :caption: Delta times between Dynamic Time (TT) and Universal Time (UT1)
    :align: center

Sidereal Time
-------------

Sidereal time is based on the rotation of the Earth with respect to the fixed stars in the sky.
There is a slight difference between a sidereal day and a solar day due to the Earth's rotation about the Sun.
As the Earth rotates 360 degrees in approximately 23 hours, 56 minutes and 4 seconds, the difference between a mean sidereal day and solar day is about 3 minutes and 56 seconds.
