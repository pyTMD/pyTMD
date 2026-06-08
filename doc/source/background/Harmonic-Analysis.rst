.. _harmonic-analysis:

Harmonic Analysis
==================

Using the :ref:`harmonic method <harmonic-method>` for tidal prediction solves the *forward problem*.
Given the harmonic amplitude :math:`A_k`  and phase :math:`\theta_k` for constituent :math:`k`, compute the tidal elevation or current at a time and place [see :ref:`Equation 1.1 <eq:1.1>` in :ref:`ocean-load-tides`].
Additional terms such as the equilibrium phase :math:`G_k(t)` [see :ref:`Equation 1.2 <eq:1.2>`], and :term:`nodal corrections <Nodal Corrections>` :math:`f_k(t)` and :math:`u_k(t)` are calculated using the *prediction* times.

Harmonic analysis solves the complementary *inverse problem*.
Given a time series of observed tidal elevations or currents, estimate the amplitude and phase of a set of tidal constituents.
In this problem, the additional terms of the equilibrium phase :math:`G_k(t)` and nodal corrections :math:`f_k(t)` and :math:`u_k(t)` are calculated from the *observation* times.

Design Matrix
-------------

The system of equations for solving for :math:`K` number of constituents with :math:`N` number of observations at times :math:`t_1, \ldots, t_N` is:

.. math::
    :label: 5.1
    :name: eq:5.1

    \mathbf{h} = \mathbf{M}^\mathsf{T}\mathbf{x} + \boldsymbol{\sigma}

where:

- :math:`\mathbf{h}` is the observation vector :math:`[h(t_1), \ldots, h(t_N)]`
- :math:`\mathbf{M}^\mathsf{T}` is the transpose of the design matrix in :ref:`Equation 5.2 <eq:5.2>`
- :math:`\mathbf{x}` is the parameter vector :math:`[z_0, \ldots, c_\alpha, s_\alpha, c_\beta, s_\beta, \ldots, c_K, s_K]` with terms for the datum offset and the cosine and sine coefficients for each constituent
- and :math:`\boldsymbol{\sigma}` is the residual vector (non-tidal signal plus measurement noise)

.. math::
    :label: 5.2
    :name: eq:5.2

    \mathbf{M} =
      \begin{bmatrix}
         1 & \ldots & 1 \\
         \vdots & \ddots & \vdots \\
         f_\alpha(t_1)\cos(G_\alpha(t_1) + u_\alpha(t_1)) & \ldots & f_\alpha(t_N)\cos(G_\alpha(t_N) + u_\alpha(t_N)) \\
         -f_\alpha(t_1)\sin(G_\alpha(t_1) + u_\alpha(t_1)) & \ldots & -f_\alpha(t_N)\sin(G_\alpha(t_N) + u_\alpha(t_N)) \\
         f_\beta(t_1)\cos(G_\beta(t_1) + u_\beta(t_1)) & \ldots & f_\beta(t_N)\cos(G_\beta(t_N) + u_\beta(t_N)) \\
         -f_\beta(t_1)\sin(G_\beta(t_1) + u_\beta(t_1)) & \ldots & -f_\beta(t_N)\sin(G_\beta(t_N) + u_\beta(t_N)) \\
         \vdots & \ddots & \vdots \\
         f_K(t_1)\cos(G_K(t_1) + u_K(t_1)) & \ldots & f_K(t_N)\cos(G_K(t_N) + u_K(t_N)) \\
         -f_K(t_1)\sin(G_K(t_1) + u_K(t_1)) & \ldots & -f_K(t_N)\sin(G_K(t_N) + u_K(t_N))
      \end{bmatrix}

Each pair of rows in the design matrix that contain harmonic terms, :math:`f_k(t)\cos(G_k(t) + u_k(t))` and :math:`-f_k(t)\sin(G_k(t) + u_k(t))`, encode the time-varying phase of a single tidal constituent :math:`k`.

The first row of the design matrix (consisting entirely of ones) allows for the encoding of a constant term.
This will solve for the average elevation or current, and is by default included in the model design matrix.
Real observations include non-tidal signals, such as from waves, storm surges, sea level changes, and instrument drifts.
Augmenting the design matrix to include more polynomial terms will simultaneously fit and remove some of the long-term non-tidal signals, such as from secular instrument drift.
If the time series is long enough to statistically resolve them, including higher-order polynomial terms can also solve for trends or accelerations in the data.

Least-Squares Solution
----------------------

The least-squares solution of this system of equations minimizes the residuals of

.. math::
   :label: 5.3
   :name: eq:5.3

   (\mathbf{h} - \mathbf{M}^\mathsf{T}\mathbf{x})^\mathsf{T}(\mathbf{h} - \mathbf{M}^\mathsf{T}\mathbf{x})

The solution vector :math:`\mathbf{\hat{x}}` contains the best estimates of the datum offset along with the cosine and sine terms.
The harmonic constants for each constituent :math:`A_k` and :math:`\theta_k` are calculated from these cosine and sine coefficients.
Note that the sine terms are negative in the design matrix, which follows the convention used within ``pyTMD``, and so the function that calculates the phase also uses negative values.

.. math::
    :label: 5.4
    :name: eq:5.4

    \hat{A}_k &= \sqrt{\hat{c}_k^2 + \hat{s}_k^2} \\
    \hat{\theta}_k &= \mathrm{arctan2}(-\hat{s}_k, \hat{c}_k)

``pyTMD`` includes several solver options in order to handle rank-deficient systems or include bounds on the solution variables.

.. list-table:: Solver Methods
    :header-rows: 1
    :align: center

    * - Solver
      - Method
      - Algorithm
    * - ``'lstsq'``
      - Least squares
      - ``numpy.linalg.lstsq``
    * - ``'gelsy'``
      - Complete orthogonal factorization
      - ``scipy.linalg.lstsq``
    * - ``'gelss'``
      - Singular value decomposition (SVD)
      - ``scipy.linalg.lstsq``
    * - ``'gelsd'``
      - SVD with divide-and-conquer method
      - ``scipy.linalg.lstsq``
    * - ``'bvls'``
      - Bounded-variable least squares
      - ``scipy.optimize.lsq_linear``


Nodal Corrections
-----------------

The nodal corrections :math:`f_k(t)` and :math:`u_k(t)` are evaluated at each observation time and included directly into the design matrix :ref:`Equation 5.2 <eq:5.2>`.
The amplitude :math:`\hat{A}_k`  and phase :math:`\hat{\theta}_k` estimated from the least-squares solution represent those at a "standard" epoch, and, in a perfect solution, free of the 18.6-year nodal modulation [see :ref:`nodal-corrections` in :ref:`ocean-load-tides`].
The fitted harmonic constants should be able to be directly compared with outputs from tide models or tide tables.

Minor Constituent Inference
---------------------------

Assuming that the :term:`tidal admittance <Admittance>` varies smoothly with frequency, the amplitudes and phases of minor constituents can be inferred from the major constituents :cite:p:`Foreman:1989dt,Munk:1966go,Zetler:1975uv`.
This can be the case when the record of observations is too short to independently resolve constituents that are close in frequency :cite:p:`Foreman:2009bg`.
Including inference as part of the fitting process can avoid contaminating the regression of major constituents, and can improve their overall estimation :cite:p:`Foreman:1989dt,Foreman:2009bg`.

In ``pyTMD``, this inference is performed *after* the primary regression using a bootstrap iteration procedure.
After each fit, the time series of the inferred minor constituents is reconstructed and then removed from the observation time series.
The process is repeated and the observation vector :math:`\mathbf{h}` is continually modified until the specified number of iterations has been reached.
