.. _gravity-tides:

Gravity Tides
=============

The spatial gradient of the tide-generating potential is known as the tide-generating force (TGF), and the negative of the radial tide-generating force is used to compute the gravity tide :cite:p:`Hartmann:1995va,Tamura:1982wx`.
Gravity tides, as measured by a (superconducting) gravimeter at Earth's surface, are related to but distinct from the solid Earth tide displacements described in :ref:`solid-earth-tides`.
They can be computed as the sum of several effects :cite:p:`Hartmann:1995va,Tamura:1987tp,Merriam:1992kg`:

1. **Direct**: the tide-generating force between the Earth and the planetary body. This is the entire signal for a rigid, ocean-free and non-deforming Earth.
2. **Displacement**: height changes (radial displacements) to/from the Earth's center of mass
3. **Deformation**: local changes in the Earth's density due to the tidally induced deformation
4. **Self-attraction and loading effects**: redistribution of ocean mass, which generates a tidal potential and induces deformation of the Earth's crust
5. **Non-Equilibrium effects**: ocean's non-instantaneous and non-uniform response to the tide-generating potential (estimated from ocean tide models)
6. **Centripetal effects**: centripetal acceleration from the Earth's variable rotation (modulated by :ref:`polar motion <pole-tides>` and ocean loading)

The first three effects are often combined to calculate a major component of the gravity tide based on the tide-generating force and the Earth's elastic response :cite:p:`Hartmann:1995va,Hartmann:1995jp`.
As with :ref:`solid Earth tides <solid-earth-tides>`, the :ref:`love-and-shida-numbers` are frequency-dependent (particularly in the diurnal band due to :term:`Free Core Nutation` resonance) :cite:p:`Wahr:1979vx,Wahr:1981if`.