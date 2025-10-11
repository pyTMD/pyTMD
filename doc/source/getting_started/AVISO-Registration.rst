:orphan:

.. _aviso-registration:

Registering with AVISO
----------------------

The FES tide models are available through the AVISO+ data portal.
Registering for an account is free, and provides access to a variety of oceanographic datasets.

.. note::
    If you have previously registered with AVISO+, you can use the same account to access the FES tide models.
    Historically, each product had to be requested separately, but now all AVISO+ products are made available after registration.

1. Fill out the `AVISO+ registration form <https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html>`_.
2. If you want to receive communications and updates, select the checkbox for "FES (Finite Element Solution - Oceanic Tides Heights)" under the "Auxiliary Products" section.
3. You will receive two emails: one confirming your registration, and another with your login details. Your username is typically your email address.
4. This gains you access to the `My AVISO+ data portal <https://www.aviso.altimetry.fr/en/my-aviso-plus.html>`_ and FTP servers.

After registering, you can use :func:`pyTMD.datasets.fetch_aviso_fes` to download the FES tide models.
