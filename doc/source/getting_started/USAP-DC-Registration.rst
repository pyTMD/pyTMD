:orphan:

.. _usap-dc-registration:

========================
Registering with USAP-DC
========================

The Circum-Antarctic Tidal Simulations (CATS) are available through the USAP Data Center (USAP-DC).
The models can be programmatically accessed through their `API <http://usap-dc.org/api>` after registering for a token.

.. note::
    The models can also be manually downloaded from the USAP-DC website after passing a reCAPTCHA test

1. Email `info@usap-dc.org <mailto:info@usap-dc.org>`_ to request an access token.

After registering and getting an access token, you can use :py:func:`pyTMD.datasets.fetch_usap_cats` to download the CATS models.
