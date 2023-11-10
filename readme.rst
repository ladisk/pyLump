models ?????
======

Multi Degree of Freedom Models (mass-spring-damper).
----------------------------------------------------
For more information check out the showcase examples and see documentation_. SPREMENI LINK

Basic ``models???`` usage:
--------------------------

Make an instance of the ``Model`` class:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    a = models.Model(   # TODO
        n_dof,
        mass,
        stiffness,
        damping,
        boundaries="both"
    )

Getting system properties:
~~~~~~~~~~~~~~~~~~~~~~~~~~
There are several methods available for different system properties:

.. code:: python

    M = a.get_mass_matrix()
    K = a.get_stiffness_matrix()
    C = a.get_damping_matrix()
    eig_freq = a.get_eig_freq()
    eig_val = a.get_eig_val()
    eig_vec = a.get_eig_vec()
    d_ratios = a.get_damping_ratios()

Obtaining frequency response functions and impulse response functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To obtain the FRF (frequency response function) matrix and the IRF (impulse reponse function) matrix we use the following methods:

.. code:: python

    FRF_matrix = a.get_FRF_matrix(freq)
    IRF_matrix = a.get_IRF_matrix(freq)

Calculating response:
~~~~~~~~~~~~~~~~~~~~~
We can calculate the systems response based on known excitation the following way:

.. code:: python

    response = a.get_response(
        exc_dof,
        exc,
        sampling_rate,
        resp_dof
    )

.. _documentation: https://pyfrf.readthedocs.io/en/latest/