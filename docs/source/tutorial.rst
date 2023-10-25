The tutorial
============
A case of typical usage is presented here.

For a more detailed use with real examples visit the :doc:`showcase <Showcase>` page.

Installation
------------
To use Models, first install it using pip:

.. code-block:: console

   $ pip install .....

Instance of the ``Model`` class
-------------------------------
We start by creating the Model object the following way:

.. code:: python

    a = models.Model(   # TODO
        n_dof,
        mass,
        stiffness,
        damping,
        boundaries="both"
    )

``n_dof`` argument
~~~~~~~~~~~~~~~~~~
This argument determines the number of degrees of freedom (number of masses).

``mass`` argument
~~~~~~~~~~~~~~~~~
This argument determines the mass values of masses (in kg). There are two options:

    * Specify mass with a single number (``float``, ``int``): all the masses have the same value.
    * Specify mass with array-like object of shape ``(n_dof,)``: masses have different values based on values in the array.

``stiffness`` argument
~~~~~~~~~~~~~~~~~~~~~~
This argument determines the stiffness values of springs (in N/m). There are two options:

    * Specify stiffness with a single number (``float``, ``int``): all the springs have the same value.
    * Specify stiffness with array-like object of shape ``(n_dof,)``, ``(n_dof+1,)`` or ``(n_dof-1,)`` based on ``boundaries`` argument 
      (see explanation with images below): springs have different values based on values in the array. 

``damping`` argument
~~~~~~~~~~~~~~~~~~~~
This argument determines the damping coefficient values of dampers (in N/m/s). There are two options:

    * Specify damping with a single number (``float``, ``int``): all the dampers have the same value.
    * Specify damping with array-like object of shape ``(n_dof,)``, ``(n_dof+1,)`` or ``(n_dof-1,)`` based on ``boundaries`` argument  
      (see explanation with images below): dampers have different values based on values in the array.

``boundaries`` argument
~~~~~~~~~~~~~~~~~~~~~~~
This argument determines the boundary conditions. There are 4 options available:

  * ``"both"`` -  the first mass and the last mass are connected to a rigid surface with a spring and a damper:
   
     .. image:: images/both_boundary.png
       :width: 650  
       
    .. note::
      If ``stiffness`` and/or ``damping`` arguments are passed as array-like objects, the shape of arrays for this boundary condition 
      must be ``(n_dof+1,)``.

  * ``"free"`` - the masses are free-free supported:

     .. image:: images/free_boundary.png
       :width: 520

    .. note::
      If ``stiffness`` and/or ``damping`` arguments are passed as array-like objects, the shape of arrays for this boundary condition 
      must be ``(n_dof-1,)``.

  * ``"left"`` - the first mass is connected to a rigid surface on the left with a spring and a damper:

     .. image:: images/left_boundary.png
       :width: 590 

    .. note::
      If ``stiffness`` and/or ``damping`` arguments are passed as array-like objects, the shape of arrays for this boundary condition 
      must be ``(n_dof,)``.

  * ``"right"`` - the last mass is connected to a rigid surface on the right with a spring and a damper:

     .. image:: images/right_boundary.png
       :width: 590  

    .. note::
      If ``stiffness`` and/or ``damping`` arguments are passed as array-like objects, the shape of arrays for this boundary condition 
      must be ``(n_dof,)``.

Getting system properties
-------------------------

Mass, stiffness and damping matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a result of the following methods we get the array (matrix) of dimensions ``(n_dof, n_dof)``:

  * Mass matrix (M):

    .. code:: python

      M = a.get_mass_matrix()

  * Stiffness matrix (K):

    .. code:: python

      K = a.get_stiffness_matrix()

  * Damping matrix (C):

    .. code:: python

      C = a.get_damping_matrix()

Eigen frequencies, eigen values, eigen vectors and viscous damping ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following methods are obtained via state-space model of the system:

  * Array of eigen frequencies (in Hz):

    .. code:: python

      eig_freq = a.get_eig_freq()

  * Tuple of eigen values and their conjugate pairs:

    .. code:: python

      eig_val = a.get_eig_val()

  * Tuple of eigen vectors and their conjugate pairs:

    .. code:: python

      eig_vec = a.get_eig_vec()

  * Viscous damping ratios of the system:

    .. code:: python

      d_ratios = a.get_damping_ratios()


Frequency response functions
----------------------------

To get the FRF matrix of the system we call the following method:

.. code:: python

  FRF_matrix = a.get_FRF_matrix(freq, frf_method="f", **kwargs)

``freq`` argument
~~~~~~~~~~~~~~~~~
Frequency array (in Hz) at which the FRF values are calculated.

``frf_method`` argument
~~~~~~~~~~~~~~~~~~~~~~~
Method used to calculate the FRF matrix:

  * ``"f"`` (default): frequency domain, via impedance inverse.
  * ``"s"``: state space domain, via state-space model parameters.

.. note::
  If the selected method for FRF calculation is the state space domain method (``frf_method="s"``), we can provide an optional keyword argument 
  ``n_modes`` to specify the number of modes used in FRF calculation via mode superposition method. If not specified, all the modes are used.

Impulse response functions
----------------------------

To get the impulse response (h) matrix of the system we call the following method:

.. code:: python

  h_matrix = a.get_h_matrix(freq, frf_method="f", return_t_axis=False, **kwargs)

``freq`` argument
~~~~~~~~~~~~~~~~~
Frequency array (in Hz) at which the FRF values are calculated.

``frf_method`` argument
~~~~~~~~~~~~~~~~~~~~~~~
Method used to calculate the FRF matrix, from which the impulse response functions matrix is caluclated via inverse FFT:

  * ``"f"`` (default): frequency domain, via impedance inverse.
  * ``"s"``: state space domain, via state-space model parameters.

.. note::
  If the selected method for FRF calculation is the state space domain method (``frf_method="s"``), we can provide an optional keyword argument 
  ``n_modes`` to specify the number of modes used in FRF calculation via mode superposition method. If not specified, all the modes are used.

``return_t_axis`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
If ``True``, returns the time axis of impulse response functions.

.. note::
  If ``True`` the method returns a tuple with two elements: ``(h_matrix, time_series)``.


Obtaining systems response
--------------------------

To obtain the systems response to known excitation we can use the ``get_response()`` method:

.. code:: python

  response = a.get_response(
    exc_dof,
    exc,
    sampling_rate,
    resp_dof,
    domain="f",
    frf_method="f",
    return_matrix=False,
    return_t_axis=False,
    return_f_axis=False,
    **kwargs
  )

``exc_dof`` argument
~~~~~~~~~~~~~~~~~~~~
The degrees of freedom array where excitation is applied.

``exc`` argument
~~~~~~~~~~~~~~~~
The excitation time series array of dimensions ``(len(exc_dof), time_points)``.

``sampling_rate`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
Sampling rate of excitation signals.

``resp_dof`` argument
~~~~~~~~~~~~~~~~~~~~~
Degrees of freedom where we want the response to be calculated. If ``None`` (by default) the responses at all degrees of 
freedom are calculated.

``domain`` argument
~~~~~~~~~~~~~~~~~~~~~~~
Method used to calculate the response (frequency or time domain):

  * ``"f"`` (default): multiplication in the frequency domain.
  * ``"t"``: convolution in time domain.

.. note::
  If we choose the time domain response calculation (``domain="t"``), we can also use the two additional keyword arguments ``mode`` and 
  ``method``, which control the convolution calculation. See :doc:`code documentation <code_documentation>` for further info.

``frf_method`` argument
~~~~~~~~~~~~~~~~~~~~~~~
Method used to calculate the FRF matrix:

  * ``"f"`` (default): frequency domain, via impedance inverse.
  * ``"s"``: state space domain, via state-space model parameters.

.. note::
  If the selected method for FRF calculation is the state space domain method (``frf_method="s"``), we can provide an optional keyword argument 
  ``n_modes`` to specify the number of modes used in FRF calculation via mode superposition method. If not specified, all the modes are used.

``return_matrix`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
If ``True``, returns the matrix that was used to calculate the reponse - FRF matrix (``domain="f"``) or 
impulse response matrix (``domain="t"``).

``return_t_axis`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
If ``True``, returns the time axis of response and excitation signals.

``return_f_axis`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
If ``True``, returns the frequnecy axis of the FRF matrix.

.. note::
  If any of the ``return_matrix``, ``return_t_axis``, ``return_f_axis`` is ``True``, the result of the method is a tuple with all 
  the requested returned items.