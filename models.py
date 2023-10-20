"""

Module for multi degree of freedom lumped mass (mass-spring-damper) models.

Classes:
    class Model: A lumped mass model.
"""


import numpy as np
import scipy



__version__ = "0.1"  # TODO ???
_BOUNDARIES = ["free", "both", "left", "right"]
_FRF_METHODS = ["f", "s"]
_MODES = ["full", "valid", "same"]
_METHODS = ["auto", "direct", "fft"]
_DOMAINS = ["f", "t"]



class Model:
    """
    Multi Degree of Freedom Models (mass-spring-damper).
    """

    def __init__(self, n_dof, mass, stiffness, damping, boundaries="both"):
        """
        Initiates the data class:

        :param n_dof: Number of degrees of freedom - number of masses.
        :type n_dof: int
        :param mass: Weight of connected masses in kg.
            int, float - all masses have the same weight.
            array, list, tuple of length n_dof - specify different weight for each mass in order.
        :type mass: int, float, array, list, tuple
        :param stiffness: Stiffness of springs that are connecting masses.
            int, float - all springs have the same stiffness.
            array, list, tuple of length n_dof or n_dof+1, depending on boundary condition - 
            specify different stiffness for each spring in order.
        :type stiffness: int, float, array, list, tuple
        :param damping: Damping coefficient of dampers that are connecting masses.
            int, float - all dampers have the same daming coefficient.
            array, list, tuple of length n_dof or n_dof+1, depending on boundary condition - 
            specify different damping coefficients for each damper in order.
        :type damping: int, float, array, list, tuple
        :param boundaries: Boundary conditions: ``"free"``, ``"both"``, ``"left"``, ``"right"`` - which side the masses 
            are connected to rigid surface.
        :type boundaries: str
        """
        # dof:
        if isinstance(n_dof, int):
            self.n_dof = n_dof
        else:
            raise Exception("Number of degrees of freedom should be of type int")
        
        # boundary type:
        if not (boundaries in _BOUNDARIES):
            raise Exception("Wrong boundaries type given. Can be one of %s" % _BOUNDARIES)
        self.boundaries = boundaries

        # m:
        if isinstance(mass, float) or isinstance(mass, int):
            self.m = np.empty(self.n_dof)
            self.m.fill(mass)
        else:
            self.m = mass
        # k:
        if isinstance(stiffness, float) or isinstance(stiffness, int):
            if (self.boundaries == "free"):
                self.k = np.empty(self.n_dof - 1)
            elif (self.boundaries == "left") or (self.boundaries == "right"):
                self.k = np.empty(self.n_dof)
            elif (self.boundaries == "both"):
                self.k = np.empty(self.n_dof + 1)
            self.k.fill(stiffness)
        else:
            self.k = stiffness
        # c:
        if isinstance(damping, float) or isinstance(damping, int):
            if (self.boundaries == "free"):
                self.c = np.empty(self.n_dof - 1)
            elif (self.boundaries == "left") or (self.boundaries == "right"):
                self.c = np.empty(self.n_dof)
            elif (self.boundaries == "both"):
                self.c = np.empty(self.n_dof + 1)
            self.c.fill(damping)
        else:
            self.c = damping

        # check dimensions and dofs:
        if (self.n_dof != len(self.m)):
            raise Exception("Length of mass array should be equal to the number of degrees of freedom.")
        if (self.boundaries == "free"):
            if ((self.n_dof-1) != len(self.k)) or ((self.n_dof-1) != len(self.c)):
                raise Exception("Length of stiffness and damping array for free-free supported systems should be equal to "\
                                "the number of degrees of freedom - 1.")
        elif (self.boundaries == "left") or (self.boundaries == "right"):
            if (self.n_dof != len(self.k)) or (self.n_dof != len(self.c)):
                raise Exception("Length of stiffness and damping array for left and right supported systems should be equal to "\
                                "the number of degrees of freedom.")
        elif (self.boundaries == "both"):
            if ((self.n_dof+1) != len(self.k)) or ((self.n_dof+1) != len(self.c)):
                raise Exception("Length of stiffness and damping array for both side supported systems should be equal to "\
                                "the number of degrees of freedom + 1.")

        # initiate mass, stiffness and damping matrices:
        self._ini_matrices()

        # eig_val, eig_vec, eig_freq, damping, status:
        self.eig_val = np.array([])
        self.eig_vec = np.array([])
        self.eig_freq = np.array([])
        self.v_damping = np.array([])
        self.eig_calculated = False


    def _ini_matrices(self):
        """
        Initiates mass (M), stiffness (K) and damping (C) matrix of the system.
        """

        # mass:
        self.M = np.zeros((self.n_dof, self.n_dof))
        np.fill_diagonal(self.M, self.m)

        # stiffness:
        self.K = self._fill_matrix(value_array=self.k)

        # damping:
        self.C = self._fill_matrix(value_array=self.c)


    def _fill_matrix(self, value_array):
        """
        Fills stiffness and damping matrix based on boundary conditions.

        :param value_array: Array of values to fill the matrix with, based on boundary conditions.
        :type value_array: array, list
        """

        matrix = np.zeros((self.n_dof, self.n_dof))

        if self.boundaries == "free":
            matrix[0,0] = value_array[0]
            for i in range(self.n_dof-2):
                matrix[i+1,i+1] = value_array[i]+value_array[i+1]
                matrix[i,i+1] = -value_array[i]
                matrix[i+1,i] = -value_array[i]
            matrix[self.n_dof-1,self.n_dof-1] = value_array[self.n_dof-2]
            matrix[self.n_dof-2,self.n_dof-1] = -value_array[self.n_dof-2]
            matrix[self.n_dof-1,self.n_dof-2] = -value_array[self.n_dof-2]

        elif self.boundaries == "left":
            for i in range(self.n_dof-1):
                matrix[i,i] = value_array[i]+value_array[i+1]
                matrix[i,i+1] = -value_array[i+1]
                matrix[i+1,i] = -value_array[i+1]
            matrix[self.n_dof-1,self.n_dof-1] = value_array[self.n_dof-1]

        elif self.boundaries == "right":
            matrix[0,0] = value_array[0]
            for i in range(self.n_dof-1):
                matrix[i+1,i+1] = value_array[i]+value_array[i+1]
                matrix[i,i+1] = -value_array[i]
                matrix[i+1,i] = -value_array[i]

        elif self.boundaries == "both":
            for i in range(self.n_dof-1):
                matrix[i,i] = value_array[i]+value_array[i+1]
                matrix[i,i+1] = -value_array[i+1]
                matrix[i+1,i] = -value_array[i+1]
            matrix[self.n_dof-1,self.n_dof-1] = value_array[self.n_dof-1]+value_array[self.n_dof]

        return matrix


    def _ini_eig_val_vec(self):
        """
        Initiate eigen values, eigen vectors, eigen frequencies and viscous damping 
        ratios based on system properties (M, K and C matrices).
        """
        # State-space:
        A = np.zeros((2*self.n_dof, 2*self.n_dof))
        B = np.zeros((2*self.n_dof, 2*self.n_dof))
        A[:self.n_dof, :self.n_dof] = self.C
        A[:self.n_dof, -self.n_dof:] = self.M
        A[-self.n_dof:, :self.n_dof] = self.M
        B[:self.n_dof, :self.n_dof] = self.K
        B[-self.n_dof:, -self.n_dof:] = -self.M

        # Modal analysis:
        AB_eig = np.linalg.inv(A) @ B
        val, vec = scipy.linalg.eig(AB_eig)

        roots = -val[1::2][::-1]
        roots_conj = -val[::2][::-1]
        vectors = vec[:self.n_dof, ::-2]     # non-normalized
        vectors_conj = vec[:self.n_dof, -2::-2] # non-normalized

        PHI = np.zeros_like(vec)
        PHI[:self.n_dof, :self.n_dof] = vectors
        PHI[-self.n_dof:, :self.n_dof] = roots * vectors
        PHI[:self.n_dof, -self.n_dof:] = vectors_conj
        PHI[-self.n_dof:, -self.n_dof:] = roots_conj * vectors_conj
        a_r = np.diagonal(PHI.T @ A @ PHI)
        _a_r = a_r[:self.n_dof]
        _a_r_conj = a_r[self.n_dof:]

        # A-normalization
        vectors_A = vectors / np.sqrt(_a_r) # A-normalized
        vectors_A_conj = vectors_conj / np.sqrt(_a_r_conj) # A-normalize

         # Order returned data by system roots amplitude
        order = np.argsort(np.abs(roots))
        self.eig_val = (roots[order], roots_conj[order])
        self.eig_vec = (vectors_A[:, order], vectors_A_conj[:, order])

        # Eigen frequencies and viscous damping ratios:
        w_r = np.abs(roots[order])
        self.v_damping = -np.real(roots[order]) / w_r
        self.eig_freq = w_r / 2 / np.pi

        self.eig_calculated = True
        


    def get_mass_matrix(self):
        """
        Get mass (M) matrix of the system.

        :return: Mass matrix array of shape ``(n_dof, n_dof)``.
        :rtype: ndarray
        """
        return self.M


    def get_stiffness_matrix(self):
        """
        Get stiffness (K) matrix of the system.

        :return: Stiffness matrix array of shape ``(n_dof, n_dof)``.
        :rtype: ndarray
        """
        return self.K


    def get_damping_matrix(self):
        """
        Get damping (C) matrix of the system.

        :return: Damping matrix array of shape ``(n_dof, n_dof)``.
        :rtype: ndarray
        """
        return self.C


    def get_eig_val(self):
        """
        Get state-space model eigen values and their conjugate pairs.

        :return: Tuple of eigen values array and their conjugate pairs array.
        :rtype: tuple(ndarray)
        """

        if self.eig_calculated:
            return self.eig_val
        else:
            self._ini_eig_val_vec()
            return self.eig_val


    def get_eig_vec(self):
        """
        Get state-space model mass-normalized eigen vectors and their conjugate pairs.

        :return: Tuple of mass-normalized eigen vectors array and their conjugate pairs array.
        :rtype: tuple(ndarray)
        """
        
        if self.eig_calculated:
            return self.eig_vec
        else:
            self._ini_eig_val_vec()
            return self.eig_vec
        

    def get_damping_ratios(self):
        """
        Get viscous damping ratios of the system.

        :return: Array of shape ``(n_dof,)`` of viscous damping ratios of the system.
        :rtype: ndarray
        """
        
        if self.eig_calculated:
            return self.v_damping
        else:
            self._ini_eig_val_vec()
            return self.v_damping
    
    
    def get_eig_freq(self):
        """
        Get eigen frequencies of the system (in Hz).

        :return: Array of shape ``(n_dof,)`` of eigen frequencies (in Hz).
        :rtype: ndarray
        """

        if self.eig_freq.size == 0:
            eig_val = scipy.linalg.eigh(self.K, self.M, eigvals_only=True)
            eig_val.sort()
            eig_omega = np.sqrt(np.abs(np.real(eig_val)))
            self.eig_freq = eig_omega / (2 * np.pi)

        return self.eig_freq
    

    def get_FRF_matrix(self, freq, frf_method="f"):
        """
        Get FRF matrix of the system.

        :param freq: Frequency array (in Hz) at which the FRF values are calculated.
        :param frf_method: Method of calculating the FRF matrix:
            ``"f"`` - frequency domain, based on impedance inverse or 
            ``"s"`` - state space domain, based on state-space model parameters
        :type frf_method: str
        :type freq: ndarray
        :return: FRF matrix array of shape of shape ``(n_dof, n_dof, frequency_series)``.
        :rtype: ndarray
        """

        if not (frf_method in _FRF_METHODS):
            raise Exception("Wrong frf_method type given. Can be one of %s" % _FRF_METHODS)
        
        if isinstance(freq, list):
            freq = np.array(freq)

        omega = 2 * np.pi * freq

        FRF_matrix = np.zeros([self.n_dof, self.n_dof, len(omega)], dtype="complex128")

        if frf_method == "f":
            for i, omega_i in enumerate(omega):
                FRF_matrix[:,:,i] = scipy.linalg.inv(self.K - omega_i**2 * self.M + 1j*omega_i*self.C)

        elif frf_method == "s":
            if not self.eig_calculated:
                 self._ini_eig_val_vec()

            vec = self.eig_vec[0]
            vec_conj = self.eig_vec[1]
            val = self.eig_val[0]
            val_conj = self.eig_val[1]

            for i in range(self.n_dof):
                for j in range(self.n_dof):
                    FRF_ij = (vec[i]*vec[j])[:, None] / (1j*omega[None, :] - val[:, None])
                    FRF_ij += (vec_conj[i]*vec_conj[j])[:, None] / (1j*omega[None, :] - val_conj[:, None])
                    FRF_matrix[i, j] = np.sum(FRF_ij, axis=0)

        return FRF_matrix
    

    def get_h_matrix(self, freq, frf_method="f", return_t_axis=False):
        """
        Get h (impulse response function) matrix of the system.

        :param freq: frequency array at which the FRF values are calculated.
        :type freq: ndarray
        :param frf_method: Method of calculating the FRF matrix:
            ``"f"`` - frequency domain, based on impedance inverse or 
            ``"s"`` - state space domain, based on state-space model parameters
        :type frf_method: str
        :param return_t_axis: True if you want to return the time axis.
        :type return_t_axis: bool
        :return: Impulse response (h) matrix of shape ``(n_dof, n_dof, time_series)`` or tuple based on requested returns.
        :rtype: ndarray or tuple(ndarray)
        """
        if isinstance(freq, list):
            freq = np.array(freq)

        # get FRF matrix:
        FRF_matrix = self.get_FRF_matrix(freq, frf_method=frf_method)

        # obtain h matrix (impulse response function) matrix:
        h_matrix = np.fft.irfft(FRF_matrix)

        if return_t_axis:
            T = 1/(freq[1] - freq[0])
            t = np.linspace(0, T, h_matrix.shape[2], endpoint=False)
            return h_matrix, t

        return h_matrix


    def get_response(self, exc_dof, exc, sampling_rate, resp_dof=None, domain="f", frf_method="f", 
                     mode="full", method="auto", return_matrix=False, return_t_axis=False, return_f_axis=False):
        """
        Get response time series.

        :param exc_dof: Degrees of freedom (masses) where the system is excited.
        :type exc_dof: ndarray, list
        :param exc: Excitation time array 1D (one excitation DOF) or 2D (multiple excitation DOFs).
            1D shape (single excitation DOF): (time series)
            2D shape (multiple excitation DOFs): (number of DOFs, time series)
        :type exc: ndarray
        :param sampling_rate: Sampling rate of excitation time signals.
        :type sampling_rate: int
        :param resp_dof: Degrees of freedom (masses) where the response is calculated. If None - responses of all masses are caluclated.
        :type resp_dof: ndarray, list
        :param frf_method: Method of calculating the FRF matrix:
            ``"f"`` - frequency domain, based on impedance inverse or 
            ``"s"`` - state space domain, based on state-space model parameters
        :type frf_method: str
        :param domain: Domain used for calculation: ``"f"`` - frequency domain multiplication (via FRF matrix) or 
            ``"t"`` - time domain convolution (via impulse response matrix).
        :param mode: A string indicating the size of the output (``"full"``, ``"valid"``, ``"same"``).
        :type mode: str
        :param method: A string indicating which method to use to calculate the convolution (``"auto"``, ``"direct"``, ``"fft"``).
        :type method: str
        :param return_matrix: True if you want to return the FRF matrix (``domain="f"``) or impulse response matrix (``domain="t"``) 
            used for calculation.
        :type return_matrix: bool
        :param return_t_axis: True if you want to return the time axis.
        :type return_t_axis: bool
        :param return_f_axis: True if you want to return the frequency axis.
        :type return_f_axis: bool
        :return: Response time signals array of shape ``(len(resp_dof), time_series)`` or tuple based on requested returns.
        :rtype: ndarray or tuple(ndarray)
        """

        # check domain:
        if not (domain in _DOMAINS):
            raise Exception("Wrong domain calculation type type given. Can be one of %s" % _DOMAINS)
        
        # check mode and method:
        if not (mode in _MODES):
            raise Exception("Wrong mode type given. Can be one of %s" % _MODES)
        if not (method in _METHODS):
            raise Exception("Wrong method type given. Can be one of %s" % _METHODS)
        
        # check exc_dof:
        if isinstance(exc_dof, list):
            exc_dof = np.array(exc_dof)
        if len(exc_dof.shape) > 1:
            raise Exception("Multiple dimension array not allowed for exc_dof array")
        
        # check resp_dof:
        if resp_dof is not None:
            if isinstance(resp_dof, list):
                resp_dof = np.array(resp_dof)
            if len(resp_dof.shape) > 1:
                raise Exception("Multiple dimension array not allowed for resp_dof array")
        else:  # all rsponse DOFs are calculated
            resp_dof = np.arange(0, self.n_dof, 1, dtype=int)

        # check exc:
        if isinstance(exc, list):
            exc = np.array(exc)
        if len(exc.shape) == 1:
            exc = np.expand_dims(exc, 0)
        if len(exc.shape) > 2:
            raise Exception("Input excitation array should be 1D (time series) or 2D (number of DOFs, time series)")
        
        # check sampling_rate:
        if not isinstance(sampling_rate, int):
            raise Exception("Type int required for sampling_rate")

        # freq array:
        freq = np.fft.rfftfreq(exc.shape[1], 1/sampling_rate)

        # calcualte response:
        if domain == "f":
            EXC = np.fft.rfft(exc)  # TODO: normiranje
            RESP = np.zeros((resp_dof.shape[0], freq.shape[0]), dtype="complex128")
            matrix = self.get_FRF_matrix(freq, frf_method=frf_method)
            # calculate response in frequency domain:
            for i in range(EXC.shape[1]):
                RESP[:,i] = matrix[:,:,i][np.ix_(resp_dof, exc_dof)] @ EXC[:,i]
            # back to time domain:
            resp = np.fft.irfft(RESP)  # TODO: normiranje

        elif domain == "t":
            resp = np.zeros((resp_dof.shape[0], exc.shape[1]), dtype=float)
            h_matrix = self.get_h_matrix(freq, frf_method=frf_method)
            # calculate time domain response:
            for i in range(resp_dof.shape[0]):
                for j in range(exc_dof.shape[0]):
                    resp[i] += scipy.signal.convolve(h_matrix[resp_dof[i], exc_dof[j], :], exc[j], 
                                                     mode=mode, method=method)[:exc.shape[1]]

        if return_t_axis:
            T = 1/(freq[1] - freq[0])
            t = np.linspace(0, T, exc.shape[1], endpoint=False)

        if return_matrix and return_t_axis and return_f_axis:
            return resp, matrix, t, freq
        elif return_matrix and return_t_axis:
            return resp, matrix, t
        elif return_matrix and return_f_axis:
            return resp, matrix, freq
        elif return_t_axis and return_f_axis:
            return resp, t, freq
        elif return_t_axis:
            return resp, t
        elif return_f_axis:
            return resp, freq
        elif return_matrix:
            return resp, matrix
        else:
            return resp
        




















    # def get_response_f_domain(self, exc_dof, exc, sampling_rate, resp_dof=None, frf_method="f", 
    #                           return_FRF=False, return_t_axis=False, return_f_axis=False):
    #     """
    #     Get response time series via multiplication in the frequency domain.

    #     :param exc_dof: Degrees of freedom (masses) where the system is excited.
    #     :type exc_dof: ndarray, list
    #     :param exc: Excitation time array 1D (one excitation DOF) or 2D (multiple excitation DOFs).
    #         1D shape (single excitation DOF): (time series)
    #         2D shape (multiple excitation DOFs): (number of DOFs, time series)
    #     :type exc: ndarray
    #     :param sampling_rate: Sampling rate of excitation time signals.
    #     :type sampling_rate: int
    #     :param resp_dof: Degrees of freedom (masses) where the response is calculated. If None - responses of all masses are caluclated.
    #     :type resp_dof: ndarray, list
    #     :param frf_method: Method of calculating the FRF matrix:
    #         ``"f"`` - frequency domain, based on impedance inverse or 
    #         ``"s"`` - state space domain, based on state-space model parameters
    #     :type frf_method: str
    #     :param return_FRF: True if you want to return the FRF matrix used for calculation.
    #     :type return_FRF: bool
    #     :param return_t_axis: True if you want to return the time axis.
    #     :type return_t_axis: bool
    #     :param return_f_axis: True if you want to return the frequency axis.
    #     :type return_f_axis: bool
    #     :return: Response time signals array of shape ``(len(resp_dof), time_series)`` or tuple based on requested returns.
    #     :rtype: ndarray or tuple(ndarray)
    #     """
    #     # check exc_dof:
    #     if isinstance(exc_dof, list):
    #         exc_dof = np.array(exc_dof)
    #     if len(exc_dof.shape) > 1:
    #         raise Exception("Multiple dimension array not allowed for exc_dof array")
        
    #     # check resp_dof:
    #     if resp_dof is not None:
    #         if isinstance(resp_dof, list):
    #             resp_dof = np.array(resp_dof)
    #         if len(resp_dof.shape) > 1:
    #             raise Exception("Multiple dimension array not allowed for resp_dof array")
    #     else:  # all rsponse DOFs are calculated
    #         resp_dof = np.arange(0, self.n_dof, 1, dtype=int)

    #     # check exc:
    #     if isinstance(exc, list):
    #         exc = np.array(exc)
    #     if len(exc.shape) == 1:
    #         exc = np.expand_dims(exc, 0)
    #     if len(exc.shape) > 2:
    #         raise Exception("Input excitation array should be 1D (time series) or 2D (number of DOFs, time series)")
        
    #     # check sampling_rate:
    #     if not isinstance(sampling_rate, int):
    #         raise Exception("Type int required for sampling_rate")

    #     # go to frequency domain and obtain FRF matrix:
    #     freq = np.fft.rfftfreq(exc.shape[1], 1/sampling_rate)
    #     EXC = np.fft.rfft(exc)  # TODO: normiranje
    #     RESP = np.zeros((resp_dof.shape[0], freq.shape[0]), dtype="complex128")
    #     FRF_matrix = self.get_FRF_matrix(freq, frf_method=frf_method)

    #     # calculate response in frequency domain:
    #     for i in range(EXC.shape[1]):
    #         RESP[:,i] = FRF_matrix[:,:,i][np.ix_(resp_dof, exc_dof)] @ EXC[:,i]

    #     # back to time domain:
    #     resp = np.fft.irfft(RESP)  # TODO: normiranje

    #     if return_t_axis:
    #         T = 1/(freq[1] - freq[0])
    #         t = np.linspace(0, T, exc.shape[1], endpoint=False)

    #     if return_FRF and return_t_axis and return_f_axis:
    #         return resp, FRF_matrix, t, freq
    #     elif return_FRF and return_t_axis:
    #         return resp, FRF_matrix, t
    #     elif return_FRF and return_f_axis:
    #         return resp, FRF_matrix, freq
    #     elif return_t_axis and return_f_axis:
    #         return resp, t, freq
    #     elif return_t_axis:
    #         return resp, t
    #     elif return_f_axis:
    #         return resp, freq
    #     elif return_FRF:
    #         return resp, FRF_matrix
    #     else:
    #         return resp
    

    # def get_response_t_domain(self, exc_dof, exc, sampling_rate, resp_dof=None, frf_method="f", mode="full", method="auto", 
    #                           return_h=False, return_t_axis=False, return_f_axis=False):
    #     """
    #     Get response time series via convolution in time domain.

    #     :param exc_dof: Degrees of freedom (masses) where the system is excited.
    #     :type exc_dof: ndarray, list
    #     :param exc: Excitation time array 1D (one excitation DOF) or 2D (multiple excitation DOFs).
    #         1D shape (single excitation DOF): (time series)
    #         2D shape (multiple excitation DOFs): (number of DOFs, time series)
    #     :type exc: ndarray
    #     :param sampling_rate: Sampling rate of excitation time signals.
    #     :type sampling_rate: int
    #     :param resp_dof: Degrees of freedom (masses) where the response is calculated. If None - responses of all masses are caluclated.
    #     :type resp_dof: ndarray, list
    #     :param frf_method: Method of calculating the FRF matrix:
    #         ``"f"`` - frequency domain, based on impedance inverse or 
    #         ``"s"`` - state space domain, based on state-space model parameters
    #     :type frf_method: str
    #     :param mode: A string indicating the size of the output (``"full"``, ``"valid"``, ``"same"``).
    #     :type mode: str
    #     :param method: A string indicating which method to use to calculate the convolution (``"auto"``, ``"direct"``, ``"fft"``).
    #     :type method: str
    #     :param return_h: True if you want to return the h (impulse reposnse function) matrix used for calculation.
    #     :type return_h: bool
    #     :param return_t_axis: True if you want to return the time axis.
    #     :type return_t_axis: bool
    #     :param return_f_axis: True if you want to return the frequency axis.
    #     :type return_f_axis: bool
    #     :return: Response time signals array of shape ``(len(resp_dof), time_series)`` or tuple based on requested returns.
    #     :rtype: ndarray or tuple(ndarray)
    #     """

    #     # check mode and method:
    #     if not (mode in _MODES):
    #         raise Exception("Wrong mode type given. Can be one of %s" % _MODES)
    #     if not (method in _METHODS):
    #         raise Exception("Wrong method type given. Can be one of %s" % _METHODS)
        
    #     # check exc_dof:
    #     if isinstance(exc_dof, list):
    #         exc_dof = np.array(exc_dof)
    #     if len(exc_dof.shape) > 1:
    #         raise Exception("Multiple dimension array not allowed for exc_dof array")
        
    #     # check resp_dof:
    #     if resp_dof is not None:
    #         if isinstance(resp_dof, list):
    #             resp_dof = np.array(resp_dof)
    #         if len(resp_dof.shape) > 1:
    #             raise Exception("Multiple dimension array not allowed for resp_dof array")
    #     else:  # all rsponse DOFs are calculated
    #         resp_dof = np.arange(0, self.n_dof, 1, dtype=int)

    #     # check exc:
    #     if isinstance(exc, list):
    #         exc = np.array(exc)
    #     if len(exc.shape) == 1:
    #         exc = np.expand_dims(exc, 0)
    #     if len(exc.shape) > 2:
    #         raise Exception("Input excitation array should be 1D (time series) or 2D (number of DOFs, time series)")
        
    #     # check sampling_rate:
    #     if not isinstance(sampling_rate, int):
    #         raise Exception("Type int required for sampling_rate")
        
    #     # obtain h matrix obtain h matrix:
    #     freq = np.fft.rfftfreq(exc.shape[1], 1/sampling_rate)  
    #     # TODO: normiranje
    #     resp = np.zeros((resp_dof.shape[0], exc.shape[1]), dtype=float)
    #     h_matrix = self.get_h_matrix(freq, frf_method=frf_method)

    #     # calculate time domain response:
    #     for i in range(resp_dof.shape[0]):
    #         for j in range(exc_dof.shape[0]):
    #             resp[i] += scipy.signal.convolve(h_matrix[resp_dof[i], exc_dof[j], :], exc[j], mode=mode, method=method)[:exc.shape[1]]

    #     if return_t_axis:
    #         T = 1/(freq[1] - freq[0])
    #         t = np.linspace(0, T, exc.shape[1], endpoint=False)

    #     if return_h and return_t_axis and return_f_axis:
    #         return resp, h_matrix, t, freq
    #     elif return_h and return_t_axis:
    #         return resp, h_matrix, t
    #     elif return_h and return_f_axis:
    #         return resp, h_matrix, freq
    #     elif return_t_axis and return_f_axis:
    #         return resp, t, freq
    #     elif return_t_axis:
    #         return resp, t
    #     elif return_f_axis:
    #         return resp, freq
    #     elif return_h:
    #         return resp, h_matrix
    #     else:
    #         return resp