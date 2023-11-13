"""
Unit tests for pyLump.py
"""

import numpy as np
import scipy
import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pyLump


def test_matrices():
    # both:
    model = pyLump.Model(3, 1, 1, 1, boundaries="both")

    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    K = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    C = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    np.testing.assert_equal(model.get_mass_matrix(), M)
    np.testing.assert_equal(model.get_stiffness_matrix(), K)
    np.testing.assert_equal(model.get_damping_matrix(), C)

    # left
    model = pyLump.Model(3, 1, 1, 1, boundaries="left")

    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    K = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])
    
    C = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])
    
    np.testing.assert_equal(model.get_mass_matrix(), M)
    np.testing.assert_equal(model.get_stiffness_matrix(), K)
    np.testing.assert_equal(model.get_damping_matrix(), C)

    # right:
    model = pyLump.Model(3, 1, 1, 1, boundaries="right")

    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    K = np.array([[1, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    C = np.array([[1, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    np.testing.assert_equal(model.get_mass_matrix(), M)
    np.testing.assert_equal(model.get_stiffness_matrix(), K)
    np.testing.assert_equal(model.get_damping_matrix(), C)

    # free
    model = pyLump.Model(3, 1, 1, 1, boundaries="free")

    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    K = np.array([[1, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])
    
    C = np.array([[1, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 1]])
    
    np.testing.assert_equal(model.get_mass_matrix(), M)
    np.testing.assert_equal(model.get_stiffness_matrix(), K)
    np.testing.assert_equal(model.get_damping_matrix(), C)


def test_FRF():
    model = pyLump.Model(3, 1, 1, 1, boundaries="both")

    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    K = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    C = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    freq = np.linspace(0, 2, 2000)
    w = 2*np.pi*freq

    frf = np.zeros([3, 3, len(freq)], dtype="complex128")

    for i, wi in enumerate(w):
        frf[:,:,i] = scipy.linalg.inv(K - wi**2 * M + 1j*wi*C)
    
    # f method:
    np.testing.assert_equal(np.abs(model.get_FRF_matrix(freq, frf_method="f")), np.abs(frf))
    # s method:
    np.testing.assert_allclose(np.abs(model.get_FRF_matrix(freq, frf_method="s")), np.abs(frf), rtol=1e-12)


def test_h():
    model = pyLump.Model(3, 1, 1, 1, boundaries="both")

    M = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    K = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    C = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    freq = np.linspace(0, 2, 2000)
    w = 2*np.pi*freq

    frf = np.zeros([3, 3, len(freq)], dtype="complex128")

    for i, wi in enumerate(w):
        frf[:,:,i] = scipy.linalg.inv(K - wi**2 * M + 1j*wi*C)
    
    # f method:
    np.testing.assert_equal(model.get_IRF_matrix(freq, frf_method="f"), np.fft.irfft(frf))
    # s method:
    np.testing.assert_allclose(model.get_IRF_matrix(freq, frf_method="s"), np.fft.irfft(frf), rtol=1e-5)
    
    
def test_response():
    model = pyLump.Model(3, 0.1, 10, 0.01, boundaries="both")

    M = np.array([[0.1, 0, 0],
                  [0, 0.1, 0],
                  [0, 0, 0.1]])
    
    K = np.array([[20, -10, 0],
                  [-10, 20, -10],
                  [0, -10, 20]])
    
    C = np.array([[0.02, -0.01, 0],
                  [-0.01, 0.02, -0.01],
                  [0, -0.01, 0.02]])

    exc_dof = [0]
    exc = np.zeros(1000)
    exc[0] = 1
    sampling_rate=10

    freq = np.fft.rfftfreq(1000, 1/sampling_rate)
    w = 2*np.pi*freq

    frf = np.zeros([3, 3, len(freq)], dtype="complex128")

    for i, wi in enumerate(w):
        frf[:,:,i] = scipy.linalg.inv(K - wi**2 * M + 1j*wi*C)

    # f method:
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="f", frf_method="f"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-11)
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="t", frf_method="f"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-11)
    # s method:
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="f", frf_method="s"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-9)
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="t", frf_method="s"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-9)
    


if __name__ == '__mains__':
    np.testing.run_module_suite()