"""
Unit tests for pyLump.py
"""

import numpy as np
import scipy
import pytest
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
    np.testing.assert_allclose(np.abs(model.get_FRF_matrix(freq, frf_method="s")), np.abs(frf), rtol=1e-10)


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

    freq = np.fft.rfftfreq(exc.shape[0], 1/sampling_rate)
    w = 2*np.pi*freq

    frf = np.zeros([3, 3, len(freq)], dtype="complex128")

    for i, wi in enumerate(w):
        frf[:,:,i] = scipy.linalg.inv(K - wi**2 * M + 1j*wi*C)

    # f method:
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="f", frf_method="f"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-8)
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="t", frf_method="f"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-8)
    # s method:
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="f", frf_method="s"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-8)
    np.testing.assert_allclose(model.get_response(exc_dof, exc, sampling_rate, domain="t", frf_method="s"), 
                               np.fft.irfft(frf[0,:]), rtol=1e-8)
    


@pytest.mark.parametrize("frf_method", ["f", "s"])
def test_response_FRF_matrix_alignment(frf_method):
    # https://github.com/ladisk/pyLump/issues/3
    fs, N = 2000, 4000
    model = pyLump.Model(3, mass=1.0, stiffness=2e4, damping=2.0, boundaries="both")
    exc = np.random.default_rng(0).standard_normal(N)
    _, H_used, freq = model.get_response([0], exc, fs, resp_dof=[0, 1, 2], domain="f",
                                         frf_method=frf_method, return_matrix=True, 
                                         return_f_axis=True)
    np.testing.assert_equal(freq, np.fft.rfftfreq(N, 1/fs))
    np.testing.assert_allclose(H_used, model.get_FRF_matrix(freq, frf_method=frf_method), 
                               rtol=1e-12, atol=0)


@pytest.mark.parametrize("domain", ["f", "t"])
def test_response_harmonic(domain):
    # periodic harmonic excitation at bin k: steady state response is H(f_k) * excitation
    fs, N, k = 1000, 2000, 37
    model = pyLump.Model(3, mass=1.0, stiffness=2e4, damping=50.0, boundaries="both")
    f_k = k * fs / N
    t = np.arange(N) / fs
    exc = np.cos(2*np.pi*f_k*t)
    resp = model.get_response([0], exc, fs, domain=domain)
    H = model.get_FRF_matrix(np.array([f_k]))[:, 0, 0]
    expected = np.abs(H)[:, None] * np.cos(2*np.pi*f_k*t + np.angle(H)[:, None])
    if domain == "f":
        np.testing.assert_allclose(resp, expected, rtol=0, atol=1e-12 * np.abs(H).max())
    else:  # linear convolution: compare after the transient has decayed
        np.testing.assert_allclose(resp[:, N//2:], expected[:, N//2:], rtol=0, 
                                   atol=1e-3 * np.abs(H).max())


@pytest.mark.parametrize("frf_method", ["f", "s"])
def test_response_free(frf_method):
    # singular stiffness matrix: zero frequency bin of the FRF is set to 0
    fs, N = 1000, 2000
    model = pyLump.Model(3, mass=1.0, stiffness=2e4, damping=2.0, boundaries="free")
    exc = np.random.default_rng(1).standard_normal(N)
    for domain in ["f", "t"]:
        resp, H_used, freq = model.get_response([0], exc, fs, domain=domain, 
                                                frf_method=frf_method, 
                                                return_matrix=True, return_f_axis=True)
        assert np.all(np.isfinite(resp))
    _, H_used, freq = model.get_response([0], exc, fs, domain="f", frf_method=frf_method, 
                                         return_matrix=True, return_f_axis=True)
    np.testing.assert_equal(H_used[:, :, 0], 0)
    np.testing.assert_allclose(H_used[:, :, 1:], 
                               model.get_FRF_matrix(freq[1:], frf_method=frf_method), rtol=1e-12)


@pytest.mark.parametrize("domain", ["f", "t"])
def test_response_odd_length(domain):
    model = pyLump.Model(3, mass=1.0, stiffness=2e4, damping=2.0, boundaries="both")
    exc = np.random.default_rng(2).standard_normal(1001)
    resp = model.get_response([0], exc, 1000, domain=domain)
    assert resp.shape == (3, 1001)


if __name__ == '__mains__':
    np.testing.run_module_suite()