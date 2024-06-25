import numpy as np
import matplotlib.pyplot as plt


def Hilbert_transform(k):
    H = np.hstack((-1j * np.ones((k+1)), 1j * np.ones((k))))
    H[0] = 0
    H[1], H[2*k] = -0.9j, 0.9j
    H[k], H[k+1] = -0.7j, 0.7j
    
    r1 = np.fft.ifft(H)
    h = np.hstack((r1[k+1:], r1[:k+1]))
    
    F = np.arange(0, 1, 0.0001)
    R = np.zeros_like(F, dtype = complex)
    n = np.arange(-k, k+1, 1)
    for i, f in enumerate(F):
        base = np.exp(-1j * 2 * np.pi * f * n)
        R[i] = np.sum(h * base)
    
    return h.real, R.imag
        
        
def plot_response(h, R_imag):
    plt.plot(np.arange(0, 1, 0.0001), R_imag)
    plt.title('Imaginary Part Frequency Response')
    plt.xlabel('normalized frequency F')
    plt.show()
    
    plt.stem(np.arange(h.shape[0]), h) 
    plt.title('Impulse Response h[n]')
    plt.show()


if __name__ == '__main__':
    h, R_imag = Hilbert_transform(k = 15)
    plot_response(h, R_imag)