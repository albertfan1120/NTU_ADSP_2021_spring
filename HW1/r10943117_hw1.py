import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class Low_pass_FIR():
    def __init__(self):
        self.N = 17  
        self.W_pass = 1
        self.W_stop = 0.6
        self.delta  = 0.0001 # in step 5
        self.k = int((self.N - 1) / 2)  # 8
        self.f = np.arange(0, 0.5, 0.0001)
        
    
    def mini_max(self):
        Hd = np.where(self.f <= 0.225, 1, 0)
        W = self.get_W()
        
        # step 1
        F = np.array([0, 0.03, 0.11, 0.13, 0.16, 0.32, 0.35, 0.4, 0.45, 0.5])
        E1 = None
        
        do = True
        error_list = []
        while(do):
            # step 2
            M = self.get_matrix(F)
            b = np.where(F <= 0.225, 1, 0)
            x = np.linalg.solve(M, b)
            s, e = x[:-1], x[-1]
            
            # step 3
            R = self.get_R(s)
            err = (R - Hd) * W
            
            # step 4
            P = self.get_extreme(err)

            # step 5
            E0 = np.abs(err).max()
            error_list.append(E0)
            if E1 == None: # first iter
                E1, F = E0, P
            elif 0 <= E1 - E0 <= self.delta:
                do = False
            else: 
                E1, F = E0, P
        
        # calculate h
        h = np.zeros((self.N))
        h[self.k] = s[0]
        for n in range(1, self.k + 1): h[self.k + n] = h[self.k - n] = s[n] / 2
            
        return R, h, error_list
        

    def get_W(self):
        W = np.zeros_like(self.f)
        for i, f in enumerate(self.f):
            if f <= 0.2: W[i] = self.W_pass
            elif f >= 0.25: W[i] = self.W_stop
        
        return W
    
    
    def get_matrix(self, F):
        two_pi_k = np.arange(1, self.k + 1) * 2 * np.pi
        M = np.zeros((self.k + 2, self.k + 2))
        for i in range(self.k + 2):
            cos = np.cos(F[i] * two_pi_k)
            W = self.W_pass if F[i] <= 0.24 else self.W_stop
            
            M[i] = np.hstack([1, cos, (-1) ** i / W]) 
        
        return M
    
    
    def get_R(self, s):
        n_pi = np.arange(0, self.k + 1) * 2 * np.pi
        R = np.zeros(int(0.5 / 0.0001))
        for i, f in enumerate(self.f): R[i] = (s * np.cos(n_pi * f)).sum()

        return R
    
    
    def get_extreme(self, err):
        local = np.hstack([self.f[signal.argrelextrema(err, np.greater)],
                           self.f[signal.argrelextrema(err, np.less)]])
        k_2 = self.k + 2
        
        # two bounds F = 0, 0.5
        num = local.shape[0]
        if k_2 > num:
            if err[0] > err[1] and err[0] > 0:   local = np.hstack([local, self.f[0]])
            elif err[0] < err[1] and err[0] < 0: local = np.hstack([local, self.f[0]])
        
        num = local.shape[0]
        if k_2 > num:
            if err[-1] > err[-2] and err[-1] > 0: local = np.hstack([local, self.f[4999]])
            elif err[-1] < err[-2] and err[-1] < 0: local = np.hstack([local, self.f[4999]])
            
        # two transition band
        num = local.shape[0]
        if k_2 > num:
            if err[2000] > err[1999] and err[2000] > 0: local = np.hstack([local, self.f[2000]])
            elif err[2000] < err[1999] and err[2000] < 0: local = np.hstack([local, self.f[2000]])
            
        num = local.shape[0]
        if k_2 > num:
            if err[2500] > err[2501] and err[2500] > 0: local = np.hstack([local, self.f[2500]])
            elif err[2500] < err[2501] and err[2500] < 0: local = np.hstack([local, self.f[2500]])
        
        return np.sort(local)
        
        
def plot_response(R, h):
    plt.plot(np.arange(0, 0.5, 0.0001), R)
    plt.title('Frequency Response')
    plt.xlabel('normalized frequency F')
    plt.show()
    
    plt.stem(np.arange(h.shape[0]), h) 
    plt.title('impulse response h[n]')
    plt.show()


if __name__ == '__main__':
    R, h, error_list = Low_pass_FIR().mini_max()
    
    plot_response(R, h)
    
    print("Error for each ieration")
    print(error_list)
