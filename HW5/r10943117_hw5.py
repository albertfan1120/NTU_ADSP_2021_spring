import numpy as np
def find_inverse(alpha, M):
    for k in range(1,M):
        if (alpha*k)%M==1:
            return k

def check_alpha(alpha, N, M):
    for k in range(1,N):
        if alpha**k%M==1:
            return False
    if alpha**N%M!=1:
        return False
    return True


def split_num(x):
    return x//2, x-x//2

def mod_multiply(alpha, exp, M):
    if exp>3:
        a, b = split_num(exp)
        return ((mod_multiply(alpha,a,M)) * (mod_multiply(alpha,b,M)) )%M
    else:
        return (alpha**(exp))%M

if __name__ == '__main__':
    N=4
    M=5
    forward = np.ones((N,N))
    backward = np.ones((N,N))

    alpha = 2
    while True:
        if check_alpha(alpha,N,M):
            break
        alpha+=1
    print('alpha:', alpha)

    for i in range(N):
        for j in range(N):
            forward[i][j] = mod_multiply(alpha, i*j, M)

    alpha_inv = find_inverse(alpha,M)
    N_inv = find_inverse(N,M)
    print('alpha_inv:', alpha_inv)
    print('N_inv:', N_inv)

    for i in range(N):
        for j in range(N):
            backward[i][j] = mod_multiply(alpha_inv, i*j, M)
    backward = N_inv*backward

    print('\nforward:')
    print(forward)
    print('\nbackward:')
    print(backward)