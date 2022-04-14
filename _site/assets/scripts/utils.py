import os
from skimage.util import view_as_blocks, view_as_windows
import numpy as np
import scipy.fftpack
from tqdm import tqdm

cc,rr = np.meshgrid(np.arange(8), np.arange(8))
T = np.sqrt(2 / 8) * np.cos(np.pi * (2*cc + 1) * rr / (2 * 8))
T[0,:] /= np.sqrt(2)
D = np.zeros((64,64))
for i in range(64):
    dcttmp = np.zeros((8,8)) 
    dcttmp[ i//8,np.mod(i,8)] = 1
    TTMP = T@dcttmp@T.T
    D[:,i] = TTMP.ravel()
    
# C ... matrix of DCT coefficients
# I ... matrix of image pixels
# Q ... quantization matrix
# compression and decompression functions DON'T return integers
    
def reshape_view_to_original(arr, orig):
    return np.transpose(arr.reshape(orig.shape[0]//8,orig.shape[1]//8,8,8), [0,2,1,3]).reshape(orig.shape)

def decompress_view(C,Q):
    return (T.T)@(C*Q)@(T) + 128

def decompress_vect(C,Q):
    return ((C*Q)@D) + 128

def decompress_variance_vect(x, Q):
    # equivalent of (T.T*T.T)@(x*Q**2)@(T*T)
    return np.einsum('ijj->ij', D.T@np.einsum('ij, jk->ijk', x, np.diagflat(Q**2))@D)

def compress_view(C,Q):
    return (T@(C-128)@T.T)/Q

def decompress_image(C,Q):
    view = decompress_view(view_as_blocks(C, (8,8)), Q)
    I = reshape_view_to_original(view, C)
    return I

def compress_image(I, Q):
    view = compress_view(view_as_blocks(I, (8,8)), Q)
    C = reshape_view_to_original(view, I)
    return C

def get_error(view, uint=False):
    uint_view = np.round(view)
    if uint:
        uint_view[uint_view>255] = 255
        uint_view[uint_view<0] = 0
    return view - uint_view

def get_spatial_error_var(view, Q):
    I = decompress_view(view, Q)
    return np.var(get_error(I), axis=(-1,-2))

def get_spatial_error_moments(view, Q, uint=False):
    I = decompress_view(view, Q)
    return np.mean(get_error(I, uint), axis=(-1,-2)), np.var(get_error(I, uint), axis=(-1,-2))
    
def binary_entropy(p):
    p[p < 0] = 1
    p0 = 1-p
    p = np.stack([p0, p])
    H = -p*np.log2(p)
    return np.nansum(H)

def invxlnx2_fast(y,f):

    i_large = y>=1000
    i_small = y<1000
    iyL = (np.floor(y[i_small]/0.01)).astype(np.int32)
    iyR = iyL + 1
    iyR[iyR>=100000] = 100000-1

    x = np.zeros(y.shape)
    x[i_small] = f[iyL] + (y[i_small]-(iyL)*0.01)*(f[iyR]-f[iyL])

    z = y[i_large]/np.log(y[i_large]-1)
    for j in range(20):
        z = y[i_large]/np.log(z-1)
        
    x[i_large] = z
    return x

def binary_probs(FI, payload):
    L, R = 5e-2, 5e1
    ixlnx2 = np.load('ixlnx2.npy')
    fL = binary_entropy(1/invxlnx2_fast(L*FI, ixlnx2)) - payload
    fR = binary_entropy(1/invxlnx2_fast(R*FI, ixlnx2)) - payload
    max_iter = 80
    i = 0 
    while (fL*fR > 0) and (i<max_iter):
        i += 1
        if fL > 0:
            R *= 2
            fR = binary_entropy(1/invxlnx2_fast(R*FI, ixlnx2)) - payload
        else:
            L /= 2
            fL = binary_entropy(1/invxlnx2_fast(L*FI, ixlnx2)) - payload
            
    i, fM, TM = 0, 1, np.zeros([max_iter,2])
    while (np.abs(fM) > 1e-2) and (i < max_iter):
        M = (L+R)/2
        fM = binary_entropy(1/invxlnx2_fast(M*FI, ixlnx2)) - payload
        if fL*fM < 0:
            R = M
            fR = fM
        else:
            L = M
            fl = fM
        TM[i,:] = [fM, M]
        i += 1
        
    if i == max_iter:
        M = TM[np.argmin(np.abs(TM[:i,0])),1]
        
    beta = 1/invxlnx2_fast(M*FI, ixlnx2)
    return beta
    
def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho')

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def ternary_entropy(pP1, pM1):
    p0 = 1-pP1-pM1
    p0[p0 <= 0] = 1
    #pP1[pP1==0]=1
    #pM1[pM1==0]=1
    p = np.stack([p0, pP1, pM1])
    H = -p*np.log2(p)
    return np.nansum(H)

def calc_lambda(rho_p1, rho_m1, message_length, n):
    l3 = 1e+3
    m3 = float(message_length+1)
    iterations = 0
    while m3 > message_length:
        l3 *= 2
        pP1 = (np.exp(-l3 * rho_p1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        pM1 = (np.exp(-l3 * rho_m1)) / (1 + np.exp(-l3 * rho_p1) + np.exp(-l3 * rho_m1))
        m3 = ternary_entropy(pP1, pM1)

        iterations += 1
        if iterations > 10:
            return l3
    l1 = 0
    m1 = float(n)
    lamb = 0
    iterations = 0
    alpha = float(message_length)/n
    # limit search to 30 iterations and require that relative payload embedded 
    # is roughly within 1/1000 of the required relative payload
    while float(m1-m3)/n > alpha/1000.0 and iterations<30:
        lamb = l1+(l3-l1)/2
        pP1 = (np.exp(-lamb*rho_p1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        pM1 = (np.exp(-lamb*rho_m1))/(1+np.exp(-lamb*rho_p1)+np.exp(-lamb*rho_m1))
        m2 = ternary_entropy(pP1, pM1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
    iterations += 1;
    return lamb

def embedding_simulator(x, rhoP1, rhoM1, m):
    n = x.size
    lamb = calc_lambda(rhoP1, rhoM1, m, n)
    pChangeP1 = (np.exp(-lamb * rhoP1)) / (1 + np.exp(-lamb * rhoP1) + np.exp(-lamb * rhoM1));
    pChangeM1 = (np.exp(-lamb * rhoM1)) / (1 + np.exp(-lamb * rhoP1) + np.exp(-lamb * rhoM1));
    y = x.copy()
    randChange = np.random.random(y.shape)
    y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1;
    y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1;
    return y, pChangeP1, pChangeM1








