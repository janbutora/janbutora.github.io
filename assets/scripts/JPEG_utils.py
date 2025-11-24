from skimage.util import view_as_blocks
import numpy as np

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

def get_DCT_variance(x,Q):
    return (T*T)@(x)@(T.T*T.T)/Q**2

def get_spatial_variance(x, Q):
    return (T.T*T.T)@(x*Q**2)@(T*T)
