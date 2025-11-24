from skimage.util import view_as_blocks
import numpy as np
from scipy.signal import convolve2d
from scipy.fftpack import idct
from JPEG_utils import *
from embedding_simulator import Embedding_simulator

# DCT filters
def w(k):
    if k == 0:
        return 1/np.sqrt(2)
    else:
        return 1

def f_DCT():
    f = np.zeros((8,8,8,8))
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    f[i,j,k,l] = w(k)*w(l)/4*np.cos(np.pi*k*(2*i+1)/16)*np.cos(np.pi*l*(2*j+1)/16)
    return f


def jeep(I, Q, payload):
    # I ... uncompressed precover image
    # Q ... Quantization table
    # payload ... relative payload in bpnzAC
    Q = np.float32(Q)
    C = compress_image(np.float32(I), Q) # DCT coefficients before rounding to integers
    C_COEFFS = np.round(C)
    e = C - C_COEFFS
    e_view = view_as_blocks(e, (8,8))
    nzAC = np.count_nonzero(C_COEFFS) - np.count_nonzero(C_COEFFS[::8,::8])
    m = np.round(payload*nzAC)


    spatial_var = view_as_blocks(MiPOD_variance(I), (8,8))



    spatial_var[spatial_var<1e-5] = 1e-5

    e_var = get_DCT_variance(spatial_var, 1)
    e_var[e_var<1e-10] = 1e-10

    e_var = view_as_blocks(1/np.sqrt(post_filter_FI(1/reshape_view_to_original(e_var,e)**2)), (8,8)) # smooth DCT variances

    spatial_var = get_spatial_variance(e_var,1)

    kargs = (e_view, spatial_var, Q)

    pP1, pM1, M = Embedding_simulator.calc_lambda_FI(m, newton_solver_JEEP_full, kargs)
    pP1 = reshape_view_to_original(pP1, C)
    pM1 = reshape_view_to_original(pM1, C)

    S_COEFFS = C_COEFFS.copy()
    randChange = np.random.random(S_COEFFS.shape)
    S_COEFFS[randChange < pP1] += 1
    S_COEFFS[randChange >= 1-pM1] -= 1
    return S_COEFFS

# Fisher Information smoothing
def post_filter_FI(FI, F=[[1, 3, 1], [3, 4, 3], [1, 3, 1]]):
    F = F / np.sum(F)
    tmp = np.pad(FI, ((8, 8), (8, 8)))
    tmp[0:8, :] = tmp[8:16, :]
    tmp[:, 0:8] = tmp[:, 8:16]

    tmp[-8:, :] = tmp[-16:-8, :]
    tmp[:, -8:] = tmp[:, -16:-8]

    FI = (
        tmp[0:-16, 0:-16] * F[0, 0]
        + tmp[8:-8, 0:-16] * F[1, 0]
        + tmp[16:, 0:-16] * F[2, 0]
        + tmp[0:-16, 8:-8] * F[0, 1]
        + tmp[8:-8, 8:-8] * F[1, 1]
        + tmp[16:, 8:-8] * F[2, 1]
        + tmp[0:-16, 16:] * F[0, 2]
        + tmp[8:-8, 16:] * F[1, 2]
        + tmp[16:, 16:] * F[2, 2]
    )
    return FI

# Newton Solver
def newton_solver_JEEP_full(l, e, s_var, Q):
    acc = 1e-7
    beta = 1e-3*np.ones((*e.shape,2))
    ind = e<10
    i = 0
    max_iter = 20
    wetCost = 1e20
    f = f_DCT()
    FI_p = np.einsum('ijkl,abij->abkl', f**4, 1/s_var**2)*Q**4 # effect of variance


    FI_m = FI_p*(1+2*e)**4
    FI_pm = FI_p*(1+2*e)**2*(1-2*e)**2
    FI_p = FI_p*(1-2*e)**4
    FI_p[FI_p>wetCost] = wetCost
    FI_m[FI_m>wetCost] = wetCost
    FI_pm[FI_pm>wetCost] = wetCost

    maxCostMat = e < -1
    maxCostMat[:,:,0,0] = True
    maxCostMat[:,:,0,4] = True
    maxCostMat[:,:,4,0] = True
    maxCostMat[:,:,4,4] = True
    FI_p[maxCostMat*(np.abs(e)>0.4999)] = wetCost
    FI_m[maxCostMat*(np.abs(e)>0.4999)] = wetCost

    while (ind.sum() > 0) and (i<max_iter):
        i+=1
        beta[ind] = np.clip(beta[ind],1e-16,0.5)
        P1 = beta[ind,0]
        M1 = beta[ind,1]
        F1 = P1*FI_p[ind] + M1*FI_pm[ind] - l*np.log((1-P1-M1)/P1)
        F2 = M1*FI_m[ind] + P1*FI_pm[ind] - l*np.log((1-P1-M1)/M1)
        F1 = np.nan_to_num(F1)
        F2 = np.nan_to_num(F2)
        M11 = FI_p[ind] + l*((1-M1)/(P1*(1-P1-M1)))
        M22 = FI_m[ind] + l*((1-P1)/(M1*(1-P1-M1)))
        M12 = FI_pm[ind] + l/(1-P1-M1)
        M11 = np.nan_to_num(M11)
        M22 = np.nan_to_num(M22)
        M12 = np.nan_to_num(M12)

        detM = M11*M22-M12**2
        detM = np.nan_to_num(detM)
        upd1 = (M22*F1-M12*F2)/detM
        upd2 = (M11*F2-M12*F1)/detM

        tmp1 = P1 - upd1
        tmp2 = M1 - upd2

        beta[ind,0] = tmp1
        beta[ind,1] = tmp2
        beta[beta>0.5] = 0
        beta = np.clip(beta,0,0.5)
        ind[ind] = (np.abs(upd1)>acc) + (np.abs(upd2)>acc)


    beta = np.nan_to_num(beta)
    beta = np.clip(beta,0,0.5)
    return np.squeeze(np.split(beta,2,axis=-1))

# MiPOD variance estimator functions

def MiPOD_variance(I, BlockSize=3, Degree=3):
    # Estimation of the pixels' variance based on a 2D-DCT (trigonometric polynomial) model

    if BlockSize % 2 == 0:
        raise ValueError("The block dimensions should be odd!!")
    if Degree > BlockSize:
        raise ValueError("Number of basis vectors exceeds block dimension!!")
    # number of parameters per block
    q = Degree * (Degree + 1) // 2
    WienerResidual = wienerFilter(I)

    # Build G matirx
    BaseMat = np.zeros((BlockSize, BlockSize))
    BaseMat[0, 0] = 1
    G = np.zeros((BlockSize ** 2, q))
    k = 0
    for xShift in range(Degree):
        for yShift in range(Degree - xShift):
            G[:, k] = np.reshape(
                idct2(np.roll(np.roll(BaseMat, xShift, axis=0), yShift, axis=1)),
                BlockSize ** 2,
            )
            k = k + 1

    # Estimate the variance
    PadSize = [BlockSize // 2, BlockSize // 2]
    I2C = im2col(np.pad(WienerResidual, PadSize, "symmetric"), (BlockSize, BlockSize))
    PGorth = np.eye(BlockSize ** 2) - np.dot(
        G, np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T)
    )
    EstimatedVariance = np.reshape(
        np.sum(np.dot(PGorth, I2C) ** 2, axis=0) / (BlockSize ** 2 - q), WienerResidual.shape
    )
    return EstimatedVariance

def idct2(x):
    return idct(idct(x, norm="ortho").T, norm="ortho").T

def wienerFilter(Cover):
    # Compute the wiener filtering the same way matlab does (wiener2)
    # One bug is removed: the convolution used in the wiener2 function does not
    # Use zero padding here.
    lp_filt = [[0.25, 0.25, 0.0], [0.25, 0.25, 0.0], [0.0, 0.0, 0.0]]

    Loc_mean = convolve2d(Cover, lp_filt, "same")
    Loc_sigma = convolve2d(Cover ** 2, lp_filt, "same") - Loc_mean ** 2

    sigma_mean = np.average(Loc_sigma)

    WienerResidual = np.zeros(Cover.shape)
    Wiener_1st = np.zeros(Cover.shape)
    Wiener_1st[Loc_sigma != 0] = Cover[Loc_sigma != 0] - Loc_mean[Loc_sigma != 0]
    Wiener_2nd = np.zeros(Cover.shape)
    Wiener_2nd[Loc_sigma > sigma_mean] = -(
        (Loc_sigma[Loc_sigma > sigma_mean] - sigma_mean)
        / Loc_sigma[Loc_sigma > sigma_mean]
    ) * (Cover[Loc_sigma > sigma_mean] - Loc_mean[Loc_sigma > sigma_mean])

    WienerResidual = Wiener_1st + Wiener_2nd
    return WienerResidual

def im2col(A, B):

    # Parameters
    M, N = A.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel())
    return out
