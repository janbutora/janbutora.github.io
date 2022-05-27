import jpegio as jio
from skimage.util import view_as_blocks
import numpy as np
import scipy.signal

import argparse
import sys

from utils import *


def JUNI_costs(C_COEFFS, Q):
    C_COEFFS = np.copy(C_COEFFS)
    S_COEFFS = np.copy(C_COEFFS)
    cover_spatial = decompress_image(C_COEFFS, Q)
    if cover_spatial.shape[-1] == 1:
        cover_spatial = np.squeeze(cover_spatial)

    hpdf = np.array(
        [
            -0.0544158422,
            0.3128715909,
            -0.6756307363,
            0.5853546837,
            0.0158291053,
            -0.2840155430,
            -0.0004724846,
            0.1287474266,
            0.0173693010,
            -0.0440882539,
            -0.0139810279,
            0.0087460940,
            0.0048703530,
            -0.0003917404,
            -0.0006754494,
            -0.0001174768,
        ]
    )

    sign = np.array([-1 if i % 2 else 1 for i in range(len(hpdf))])
    lpdf = hpdf[::-1] * sign

    F = []
    F.append(np.outer(lpdf.T, hpdf))
    F.append(np.outer(hpdf.T, lpdf))
    F.append(np.outer(hpdf.T, hpdf))

    # Pre-compute impact in spatial domain when a jpeg coefficient is changed by 1
    spatial_impact = {}
    for i in range(8):
        for j in range(8):
            test_coeffs = np.zeros((8, 8))
            test_coeffs[i, j] = 1
            spatial_impact[i, j] = idct2(test_coeffs) * Q[i, j]

    # Pre compute impact on wavelet coefficients when a jpeg coefficient is changed by 1
    wavelet_impact = {}
    for f_index in range(len(F)):
        for i in range(8):
            for j in range(8):
                wavelet_impact[f_index, i, j] = scipy.signal.correlate2d(
                    spatial_impact[i, j],
                    F[f_index],
                    mode="full",
                    boundary="fill",
                    fillvalue=0.0,
                )  # XXX

    # Create reference cover wavelet coefficients (LH, HL, HH)
    pad_size = 16  # XXX
    spatial_padded = np.pad(cover_spatial, (pad_size, pad_size), "symmetric")
    # print(spatial_padded.shape)

    RC = []
    for i in range(len(F)):
        f = scipy.signal.correlate2d(spatial_padded, F[i], mode="same", boundary="fill")
        RC.append(f)

    k, l = C_COEFFS.shape
    nzAC = np.count_nonzero(S_COEFFS) - np.count_nonzero(S_COEFFS[::8, ::8])

    rho = np.zeros((k, l))
    tempXi = [0.0] * 3
    sgm = 2 ** (-6)

    # Computation of costs
    for row in range(k):
        for col in range(l):
            mod_row = row % 8
            mod_col = col % 8
            sub_rows = list(
                range(row - mod_row - 6 + pad_size - 1, row - mod_row + 16 + pad_size)
            )
            sub_cols = list(
                range(col - mod_col - 6 + pad_size - 1, col - mod_col + 16 + pad_size)
            )

            for f_index in range(3):
                RC_sub = RC[f_index][sub_rows][:, sub_cols]
                wav_cover_stego_diff = wavelet_impact[f_index, mod_row, mod_col]
                tempXi[f_index] = abs(wav_cover_stego_diff) / (abs(RC_sub) + sgm)

            rho_temp = tempXi[0] + tempXi[1] + tempXi[2]
            rho[row, col] = np.sum(rho_temp)

    wet_cost = 10 ** 13
    rho[rho > wet_cost] = wet_cost
    rho[np.isnan(rho)] = wet_cost
    rho[S_COEFFS > 1023] = wet_cost
    rho[S_COEFFS < -1023] = wet_cost

    return rho


def direction_to_costs(S_view, Q, direction, C_var, k, l):
    block = np.zeros_like(S_view)
    block[:, k, l] = direction
    S_mean, S_var = get_spatial_error_moments(S_view + block, Q)
    delta = np.abs(C_var - S_var)
    rho = delta
    return rho


def compute_costs_SVP(S_view, C_var, Q, k, l):
    changes = np.array([-1, 1])
    S_var = np.zeros((S_view.shape[0], changes.size))
    S_mean = np.zeros_like(S_var)
    candidate_block = np.zeros_like(S_view)

    for i, change in enumerate(changes):
        candidate_block[:, k, l] = change
        S_mean[:, i], S_var[:, i] = get_spatial_error_moments(
            S_view + candidate_block, Q
        )

    direction = changes[np.argmin(np.abs(S_var - C_var.reshape(-1, 1)), axis=1)]
    wetCost = 10 ** 13

    rho = direction_to_costs(
        S_view, Q, direction, C_var, k, l
    )  # not really needed since we already have means and variances stored
    rhoP1 = np.copy(rho)
    rhoM1 = np.copy(rho)
    rhoP1[direction == -1] = wetCost
    rhoM1[direction == 1] = wetCost

    rhoP1[C_var == 0] = wetCost
    rhoM1[C_var == 0] = wetCost
    # use uniform costs whenever they are not wet already
    rhoP1[rhoP1 < wetCost] = 1
    rhoM1[rhoM1 < wetCost] = 1

    rhoM1[S_view[:, k, l] < -1023] = wetCost
    rhoP1[S_view[:, k, l] > 1023] = wetCost

    return rhoP1, rhoM1, direction


def Embed_lattice_SVP(S_view, C_var, Q, k, l, m, uni_costs):  # always binary embedding!
    rhoP1, rhoM1, direction = compute_costs_SVP(S_view, C_var, Q, k, l)
    FI = np.min([rhoP1, rhoM1], axis=0) * uni_costs
    beta = binary_probs(FI, m)
    r = np.random.random(S_view[:, k, l].shape)
    modifPM1 = r < beta
    S_COEFFS = S_view[:, k, l]
    S_COEFFS[modifPM1] += np.int32(direction[modifPM1])
    betaP1 = np.copy(beta)
    betaM1 = np.copy(beta)
    betaP1[rhoP1 > rhoM1] = 0
    betaM1[rhoM1 > rhoP1] = 0
    return S_COEFFS, betaP1, betaM1


def Embed_SVP(C_STRUCT, payload, uni_costs=False):
    C_COEFFS = C_STRUCT.coef_arrays[0]
    Q = C_STRUCT.quant_tables[0]
    C_view = view_as_blocks(C_COEFFS, (8, 8)).reshape(-1, 8, 8)
    S_view = np.copy(C_view)
    betaP1 = np.zeros(S_view.shape)
    betaM1 = np.copy(betaP1)

    C_mean, C_var = get_spatial_error_moments(C_view, Q)
    nzAC = np.count_nonzero(C_COEFFS) - np.count_nonzero(C_COEFFS[::8, ::8])

    if uni_costs:
        uni_rho = JUNI_costs(C_COEFFS, Q)
        uni_rho = view_as_blocks(uni_rho, (8, 8)).reshape(-1, 8, 8)
    else:
        uni_rho = np.ones_like(C_view)

    P = np.random.permutation(64)
    mode_m = np.ones(P.shape) * np.round(payload * nzAC / 64)  # payload per mode

    for i in range(64):
        k = P[i] // 8
        l = np.mod(P[i], 8)
        # run for every mode (k,l) separately
        S_view[:, k, l], betaP1[:, k, l], betaM1[:, k, l] = Embed_lattice_SVP(
            S_view, C_var, Q, k, l, mode_m[i], uni_rho[:, k, l]
        )

    S_COEFFS = reshape_view_to_original(S_view, C_COEFFS)
    betaP1 = reshape_view_to_original(betaP1, C_COEFFS)
    betaM1 = reshape_view_to_original(betaM1, C_COEFFS)
    S_STRUCT = C_STRUCT
    S_STRUCT.coef_arrays[0][:] = S_COEFFS

    return S_STRUCT
