import numpy as np
import jpeglib
from PIL import Image
import os
from JPEG_utils import *
from embedding_simulator import Embedding_simulator

_WET_COST_ROBUST = 10 ** 16
_WET_COST_UERD = 10 ** 13
_METHOD = 'convert'
_FLAG_TRELLIS = ' -notrellis '

def compute_cost(energies, c_quant):
    """
    Computation of costs
    """
    (m, n) = energies.shape
    m -= 2
    n -= 2
    k = m * 8
    l = n * 8
    rho = np.zeros((k, l))
    for block_row in range(m):
        for block_col in range(n):
            block_rho = energies[block_row + 1][block_col + 1] + \
                                (energies[block_row][block_col] + \
                                    energies[block_row][block_col + 1] + \
                                    energies[block_row][block_col + 2] + \
                                    energies[block_row + 1][block_col] + \
                                    energies[block_row + 1][block_col + 2] + \
                                    energies[block_row + 2][block_col + 1] + \
                                    energies[block_row + 2][block_col] + \
                                    energies[block_row + 2][block_col + 2]) * 0.25
            if (block_rho == 0):
                rho[block_row * 8:(block_row + 1) * 8, block_col * 8:(block_col + 1) * 8] = _WET_COST_UERD
            else:
                for row in range(8):
                    for col in range(8):
                        if (row == 0 and col == 0):
                            mode_rho = 0.5 * (c_quant[0][1] + c_quant[1][0])
                        else:
                            mode_rho = c_quant[row][col]
                        rho[block_row * 8 + row][block_col * 8 + col] = (mode_rho / block_rho)

    return rho

def compute_Dmn(c_coeffs, c_quant):
    """
    Comutation of energies of blocks
    """
    (m, n) = c_coeffs.shape
    m = m // 8
    n = n // 8
    energies = np.zeros((m+2, n+2))
    for block_row in range(m):
        for block_col in range(n):
            current_block = np.copy(c_coeffs[block_row * 8:(block_row + 1) * 8,block_col * 8:(block_col + 1) * 8])
            current_block[0][0] = 0
            for row in range(8):
                for col in range(8):
                    energies[block_row + 1][block_col + 1] += abs(current_block[row][col]) * c_quant[row][col]

    energies[0][0] = energies[1][1]
    energies[0][n+1] = energies[1][n]
    energies[m+1][0] = energies[m][1]
    energies[m+1][n+1] = energies[m][n]
    for block_row in range(m):
        energies[block_row + 1][0] = energies[block_row + 1][1]
        energies[block_row + 1][n+1] = energies[block_row + 1][n]
    for block_col in range(n):
        energies[0][block_col + 1] = energies[1][block_col + 1]
        energies[m+1][block_col + 1] = energies[m][block_col + 1]

    return energies

def UERD(c_struct, i=0):
    """
    Compute costs according to cover in spatial and DCT domains
    """
    try:
        j = min(i, 1)
        c_quant = c_struct.quant_tables[j][:]
        c_coeffs = np.copy(c_struct.coef_arrays[i][:])

        # Compute the block energies
        energies = compute_Dmn(c_coeffs, c_quant)
        # Compute rhos
        rho = compute_cost(energies, c_quant)
        # Adjust embedding costs
        rho[rho > _WET_COST_UERD] = _WET_COST_UERD # Threshold on the costs
        rho[np.isnan(rho)] = _WET_COST_UERD # Check if all elements are numbers
        return rho
    except (TypeError):
        pass

def compute_rhos(c_struct, i=0):
    rho = UERD(c_struct, i)

    rhoP1 = np.copy(rho)
    rhoM1 = np.copy(rho)

    c_coeffs= c_struct.coef_arrays[i][:]
    rhoP1[c_coeffs > 1023] = _WET_COST_UERD # Do not embed +1 if the DCT coeff has max value
    rhoM1[c_coeffs < -1023] = _WET_COST_UERD # Do not embed -1 if the DCT coeff has min value

    return rhoP1, rhoM1

def compress_image(inputPath, outputPath, quality):
    if _METHOD == 'convert':
        cmd = 'convert -define jpeg:optimize-coding=false -quality ' + str(quality) + ' ' + inputPath + ' ' + outputPath
    elif _METHOD == 'mozjpeg':
        cmd = '/opt/homebrew/opt/mozjpeg/bin/cjpeg' + _FLAG_TRELLIS + '-quality ' + str(quality) + ' -grayscale -outfile ' + outputPath + ' ' + inputPath
    os.system(cmd)

def recompress_coefficients(C, qf):
    path = 'tmp.jpg'
    DCpath = 'tmp_DC.jpg'
    I = np.clip(np.round(decompress_image(C,1)), 0, 255).astype(np.uint8)
    Image.fromarray(I).save(path, quality=qf)
    s_struct = jpeglib.to_jpegio(jpeglib.read_dct(path))
    s_struct.coef_arrays[0][:] = np.int32(C)
    s_struct.write(path)
    compress_image(path, DCpath, qf)
    STRUCT = jpeglib.to_jpegio(jpeglib.read_dct(DCpath))
    D = np.float32(STRUCT.coef_arrays[0])
    os.remove(path)
    os.remove(DCpath)
    return D

def zigzag(n):
    indexorder = sorted(((x,y) for x in range(n) for y in range(n)),
        key = lambda p: (p[0]+p[1], -p[1] if (p[0]+p[1]) % 2 else p[1]) )
    d = dict((index,n) for n,index in enumerate(indexorder))
    return np.array([d[(x,y)] for x in range(n) for y in range(n)]).reshape(n,n)


def init_scan(scan_mode , h, w):
    perm_scan = np.zeros((h, w, 8, 8))

    if scan_mode == 1 : #zig zag scan
        perm_scan[:] = zigzag(8)

    elif scan_mode == 2 : # inv_zig_zag
        perm_scan[:] = np.flipud(np.fliplr(zigzag(8)))

    else : # random
        for i in range(h):
            for j in range(w):
                perm_scan[i,j,:,:] = np.reshape(np.random.permutation(64), (8,8)).astype(int)

    return perm_scan

def get_robust(P1,P2,idx):
    size = P1.shape[0]*P1.shape[1]
    return np.all(np.reshape(P1[idx]==P2[idx], (size,-1)), axis=1)

def comp_rob_set_per_mode(pseudo_stego, im_name  , P1_STRUCTs, scan_array, m, quality):
    # m = pseudo-mode number
    (h, w) = pseudo_stego.shape[:2]
    robust_set = np.zeros((h*w))
    sets = P1_STRUCTs.keys()
    P1 = {}
    P1_path = {}
    P2_path = {}

    # Initialize paths and single-compressed DCTs
    for s in sets:
        P1[s] = np.copy(pseudo_stego) # THIS
        P1_path[s] = '/tmp/'+im_name+'_p1_' + s + '.jpg'
        P2_path[s] = '/tmp/'+im_name+'_p2_' + s + '.jpg'

    current_mode = scan_array==m

    # Embedding
    P1['p1'][current_mode] += 1
    P1['m1'][current_mode] -= 1

    P2_STRUCTs = {}
    P2 = {}
    current = {}
    for s in sets:
        # Recompress images
        P1_STRUCTs[s].coef_arrays[0][:] = np.copy(reshape_view_to_original(P1[s], P1_STRUCTs[s].coef_arrays[0]))
        P1_STRUCTs[s].write(P1_path[s])
        compress_image(P1_path[s], P2_path[s], quality)
        # Get double-compressed DCTs
        P2_STRUCTs[s] = jpeglib.to_jpegio(jpeglib.read_dct(P2_path[s]))
        P2[s] = view_as_blocks(np.copy(P2_STRUCTs[s].coef_arrays[0]), (8,8))
        # Check which embedding change survived recompression
        current[s] = get_robust(P1[s],P2[s], current_mode) # s survives recompression

    # Check that already processed modes are intact
    previous_modes = scan_array<m
    previous = {}
    previous['p1'] = get_robust(P2['p1'],P2['0'], previous_modes) # +1 doesn't change processed modes
    previous['m1'] = get_robust(P2['m1'],P2['0'], previous_modes) # -1 doesn't change processed modes

    # Compute the robust set for this mode
    robust_set[current['0']*current['m1']*previous['m1']] = 3 # -1, 0 possible
    robust_set[current['0']*current['p1']*previous['p1']] = 2 # 1, 0 possible
    robust_set[current['0']*current['p1']*current['m1']*previous['p1']*previous['m1']] = 1 # -1, 0, 1 possible

    # Undo the embedding changes
    for s in sets:
        P1[s][current_mode] = np.copy(pseudo_stego[current_mode])
        P1_STRUCTs[s].coef_arrays[0][:] = np.copy(reshape_view_to_original(P1[s], P1_STRUCTs[s].coef_arrays[0]))

    for s in sets:
        os.remove(P1_path[s])
        os.remove(P2_path[s])

    return robust_set

def comp_rob_set(C, Q, im_name,scan_array,quality):
    I = np.clip(np.round(decompress_image(C,Q)), 0, 255).astype(np.uint8)
    coverPath = '/tmp/' + im_name + '.jpg'
    Image.fromarray(I).save(coverPath, quality=quality)
    pseudo_stego = view_as_blocks(np.copy(C), (8,8))
    (h,w) = pseudo_stego.shape[:2]

    sets = ['p1', 'm1', '0']
    P1_STRUCTs = {}
    for s in sets:
        P1_STRUCTs[s] = jpeglib.to_jpegio(jpeglib.read_dct(coverPath))
        P1_STRUCTs[s].coef_arrays[0][:] = np.copy(C)

    whole_rob_set = np.zeros((h,w,8,8))
    for m in range(64):
        # Compute the robust set for no changes
        rob_set = comp_rob_set_per_mode(pseudo_stego, im_name , P1_STRUCTs, scan_array, m, quality)
        whole_rob_set[scan_array==m] = rob_set

    os.remove(coverPath)
    return whole_rob_set

def rob_embedding(C,Q,im_name, payload, scan_array, whole_rob_set, quality):
    I = np.clip(np.round(decompress_image(C,Q)), 0, 255).astype(np.uint8)
    coverPath = '/tmp/' + im_name + '.jpg'
    Image.fromarray(I).save(coverPath, quality=quality)
    P1_STRUCT = jpeglib.to_jpegio(jpeglib.read_dct(coverPath))
    P1_STRUCT.coef_arrays[0][:] = np.copy(C)
    pseudo_stego = view_as_blocks(np.copy(C), (8,8))
    cover = np.copy(pseudo_stego)
    sets = ['p1', 'm1', '0']
    P1_STRUCTs = {}
    for s in sets:
        P1_STRUCTs[s] = jpeglib.to_jpegio(jpeglib.read_dct(coverPath))
        P1_STRUCTs[s].coef_arrays[0][:] = np.copy(C)

    os.remove(coverPath)

    rhoP1, rhoM1 = compute_rhos(P1_STRUCT, i=0)
    rhoP1 = view_as_blocks(rhoP1, (8,8))
    rhoM1 = view_as_blocks(rhoM1, (8,8))
    nzAC = np.count_nonzero(pseudo_stego) - np.count_nonzero(pseudo_stego[:,:,0,0])

    # Compute probabilities map from costs map
    p_change_P1, p_change_M1 = Embedding_simulator.compute_proba(rhoP1, rhoM1, round(float(payload) * nzAC), pseudo_stego.size)

    # Set non robust coefficients to wet costs
    rhoP1[whole_rob_set==0] =  _WET_COST_ROBUST
    rhoM1[whole_rob_set==0] =  _WET_COST_ROBUST
    rhoM1[whole_rob_set==2] =  _WET_COST_ROBUST
    rhoP1[whole_rob_set==3] =  _WET_COST_ROBUST

    # Compute probabilities map from costs map
    p_change_P1_new, p_change_M1_new = Embedding_simulator.compute_proba(rhoP1, rhoM1, round(float(payload) * nzAC), pseudo_stego.size)

    # Check if there is enough non forbidden coefficients to embed the message
    target_message_size = round(float(payload) * nzAC)
    actual_message_size = Embedding_simulator.ternary_entropy(p_change_P1_new, p_change_M1_new)
    if (target_message_size - actual_message_size) / target_message_size > 0.01:
        print('Warning: Target payload is too big for image capacity : target = ' + str(target_message_size) + ', max embedding size = ' + str(actual_message_size))

    size_non_robust_mode = np.zeros(64)
    size_robust_mode = np.zeros(64)

    for m in range(64):
        size_non_robust_mode[m] = Embedding_simulator.ternary_entropy(p_change_P1[scan_array==m], p_change_M1[scan_array==m])
        size_robust_mode[m] = Embedding_simulator.ternary_entropy(p_change_P1_new[scan_array==m], p_change_M1_new[scan_array==m])

    rhoP1, rhoM1 = compute_rhos(P1_STRUCT, i=0)
    rhoP1 = view_as_blocks(rhoP1, (8,8))
    rhoM1 = view_as_blocks(rhoM1, (8,8))

    for m in range(64):
        # Compute the robust set for no changes
        rob_set = comp_rob_set_per_mode(pseudo_stego , im_name ,  P1_STRUCTs, scan_array, m, quality)
        whole_rob_set[scan_array==m] = rob_set

        # Set non robust coefficients to wet costs
        rhoP1[scan_array==m] +=  _WET_COST_ROBUST*(rob_set==0)
        rhoM1[scan_array==m] +=  _WET_COST_ROBUST*(rob_set==0)
        rhoM1[scan_array==m] +=  _WET_COST_ROBUST*(rob_set==2)
        rhoP1[scan_array==m] +=  _WET_COST_ROBUST*(rob_set==3)

        # Compute probabilities map from costs map
        p_change_P1, p_change_M1 = Embedding_simulator.compute_proba(rhoP1[scan_array==m], rhoM1[scan_array==m],size_robust_mode[m] , rhoP1[scan_array==m].size)

        # Simulate embedding
        pseudo_stego[scan_array==m] = Embedding_simulator.process(cover[scan_array==m], p_change_P1, p_change_M1)

        # Update single compressed DCTs
        for s in sets:
            view = view_as_blocks(P1_STRUCTs[s].coef_arrays[0], (8,8))
            view[scan_array==m] = np.copy(pseudo_stego[scan_array==m])

    pseudo_stego = reshape_view_to_original(pseudo_stego, I)
    whole_rob_set = reshape_view_to_original(whole_rob_set, I)
    return pseudo_stego, whole_rob_set

def Embed(imPath, quality, payload, scan=1):
    # scan=1 - Low-to-High
    # scan=2 - High-to-Low
    # scan=3 - Random
    if imPath[-4:] == 'jpeg' or imPath[-3:] == 'jpg':
        # already provided JPEG file
        coverPath = imPath
    else:
        # create single-compressed cover image
        coverPath = '.'.join(imPath.split('.')[:-1]) + '.jpg'
        compress_image(imPath, coverPath, quality)
    im_name = coverPath.split('/')[-1]
    STRUCT = jpeglib.to_jpegio(jpeglib.read_dct(coverPath))
    C = STRUCT.coef_arrays[0].astype(np.float32)
    Q = STRUCT.quant_tables[0]
    r, c = np.array(C.shape)//8
    scan_array = init_scan(scan , r , c)
    rob_set = comp_rob_set(C, Q, im_name, scan_array,quality)
    print('Robust set size:', np.sum(rob_set!=0)/np.size(rob_set))

    S, whole_rob_set = rob_embedding(C, Q, im_name, payload , scan_array , rob_set, quality)

    DS = recompress_coefficients(S, quality) # recompressed stego image
    robust_idx = whole_rob_set != 0
    if np.sum(DS[robust_idx] != S[robust_idx]) > 0:
        print('Warning: Some robust coefficients have changed during recompression!')
    if np.sum(S[~robust_idx]!=C[~robust_idx]) > 0: # Only useful with STC implementation
        print('Warning: Some non-robust coefficients have been embedded! Message is probably too big the robust set.')

    return S # return single-compressed stego image (before recompression)
