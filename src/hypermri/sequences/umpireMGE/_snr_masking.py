import numpy as np
from hypermri.utils.utils_general import estimate_noise


def SNR(magnitude, factor=0.05, k_space_array = None, plot = False):
    if not isinstance(k_space_array, type(None)):
        stdN, idx= getStddevNoiseKSpace(k_space_array, factor, return_in_image_space=True, plot = plot)
    elif magnitude.ndim == 2:
        stdN, idx= getStddevNoise2D(magnitude, factor)
    elif magnitude.ndim == 3:
        stdN, idx = getStddevNoise3D(magnitude, factor)
    else:
        raise TypeError("Only 2D and 3D arrays allowed.")

    magnitude_scaled = magnitude / stdN

    return magnitude_scaled, idx, stdN


def createMaskFromSliceIdx(arr, idx):
    """To dislpay noisy are selection"""
    binary_mask = np.ones_like(arr, dtype="int")
    binary_mask[idx] = 0
    return binary_mask


def createMaskFromThreshold(arr, thr):
    """To mask SNR image"""
    return np.ma.masked_where(arr < thr, arr).mask


def getStddevNoise2D(image_arr, factor=0.05, mode="edge"):
    """
    Only tested for Magnitude Images!

    Assumed image extensions:

            0,0---------0,B
             |           |
             |           |
             |           |
            A,0---------A,B

    Either finds corner with lowest avg. signal or edge with lowest avg. signal.
    """
    A, B = image_arr.shape

    int_gt0 = lambda x: int(x) if x > 1 else int(1)

    H = int_gt0(A * factor)
    W = int_gt0(B * factor)

    if mode.lower() == "corner":
        # define all four corner slice indices
        c1 = slice(0, H), slice(0, W)  # top left
        c2 = slice(0, H), slice(B - W, B)  # top right
        c3 = slice(A - H, A), slice(0, W)  # bottom left
        c4 = slice(A - H, A), slice(B - W, B)  # bottom right
        idx_list = [c1, c2, c3, c4]

    elif mode.lower() == "edge":
        # define all four edge slice indices
        l1 = slice(0, H), slice(None)  # top
        l2 = slice(None), slice(0, W)  # left
        l3 = slice(A - H, A), slice(None)  # bottom
        l4 = slice(None), slice(B - W, B)  # right
        idx_list = [l1, l2, l3, l4]

    else:
        raise NameError(f"{mode} is not an option, choose from: 'corner', 'edge'.")

    # get the mean signal of each of the 4 defined regions
    meanSignals = np.zeros(4)

    for i, idx in enumerate(idx_list):
        meanSignals[i] = np.mean(np.abs(image_arr[idx]))

    # determine the region with the least signal
    i_leastSignal = np.argmin(np.abs(meanSignals))

    # pick the slice index which yielded the lowest signal
    idx_leastSignal = idx_list[i_leastSignal]

    # get 'noisiest' image area
    noisy_area = image_arr[idx_leastSignal]

    return np.std(noisy_area)/0.66, idx_leastSignal


def getStddevNoise3D(image_arr, factor=0.05, factor_depth=0.1, mode="surface"):
    """Find noisy image area and return stddev(Noise).

    Only tested for Magnitude Images!

    Assumed image extensions:

            0,0---------0,B
             |           |
             |           |   and C is depth
             |           |
            A,0---------A,B

    Go through all six surfaces (with relative width dimension*factor) of the cuboid
    and find the one with the least signal.
    """
    A, B, C = image_arr.shape

    int_gt0 = lambda x: int(x) if x > 1 else int(1)

    H = int_gt0(A * factor)
    W = int_gt0(B * factor)
    D = int_gt0(C * factor_depth)

    if mode.lower() == "surface":
        f1 = slice(0, H), slice(None), slice(None)
        f2 = slice(A - H, A), slice(None), slice(None)
        f3 = slice(None), slice(0, W), slice(None)
        f4 = slice(None), slice(B - W, B), slice(None)
        f5 = slice(None), slice(None), slice(0, D)
        f6 = slice(None), slice(None), slice(C - D, C)
        idx_list = [ f1, f2,f3, f4, f5, f6]
    else:
        raise NameError(f"{mode} is not an option, choose from: 'surface'.")

    # get the mean signal of each of the 4 defined regions
    meanSignals = np.zeros(6)

    for i, idx in enumerate(idx_list):
        meanSignals[i] = np.mean(np.abs(image_arr[idx]))

    # determine the region with the least signal
    i_leastSignal = np.argmin(meanSignals)

    # use the slice index which yielded the lowest signal to get the 'noisiest' image area
    noisy_area = image_arr[idx_list[i_leastSignal]]

    return np.std(noisy_area)/0.66, idx_list[i_leastSignal]


def getStddevNoiseKSpace(k_space_arr, factor=0.05, factor_depth=0.1, return_in_image_space=True, min_rel_signal=20e-2, plot = False):
    """Find noisy image area and return stddev(Noise) from a k-space image.

    Assumed image extensions:

            0,0---------0,B
             |           |
             |           |   and C is depth
             |           |
            A,0---------A,B

    Go through all six surfaces (with relative width dimension*factor) of the cuboid
    and find the one with the least signal, then calculate noise and return.

    """
    abs_k = np.abs(k_space_arr)

    # crop away zero-padded planes along all spatial axes before searching for the noisiest surface
    crop = [slice(None)] * 4
    for axis in (1, 2, 3):
        other_axes = tuple(a for a in range(4) if a != axis)
        plane_mean = abs_k.mean(axis=other_axes)
        # reference only from non-(near-)zero planes
        ref = np.median(plane_mean[plane_mean > 1e-6 * plane_mean.max()])
        valid = np.flatnonzero(plane_mean > min_rel_signal * ref)
        crop[axis] = slice(valid[0], valid[-1] + 1)
    crop = tuple(crop)

    k_crop = k_space_arr[crop]
    abs_k = abs_k[crop]
    _, A, B, C = k_crop.shape

    int_gt0 = lambda x: int(x) if x > 1 else int(1)
    H = int_gt0(A * factor)
    W = int_gt0(B * factor)
    D = int_gt0(C * factor_depth)

    idx_list = [
        np.s_[:, :H, :, :], np.s_[:, -H:, :, :],
        np.s_[:, :, :W, :], np.s_[:, :, -W:, :],
        np.s_[:, :, :, :D], np.s_[:, :, :, -D:],
    ]

    meanSignals = np.array([np.mean(abs_k[idx]) for idx in idx_list])
    i_leastSignal = np.argmin(meanSignals)
    noisy_area = k_crop[idx_list[i_leastSignal]]


    #reorder k-space to get format necessary to use estimate_noise()
    if i_leastSignal == 2 or i_leastSignal == 3:
        reordered = np.transpose(noisy_area, (0,2,1,3))
    elif i_leastSignal == 4 or i_leastSignal == 5:
        reordered = np.transpose(noisy_area, (0,3,1,2))
    else:
        reordered = noisy_area

    expanded_dims = np.expand_dims(reordered,-1)
    expanded_dims = np.expand_dims(expanded_dims.data,-1)

    noise = estimate_noise(expanded_dims,  slice_ind = 0, signal_domain="temporal", plot=plot)
    if return_in_image_space:
        return noise * np.sqrt(A * B * C), (crop, idx_list[i_leastSignal])
    else:
        return noise, (crop, idx_list[i_leastSignal])
