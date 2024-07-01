import numpy as np
import skimage as ski
from matplotlib import colors
from skimage.morphology import disk

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk


def map_processing(file):
    """
    Creating a map from a fits file and processing said map.

    Parameters
    ----------
    file : fits
        fits file to turn into map.

    Returns
    -------
    im_map : sunpy.map.Map
        processed map created from fits file.

    """
    im_map = Map(file)
    if im_map.meta["CROTA2"] >= 100:
        data = np.flip(im_map.data, 1)[::-1]
        im_map = Map(data, im_map.meta)
    im_map.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(im_map))] = np.nan
    im_map.cmap.set_bad("k")
    im_map.plot_settings["norm"] = colors.Normalize(vmin=-200, vmax=200)
    return im_map


def STL(im_map):
    """

    We apply Smoothing, a noise Threshold, and an LOS correction, respectively, to the data.

    Parameters
    ----------
    im_map : sunpy.map.Map
        Processed SunPy magnetogram map.

    Returns
    -------
    smooth_map : sunpy.map.Map
        Map after applying Gaussian smoothing.
    filtered_labels : numpy.ndarray
        2D array with each pixel labelled.
    mask_sizes : numpy.ndarray
        Boolean array indicating the sizes of each labeled region.

    """

    cosmap = ar_cosmap(im_map)[0]
    cosmap_data = cosine_correction(im_map, cosmap)[0]

    thresh = 100
    negmask = cosmap_data < -thresh
    posmask = cosmap_data > thresh
    mask = negmask | posmask

    dilated_mask = ski.morphology.binary_dilation(mask, disk(5))
    smoothed_data = ski.filters.gaussian(np.nan_to_num(cosmap_data) * ~dilated_mask, sigma=16)
    smooth_map = Map(smoothed_data, im_map.meta)

    labels = ski.measure.label(dilated_mask)
    min_size = 5000
    label_sizes = np.bincount(labels.ravel())
    mask_sizes = label_sizes > min_size
    mask_sizes[0] = 0
    filtered_labels = mask_sizes[labels]
    return smooth_map, filtered_labels, mask_sizes


def ar_cosmap(im_map):
    """
    Get the cosine map and off-limb pixel map using WCS.

    Parameters
    ----------
    im_map : sunpy.map.Map
        Processed SunPy magnetogram map.

    Returns
    -------
    cos_cor : numpy.ndarray
        Array of cosine correction factors for each pixel. Values greater than a threshold (edge) are set to 1.
    d_angular : numpy.ndarray
        Array of angular distances from the disk center in radians.
    off_limb : numpy.ndarray
        Binary array where pixels on the disk are 1 and pixels off the disk are 0.
    """

    ## Take off an extra percent from the disk to get rid of limb effects
    edge = 0.99

    x, y = np.meshgrid(*[np.arange(v.value) for v in im_map.dimensions]) * u.pixel
    hp_coords = im_map.pixel_to_world(x, y)
    xx = hp_coords.Tx.value
    yy = hp_coords.Ty.value
    d_radial = np.sqrt((xx**2.0) + (yy**2.0))

    cos_cor = np.copy(d_radial)
    cos_cor_ratio = cos_cor / im_map.meta["RSUN_OBS"]

    cos_cor_ratio = np.clip(cos_cor_ratio, -1, 1)
    d_angular = np.arcsin(cos_cor_ratio)
    cos_cor = 1.0 / np.cos(d_angular)

    off_disk = np.where(d_radial > (im_map.meta["RSUN_OBS"] * edge))
    cos_cor[off_disk] = 1.0

    off_limb = np.copy(d_radial)
    off_disk_mask = np.where(d_radial >= (im_map.meta["RSUN_OBS"] * edge))
    off_limb[off_disk_mask] = 0.0
    on_disk_mask = np.where(d_radial < (im_map.meta["RSUN_OBS"] * edge))
    off_limb[on_disk_mask] = 1.0

    return cos_cor, d_angular, off_limb


def cosine_correction(im_map, cosmap):
    """
    Perform magnetic field cosine correction.

    Parameters
    ----------
    inmap : sunpy.map.Map
        Processed SunPy magnetogram map.
    cosmap : numpy.ndarray
        An array of the cosine correction factors for each pixel.

    Returns
    -------
    corrected_data : numpy.ndarray
        The magnetic field data after applying the cosine correction.
    cosmap : numpy.ndarray
        The cosine correction factors, limited to the max value allowed.

    """
    angle_limit = np.arcsin(1.0 - im_map.meta["CDELT1"] / im_map.meta["RSUN_OBS"])
    cos_limit = 1.0 / np.cos(angle_limit)
    cosmap_limit = np.where((cosmap) > cos_limit)
    cosmap[cosmap_limit] = cos_limit
    return im_map.data * cosmap, cosmap
