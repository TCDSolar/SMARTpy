import numpy as np
import skimage as ski
from matplotlib import colors
from skimage.morphology import disk

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk


def map_threshold(im_map):
    """
    Creating a map from a fits file and processing said map.

    Parameters
    ----------
    file : Map
        Unprocessed magnetogram map.

    Returns
    -------
    im_map : Map
        Processed magnetogram map.

    """
    im_map.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(im_map))] = np.nan
    im_map.cmap.set_bad("k")
    im_map.plot_settings["norm"] = colors.Normalize(vmin=-200, vmax=200)
    return im_map


@u.quantity_input
def smooth_los_threshold(
    im_map: Map,
    thresh: u.Quantity[u.Gauss] = 100 * u.Gauss,
    dilation_radius: u.Quantity[u.arcsec] = 2 * u.arcsec,
    sigma: u.Quantity[u.arcsec] = 10 * u.arcsec,
    min_size: u.Quantity[u.arcsec] = 2250 * u.arcsec,
):
    """

    We apply Smoothing, a noise Threshold, and an LOS correction, respectively, to the data.

    Parameters
    ----------
    im_map : Map
        Processed SunPy magnetogram map.
    thresh : int, optional
        Threshold value to identify regions of interest (default is 100 Gauss).
    dilation_radius : int, optional
        Radius of the disk for binary dilation (default is 2 arcsecs).
    sigma : int, optional
        Standard deviation for Gaussian smoothing (default is 10 arcsecs).
    min_size : int, optional
        Minimum size of regions to keep in final mask (default is 2250 arcsecs**2).

    Returns
    -------
    smooth_map : Map
        Map after applying Gaussian smoothing.
    filtered_labels : numpy.ndarray
        2D array with each pixel labelled.
    mask_sizes : numpy.ndarray
        Boolean array indicating the sizes of each labeled region.

    """

    arcsec_to_pixel = ((im_map.scale[0] + im_map.scale[1]) / 2) ** (-1)
    dilation_radius = (np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix)
    sigma = (np.round(sigma * arcsec_to_pixel)).to_value(u.pix)
    min_size = (np.round(min_size * arcsec_to_pixel)).to_value(u.pix)

    cosmap_data = cosine_correction(im_map)

    negmask = cosmap_data < -thresh
    posmask = cosmap_data > thresh
    mask = negmask | posmask

    dilated_mask = ski.morphology.binary_dilation(mask, disk(dilation_radius))
    smoothed_data = ski.filters.gaussian(np.nan_to_num(cosmap_data) * ~dilated_mask, sigma)
    smooth_map = Map(smoothed_data, im_map.meta)

    labels = ski.measure.label(dilated_mask)
    label_sizes = np.bincount(labels.ravel())
    mask_sizes = label_sizes > min_size
    mask_sizes[0] = 0
    filtered_labels = mask_sizes[labels]
    return smooth_map, filtered_labels, mask_sizes


def get_cosine_correction(im_map: Map):
    """
    Get the cosine map and off-limb pixel map using WCS.

    Parameters
    ----------
    im_map : Map
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

    coordinates = all_coordinates_from_map(im_map)
    x = coordinates.Tx.to(u.arcsec)
    y = coordinates.Ty.to(u.arcsec)

    radial_angle = np.arccos(np.cos(x) * np.cos(y))
    cos_cor_ratio = (radial_angle / im_map.rsun_obs).value
    cos_cor_ratio = np.clip(cos_cor_ratio, -1, 1)

    cos_correction = 1 / (np.cos(np.arcsin(cos_cor_ratio)))

    on_disk = coordinate_is_on_solar_disk(coordinates)
    cos_correction[~on_disk] = 1
    off_limb = np.zeros_like(cos_correction)
    off_limb[on_disk] = 1

    return cos_correction, radial_angle, off_limb


def cosine_correction(im_map: Map, cosmap=None):
    """
    Perform magnetic field cosine correction.

    Parameters
    ----------
    inmap : Map
        Processed SunPy magnetogram map.
    cosmap : numpy.ndarray, optional
        An array of the cosine correction factors for each pixel.
        If not provided, computed using get_cosine_correction.

    Returns
    -------
    corrected_data : numpy.ndarray
        The magnetic field data after applying the cosine correction (units = Gauss).

    """
    if cosmap is None:
        cosmap = get_cosine_correction(im_map)[0]

    scale = (im_map.scale[0] + im_map.scale[1]) / 2

    angle_limit = (scale / im_map.rsun_obs).value
    cos_limit = 1 / np.cos(angle_limit)
    cosmap = np.clip(cosmap, None, cos_limit)

    corrected_data = im_map.data * cosmap * u.Gauss
    return corrected_data
