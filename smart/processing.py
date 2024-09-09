import numpy as np
import skimage as ski
from matplotlib import colors
from skimage.morphology import disk

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

__all__ = [
    "map_threshold",
    "calculate_cosine_correction",
    "cosine_correct_data",
    "smooth_los_threshold",
    "smart_prep",
]


def map_threshold(im_map):
    """
    Set off disk pixels to black and clip the vmin and vmax of the map.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Unprocessed magnetogram map.

    Returns
    -------
    im_map : `~sunpy.map.Map`
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
    dilation_radius: u.Quantity[u.arcsec] = 5 * u.arcsec,
    sigma: u.Quantity[u.arcsec] = 10 * u.arcsec,
    min_size: u.Quantity[u.arcsec] = 2250 * u.arcsec,
):
    """

    We apply Smoothing, a noise Threshold, and an LOS correction, respectively, to the data.

    Parameters
    ----------
    im_map : ``~sunpy.map.Map``
        Processed SunPy magnetogram map.
    thresh : `int`, optional
        Threshold value to identify regions of interest (default is 100 Gauss).
    dilation_radius : `int`, optional
        Radius of the disk for binary dilation (default is 2 arcsecs).
    sigma : `int`, optional
        Standard deviation for Gaussian smoothing (default is 10 arcsecs).
    min_size : `int`, optional
        Minimum size of regions to keep in final mask (default is 2250 arcsecs**2).

    Returns
    -------
    smooth_map : `~sunpy.map.Map`
        Map after applying Gaussian smoothing.
    filtered_labels : `numpy.ndarray`
        2D array with each pixel labelled.
    mask_sizes : `~numpy.ndarray`
        Boolean array indicating the sizes of each labeled region.

    """

    arcsec_to_pixel = ((im_map.scale[0] + im_map.scale[1]) / 2) ** (-1)
    dilation_radius = (np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix)
    sigma = (np.round(sigma * arcsec_to_pixel)).to_value(u.pix)
    min_size = (np.round(min_size * arcsec_to_pixel)).to_value(u.pix)

    cosmap_data = cosine_correct_data(im_map)

    negmask = cosmap_data < -thresh
    posmask = cosmap_data > thresh
    mask = negmask | posmask

    dilated_mask = ski.morphology.binary_dilation(mask, disk(dilation_radius))
    smoothed_data = ski.filters.gaussian(np.nan_to_num(cosmap_data) * dilated_mask, sigma)
    smooth_map = Map(smoothed_data, im_map.meta)

    labels = ski.measure.label(dilated_mask)
    label_sizes = np.bincount(labels.ravel())
    mask_sizes = label_sizes > min_size
    mask_sizes[0] = 0
    filtered_labels = mask_sizes[labels]
    return smooth_map, filtered_labels, mask_sizes


def calculate_cosine_correction(im_map: Map):
    """
    Find the cosine correction values for on-disk pixels.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.

    Returns
    -------
    cos_correction : `~numpy.ndarray`
        Array of cosine correction factors for each pixel. Values greater than a threshold (edge) are set to 1.

    """

    coordinates = all_coordinates_from_map(im_map)
    on_disk = coordinate_is_on_solar_disk(coordinates)

    cos_correction = np.ones_like(im_map.data)

    radial_angle = np.arccos(np.cos(coordinates.Tx[on_disk]) * np.cos(coordinates.Ty[on_disk]))
    cos_cor_ratio = (radial_angle / im_map.rsun_obs).value
    cos_cor_ratio = np.clip(cos_cor_ratio, -1, 1)

    cos_correction[on_disk] = 1 / (np.cos(np.arcsin(cos_cor_ratio)))

    return cos_correction


def cosine_correct_data(im_map: Map, cosmap=None):
    """
    Perform magnetic field cosine correction.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.
    cosmap : `numpy.ndarray`, optional
        An array of the cosine correction factors for each pixel.
        If not provided, computed using calculate_cosine_correction.

    Returns
    -------
    corrected_data : `~numpy.ndarray`
        The magnetic field data after applying the cosine correction (units = Gauss).

    """
    if cosmap is None:
        cosmap = calculate_cosine_correction(im_map)

    scale = (im_map.scale[0] + im_map.scale[1]) / 2

    angle_limit = (scale / im_map.rsun_obs).value
    cos_limit = 1 / np.cos(angle_limit)
    cosmap = np.clip(cosmap, None, cos_limit)

    corrected_data = im_map.data * cosmap * u.Gauss
    return corrected_data


def smart_prep(im_map):
    """
    Prepare map for use in segmentation and characterization processes.

    Parameters
    ----------
    im_map : `~sunpy.map.Map`
        Unprocessed SunPy magnetogram map.

    Returns
    -------
    thresholded_map : `~sunpy.map.Map`
        Processed SunPy magnetogram map.
    cos_correctiom : `~numpy.ndarray`
        Array of cosine correction factors for each pixel.

    """
    thresholded_map = map_threshold(im_map)
    smooth_map, *_ = smooth_los_threshold(thresholded_map)
    cos_correction = calculate_cosine_correction(smooth_map)
    return thresholded_map, cos_correction
