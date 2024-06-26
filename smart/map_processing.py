import numpy as np
import skimage as ski
from matplotlib import colors
from skimage.morphology import disk

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
        Processed sunpy magnetogram map.

    Returns
    -------
    smooth_map : sunpy.map.Map
        Map after applying Gaussian smoothing.
    filtered_labels : numpy.ndarray
        2D array with each pixel labelled.
    mask_sizes : numpy.ndarray
        Boolean array indicating the sizes of each labeled region.

    """
    thresh = 100
    negmask = im_map.data < -thresh
    posmask = im_map.data > thresh
    mask = negmask | posmask

    dilated_mask = ski.morphology.binary_dilation(mask, disk(5))
    smoothed_data = ski.filters.gaussian(np.nan_to_num(im_map.data) * ~dilated_mask, sigma=16)
    smooth_map = Map(smoothed_data, im_map.meta)

    labels = ski.measure.label(dilated_mask)
    min_size = 5000
    label_sizes = np.bincount(labels.ravel())
    mask_sizes = label_sizes > min_size
    mask_sizes[0] = 0
    filtered_labels = mask_sizes[labels]
    return smooth_map, filtered_labels, mask_sizes
