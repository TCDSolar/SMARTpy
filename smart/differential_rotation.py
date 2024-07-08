import numpy as np
from matplotlib import colors

from sunpy.coordinates import propagate_with_solar_surface
from sunpy.map import all_coordinates_from_map, coordinate_is_on_solar_disk


def diff_rotation(im_map, delta_im_map):
    """
    Performing differential rotation on a map in order to to correct for feature
    motions due to solar rotation.

    Parameters
    ----------
    im_map : sunpy.map.Map
        Processed SunPy magnetogram map.
    delta_im_map : sunpy.map.Map
        Processed SunPy magnetogram taken at time Δt before im_map.

    Returns
    -------
    diff_map : sunpy.map.Map
        delta_im_map differentially rotated to match im_map.

    """

    with propagate_with_solar_surface():
        diff_map = delta_im_map.reproject_to(im_map.wcs)

    diff_map.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(diff_map))] = np.nan
    diff_map.cmap.set_bad("k")
    diff_map.plot_settings["norm"] = colors.Normalize(vmin=-200, vmax=200)
    return diff_map
