from sunpy.coordinates import propagate_with_solar_surface
from sunpy.map import Map


def diff_rotation(im_map: Map, delta_im_map: Map):
    """
    Performing differential rotation on a map in order to to correct for feature
    motions due to solar rotation.

    Parameters
    ----------
    im_map : Map
        Processed SunPy magnetogram map.
    delta_im_map : Map
        Processed SunPy magnetogram taken at time Δt before im_map.

    Returns
    -------
    diff_map : Map
        delta_im_map differentially rotated to match im_map.

    """

    with propagate_with_solar_surface():
        diff_map = delta_im_map.reproject_to(im_map.wcs)

    return diff_map, delta_im_map.date
