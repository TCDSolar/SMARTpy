from sunpy.coordinates import propagate_with_solar_surface
from sunpy.map import Map

__all__ = ["diff_rotation"]


def diff_rotation(ref_map: Map, im_map: Map):
    """
    Differentially rotate input map to reference map.

    Parameters
    ----------
    ref_map : `~sunpy.map.Map`
        Reference map.
    im_map : `~sunpy.map.Map`
        Map to be reprojected.

    Returns
    -------
    diff_map : `~sunpy.map.Map`
        Differentially rotated map.
    """

    with propagate_with_solar_surface():
        diff_map = im_map.reproject_to(ref_map.wcs)

    return diff_map
