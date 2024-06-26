import numpy as np
from matplotlib import colors

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
from sunpy.map import all_coordinates_from_map, coordinate_is_on_solar_disk, make_fitswcs_header


def diff_rotation(im_map):
    """
    Performing differential rotation (96 mins) on a map in order to to correct for feature
    motions due to solar rotation.

    Parameters
    ----------
    im_map : sunpy.map.Map
        Processed SunPy magnetogram map.

    Returns
    -------
    diff_map : sunpy.map.Map
        Input map with differential rotation of 96mins.

    """
    in_time = im_map.date
    out_time = in_time + 96 * u.min
    out_frame = Helioprojective(observer="earth", obstime=out_time, rsun=im_map.coordinate_frame.rsun)

    ##############################################################################
    # Construct a WCS object for the output map.  If one has an actual ``Map``
    # object at the desired output time (e.g., the actual AIA observation at the
    # output time), one can use the WCS object from that ``Map`` object (e.g.,
    # ``mymap.wcs``) instead of constructing a custom WCS.
    """So should I have a function that searches for a magnetogram at the desired time,
    and also the one at time t - delta_t ?"""

    out_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=out_frame)
    header = make_fitswcs_header(im_map.data.shape, out_center, scale=u.Quantity(im_map.scale))
    out_wcs = WCS(header)

    with propagate_with_solar_surface():
        diff_map = im_map.reproject_to(out_wcs)

    diff_map.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(diff_map))] = np.nan
    diff_map.cmap.set_bad("k")
    diff_map.plot_settings["norm"] = colors.Normalize(vmin=-200, vmax=200)
    return diff_map
