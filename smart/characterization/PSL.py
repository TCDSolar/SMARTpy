import numpy as np
import skimage as ski
from skimage.morphology import disk

import astropy.units as u


@u.quantity_input
def separate_polarities(im_map, feature_map, dilation_radius: u.Quantity[u.arcsec] = 2 * u.arcsec):
    """
    Separate feature masks into a negative mask and positive mask, and dilating both masks.

    Parameters
    ----------
    im_map : Map
        Processed SunPy magnetogram map.
    feature_map :
    """
    posmask = (feature_map > 0).astype(int)
    negmask = (feature_map < 0).astype(int)

    arcsec_to_pixel = ((im_map.scale[0] + im_map.scale[1]) / 2) ** (-1)
    dilation_radius = (np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix)

    dilated_posmask = ski.morphology.binary_dilation(posmask, disk(dilation_radius))
    dilated_negmask = ski.morphology.binary_dilation(negmask, disk(dilation_radius))
    return dilated_posmask, dilated_negmask


def PSL_length(dilated_posmask, dilated_negmask):
    PSL_mask = ((dilated_posmask + dilated_negmask) > 1).astype(int)
    PSL_thinmask = ski.morphology.thin(PSL_mask)
    PSL_length = np.sum(PSL_thinmask)
    return PSL_length
