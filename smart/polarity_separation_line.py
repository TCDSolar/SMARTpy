import numpy as np
import skimage as ski
from skimage.morphology import disk

import astropy.units as u

# todo docstrings Friday morning


@u.quantity_input
def separate_polarities(im_map, feature_mask, dilation_radius: u.Quantity[u.arcsec] = 3 * u.arcsec):
    feature_map = feature_mask * im_map.data

    posmask = (feature_map > 0).astype(int)
    negmask = (feature_map < 0).astype(int)

    arcsec_to_pixel = 1 / ((im_map.scale[0] + im_map.scale[1]) / 2)
    dilation_radius = (np.round(dilation_radius * arcsec_to_pixel)).to_value(u.pix)

    dilated_posmask = ski.morphology.binary_dilation(posmask, disk(dilation_radius))
    dilated_negmask = ski.morphology.binary_dilation(negmask, disk(dilation_radius))
    return dilated_negmask, dilated_posmask


def polarity_separation_line_length(dilated_posmask, dilated_negmask):
    psl_mask = dilated_posmask * dilated_negmask
    psl_thinmask = ski.morphology.thin(psl_mask)
    psl_length = np.sum(psl_thinmask)
    return psl_length, psl_thinmask


@u.quantity_input
def get_strong_gradient_length(
    im_map, feature_mask, psl_thinmask, threshold: u.Quantity[u.Gauss / u.Mm] = 50 * u.Gauss / u.Mm
):
    feature_map = feature_mask * im_map.data
    y_gradient, x_gradient = np.gradient(feature_map) * (u.Gauss / u.pix)
    gradient = np.sqrt(x_gradient**2 + y_gradient**2)

    meters_per_arcsec = im_map.rsun_meters * im_map.rsun_obs
    meters_per_pixel = meters_per_arcsec * (im_map.scale[0] + im_map.scale[1]) / 2

    gradient = (gradient / meters_per_pixel).to(u.Gauss / u.Mm)
    strong_gradient_mask = (gradient > threshold) & (psl_thinmask > 0)
    strong_gradient_length = np.sum(strong_gradient_mask)
    return strong_gradient_length
