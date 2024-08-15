import numpy as np

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk

from smart.segmentation.differential_rotation import diff_rotation


def cosine_weighted_area_map(im_map: Map, feature_mask):
    """
    Calculate the cosine-weighted area map for a feature and determine the feature's total area (in square metres).

    Parameters
    ----------
    im_map : Map
        Processed SunPy magnetogram map.
    feature_mask : numpy.ndarray
        Binary mask where feature pixels = 1 and background pixels = 0.

    Returns
    -------
    total_area : astropy.units.quantity.Quantity
        The total area of the feature in square metres.
    area_map : astropy.units.quantity.Quantity
        The area map of the feature in square metres.
    """
    coords = all_coordinates_from_map(im_map)

    cos_weight = np.cos(coords.Tx.to(u.rad)) * np.cos(coords.Ty.to(u.rad))
    cos_weight[cos_weight < 0] = 0

    m_per_arcsec = im_map.rsun_meters / im_map.rsun_obs
    pixel_area = (im_map.scale[0] * m_per_arcsec) * (im_map.scale[1] * m_per_arcsec)
    pixel_area = pixel_area * u.pix**2

    area_map = feature_mask * pixel_area * cos_weight

    total_area = np.sum(area_map)

    return total_area, area_map


def extract_features(sorted_labels):
    """
    Extract binary masks for each feature found in index_and_grow_mask's sorted_labels.

    Parameters
    ----------
    sorted_labels : numpy.ndarray
        An array where each unique label corresponds to a different feature on the solar disk.

    Returns
    -------
    feature_masks : list
        A list containing a numpy.ndarray binary mask for each identified feature.
    """
    unique_labels = np.unique(sorted_labels)
    unique_labels = unique_labels[unique_labels != 0]

    feature_masks = []
    for label_value in unique_labels:
        feature_mask = (sorted_labels == label_value).astype(int)
        feature_masks.append(feature_mask)

    return feature_masks


def dB_dt(current_map: Map, previous_map: Map):
    """
    A magnetogram differentially rotated to time 't' is subtracted from a processed magnetogram from time 't', and the resultant map is divided
    by their time separation, yielding a map of the temporal change in field strength.

    Parameters
    ----------
    current_map : Map
        Processed SunPy magnetogram map from time 't'.
    previous_map : Map
        Processed SunPy magnetogram map from time 't - delta_t'.

    Returns
    -------
    dB_dt : Map
        Map showcasing the change in magnetic field strength over time.
    dB : Quantity
        The change in magnetic field strength.
    dt : Quantity
        The time interval over which the change in magnetic field strength was measured.
    """
    diff_map = diff_rotation(current_map, previous_map)

    dB = (current_map.data - diff_map.data) * u.Gauss
    dt = (current_map.date - previous_map.date).to(u.s)

    dB_dt = Map(dB / dt, current_map.meta)
    dB_dt.data[~coordinate_is_on_solar_disk(all_coordinates_from_map(dB_dt))] = np.nan
    dB_dt.cmap.set_bad("k")
    return dB_dt, dB, dt


def get_flux_emergence_rates(im_map, sorted_labels, dB_dt, dt):
    """
    Calculate the flux emergence rate for each identified feature on the magnetogram.

    Parameters
    ----------
    im_map : Map
        Processed SunPy magnetogram map.
    sorted_labels : numpy.ndarray
        An array where each unique label corresponds to a different feature on the solar disk.
    dB_dt : Map
        Map showcasing the change in magnetic field strength over time.
    dt : Quantity
        The time interval over which the change in magnetic field strength was measured.

    Returns
    -------
    emergence_rates : list of Quantity
        A list containing values for the flux emergence rate for each labelled feature in Gauss per second.
    """
    feature_masks = extract_features(sorted_labels)

    emergence_rates = []
    for feature_mask in feature_masks:
        area_map, total_area = cosine_weighted_area_map(im_map, feature_mask)

        magnetic_flux = np.nansum(dB_dt.data * area_map)

        flux_emergence_rate = magnetic_flux / dt

        emergence_rates.append(flux_emergence_rate)

    return emergence_rates


def get_properties(im_map, dB_dt, dt, sorted_labels):
    """
    Calculate various properties from each detected feature.

    Parameters
    ----------
    im_map : Map
        Processed SunPy magnetogram map.
    dB_dt : Map
        Map showcasing the change in magnetic field strength over time.
    dt : Quantity
        The time interval over which the change in magnetic field strength was measured.
    sorted_labels : numpy.ndarray
        An array where each unique label corresponds to a different feature on the solar disk.

    Returns
    -------
    properties : list
        A list containing properties related to each individual feature
    """

    feature_masks = extract_features(sorted_labels)

    properties = []
    for i, feature_mask in enumerate(feature_masks, start=1):
        total_area, area_map = cosine_weighted_area_map(im_map, feature_mask)

        dBdt_data = dB_dt.data * u.G

        magnetic_flux = np.nansum(dBdt_data * area_map)
        flux_emergence_rate = magnetic_flux / dt

        B = im_map.data * feature_mask * u.G
        B_mean = np.nanmean(B)
        B_std = np.nanstd(B)
        B_min = np.nanmin(B)
        B_max = np.nanmax(B)

        flux_pos = np.nansum(B[B > 0] * area_map[B > 0])
        flux_neg = np.nansum(B[B < 0] * area_map[B < 0])
        flux_uns = np.nansum(np.abs(B) * area_map)
        flux_imb = (flux_pos - flux_neg) / (flux_pos + flux_neg)

        properties.append(
            {
                "feature label": i,
                "flux emergence rate": flux_emergence_rate.to(u.Wb / u.s),
                "mean B": B_mean.to(u.G),
                "std B": B_std.to(u.G),
                "minimum B": B_min.to(u.G),
                "maximum B": B_max.to(u.G),
                "positive flux": flux_pos.to(u.Wb),
                "negative flux": flux_neg.to(u.Wb),
                "unsigned flux": flux_uns.to(u.Wb),
                "flux imbalance": flux_imb,
                "total area": total_area.to(u.m**2),
            }
        )

    return properties
