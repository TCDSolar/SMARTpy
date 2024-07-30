import numpy as np
import pytest

import astropy.units as u

from sunpy.map import Map, all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.map.mapbase import GenericMap

from smart.segmentation.map_processing import (
    cosine_correction,
    get_cosine_correction,
    map_threshold,
    smooth_los_threshold,
)


@pytest.fixture
def hmi_nrt():
    return "https://solmon.dias.ie/data/2024/06/06/HMI/fits/hmi.m_720s_nrt.20240606_230000_TAI.3.magnetogram.fits"


@pytest.fixture
def mag_map_sample():
    return map_threshold(Map(hmi_nrt()))


@pytest.fixture
def create_fake_map(value, shape=((4098, 4098))):
    fake_data = np.ones(shape) * value
    return Map(fake_data, mag_map_sample().meta)


def test_map_threshold(im_map):
    processed_map = map_threshold(im_map)

    assert isinstance(processed_map, GenericMap), "Result is not a SunPy Map."

    coordinates = all_coordinates_from_map(processed_map)
    on_solar_disk = coordinate_is_on_solar_disk(coordinates)
    assert np.all(np.isnan(processed_map.data[~on_solar_disk])), "Off-disk NaN values not set correctly."


def test_get_cosine_correction_shape(im_map):
    cos_cor, d_angular, off_limb = get_cosine_correction(im_map)
    assert cos_cor.shape == im_map.data.shape, "cos_cor shape != im_map.data.shape"
    assert d_angular.shape == im_map.data.shape, "d_angular shape != im_map.data.shape"
    assert off_limb.shape == im_map.data.shape, "off_limb shape != im_map.data.shape"


def test_get_cosine_correction_limits(im_map):
    cos_cor, d_angular, off_limb = get_cosine_correction(im_map)

    edge = 0.99
    coordinates = all_coordinates_from_map(im_map)
    x = coordinates.Tx.to(u.arcsec)
    y = coordinates.Ty.to(u.arcsec)
    d_radial = np.sqrt(x**2 + y**2)

    off_disk = d_radial >= (im_map.rsun_obs * edge)
    on_disk = d_radial < (im_map.rsun_obs * edge)

    assert np.all(cos_cor >= 0), "cos_cor lower limits incorrect"
    assert np.all(cos_cor <= 1 / np.cos(np.arcsin(edge))), "cos_cor upper limits incorrect"

    assert np.all(d_angular >= np.arcsin(-1) * u.rad), "d_angular lower limits incorrect"
    assert np.all(d_angular <= np.arcsin(1) * u.rad), "d_angular upper limist incorrect"

    assert np.all(off_limb[off_disk] == 0), "not all off_disk values = 0"
    assert np.all(off_limb[on_disk] == 1), "not all on_disk values = 1"


def test_cosine_correction(im_map):
    coordinates = all_coordinates_from_map(im_map)

    los_radial = np.cos(coordinates.Tx.to(u.rad)) * np.cos(coordinates.Ty.to(u.rad))
    im_map.data[:, :] = los_radial

    fake_map = Map(los_radial, im_map.meta)
    fake_cosmap = np.ones((len(los_radial), len(los_radial)))

    cosmap, corrected_data = cosine_correction(fake_map, fake_cosmap)
    corrected_data_value = corrected_data.to_value(u.Gauss)
    assert np.allclose(corrected_data_value, 1, atol=1e-4), "cosine corrected data not behaving as expected"


def test_smooth_los_threshold():
    under_thresh = create_fake_map(1)
    over_thresh = create_fake_map(1000)

    smooth_under, fl_under, mask_under = smooth_los_threshold(under_thresh, thresh=500 * u.Gauss)
    smooth_over, fl_over, mask_over = smooth_los_threshold(over_thresh, thresh=500 * u.Gauss)

    assert isinstance(smooth_under, type(under_thresh)), "smooth_under is no longer a Map"
    assert isinstance(smooth_over, type(over_thresh)), "smooth_over is no longer a Map"

    assert np.sum(fl_under) == 0, "fl should all be False when all data is below threshold"
    assert np.sum(fl_over) == len(
        fl_over.flatten()
    ), "fl should all be True when all data is above threshold "

    assert np.sum(mask_under) == 0, "no regions should have been detected in 'under_thresh'"
    assert np.sum(mask_over) > 0, "background region should have been detected in 'over thresh'"
